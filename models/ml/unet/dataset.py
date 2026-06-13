import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


@dataclass # Det her gør klassen til en simpel data container...
class SegmentationRow: 
    image_path: Path # Input billed sti
    mask_path: Path | None # Segmentation mask sti
    key: str # Unik nøgle til billedet
    basename: str # Basisnavn på img
    port_id: str # ID for havnen
    patch_id: str # Path ID hvos der er flere
    timestamp: str # Tidstempel for bileldet


def read_rows(csv_path: Path) -> list[SegmentationRow]: # Funktion til at læse CSV fil og lave rækker til objekter 
    rows: list[SegmentationRow] = [] # Tom liste med dataset rækker
    with csv_path.open("r", newline="", encoding="utf-8") as f: # Åbner csv filen
        reader = csv.DictReader(f) # læser havd den siger som en dictionary
        for row in reader: # Itererer alle rækker i vores CSV
            image_path = Path(row["image_path"]).resolve() # Konverterer image path til en absolut path objekt
            mask_raw = row.get("mask_path", "").strip() # henter så mask path hvis den er der
            mask_path = Path(mask_raw).resolve() if mask_raw else None # Hvis dene r så koverter til path 
            rows.append( # Tilføjer en segmentation Row til listen
                SegmentationRow(
                    image_path=image_path, 
                    mask_path=mask_path,
                    key=row.get("key", ""),
                    basename=row.get("basename", image_path.stem),
                    port_id=row.get("port_id", "unknown"),
                    patch_id=row.get("patch_id", ""),
                    timestamp=row.get("timestamp", ""),
                )
            )
    return rows # Returnerer listen med alle dataset rækker


def _is_binary_mask(mask: np.ndarray) -> bool: # tjek om mask indeholder andet end 0 eller 1.
    uniq = np.unique(mask) # Finder alle unikke værdier i masken
    valid = {0, 1, 255} # De værdier vi vil gedkende
    return all(int(v) in valid for v in uniq) # True hvis det funker.


def _apply_augment(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]: # AUgmentatioooooooon
    if random.random() < 0.5: # 50p for at vi flipper hori
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) #flip bileld
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT) # flip mask samme
    if random.random() < 0.5: # 50p for flip vert
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM) #flip bileld
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)# flip mask samme

    if random.random() < 0.5: # 50p for rotation
        angle = random.uniform(-10.0, 10.0) # med random rotations vinkel
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0)) # bilinear bruges igen - laver en glat rotation.
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0) # Til mask bruger vi nearest neighbour da vi gerne vil have en ren værdi og ikke interpolation

    return image, mask # return det nye transformerede


def _apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
    gamma = max(float(gamma), 1e-6)
    inv_gamma = 1.0 / gamma
    lut = [int(round(((idx / 255.0) ** inv_gamma) * 255.0)) for idx in range(256)]
    return image.point(lut * 3)


def _apply_photometric_augment(image: Image.Image) -> Image.Image:
    if random.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.8:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.25))
    if random.random() < 0.5:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.2))
    if random.random() < 0.5:
        image = _apply_gamma(image, random.uniform(0.85, 1.15))
    if random.random() < 0.2:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.9)))
    return image


def _to_tensor_image(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def _to_tensor_mask(mask: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask, dtype=np.uint8)
    arr_bin = (arr > 127).astype(np.float32)
    return torch.from_numpy(arr_bin[None, ...])


class SegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    def __init__(
        self,
        csv_path: Path,
        img_size: int = 512,
        augment: bool = False,
        photo_augment: bool = False,
        require_mask: bool = True,
        strict_binary_masks: bool = True,
        strict_shape_check: bool = True,
    ):
        self.csv_path = csv_path.resolve()
        self.rows = read_rows(self.csv_path)
        self.img_size = int(img_size)
        self.augment = augment
        self.photo_augment = photo_augment
        self.require_mask = require_mask
        self.strict_binary_masks = strict_binary_masks
        self.strict_shape_check = strict_shape_check

        if not self.rows:
            raise ValueError(f"Dataset CSV has no rows: {self.csv_path}")

        for row in self.rows:
            if not row.image_path.exists():
                raise FileNotFoundError(f"Image file not found: {row.image_path}")
            if self.require_mask and row.mask_path is None:
                raise ValueError(f"Missing mask path in row for image: {row.image_path}")
            if self.require_mask and row.mask_path is not None and not row.mask_path.exists():
                raise FileNotFoundError(f"Mask file not found: {row.mask_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.rows[idx]
        image = Image.open(row.image_path).convert("RGB")

        if row.mask_path is not None:
            mask = Image.open(row.mask_path).convert("L")
        else:
            mask = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8), mode="L")

        if self.strict_shape_check and image.size != mask.size:
            raise ValueError(
                f"Image/mask size mismatch for {row.image_path.name}: image={image.size}, mask={mask.size}"
            )

        if self.strict_binary_masks and row.mask_path is not None:
            mask_arr = np.asarray(mask, dtype=np.uint8)
            if not _is_binary_mask(mask_arr):
                uniq = np.unique(mask_arr)
                raise ValueError(
                    f"Mask is not binary (0/255 or 0/1) for {row.mask_path}. Unique values: {uniq[:20]}"
                )

        if image.size != (self.img_size, self.img_size):
            image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)

        if self.augment:
            image, mask = _apply_augment(image, mask)
        if self.photo_augment:
            image = _apply_photometric_augment(image)

        image_tensor = _to_tensor_image(image)
        mask_tensor = _to_tensor_mask(mask)

        meta = {
            "image_path": str(row.image_path),
            "mask_path": str(row.mask_path) if row.mask_path else "",
            "key": row.key,
            "basename": row.basename,
            "port_id": row.port_id,
            "patch_id": row.patch_id,
            "timestamp": row.timestamp,
        }
        return image_tensor, mask_tensor, meta
