import csv
import random
from pathlib import Path

import numpy as np


def select_device(device_preference: str):                                               # Vælger hvilken device vi skal køre på, pc eller hvad?
    import torch

    pref = (device_preference or "auto").lower()                                         # Nomraliserer input
    if pref == "auto":                                                                   # Laver så den selv vælger det bedste.
        if torch.backends.mps.is_available():                                            # Vælger mps hvis det er der
            return torch.device("mps")
        if torch.cuda.is_available():                                                    # vælger cuda hvis den kan.
            return torch.device("cuda")
        return torch.device("cpu")                                                       # hvis ingen af de andre så kører den på CPU

    if pref == "mps":                                                                    # Hvis vi gerne vil vælge selv så kan vi vælge mps
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")                                                       # hvis ingen af de andre så kører den på CPU

    if pref == "cuda":                                                                   # hvis cuda så cuda.
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")                                                       # hvis ingen af de andre så kører den på CPU

    return torch.device("cpu")                                                           # hvis ingen af de andre så kører den på CPU


def set_seed(seed: int) -> None:                                                         # Sørger for at seed er det samme så vi kan replicate runs.
    import torch

    random.seed(seed)                                                                    # Random seed til python random module
    np.random.seed(seed)                                                                 # For numpy
    torch.manual_seed(seed)                                                              # For PyTorch
    if torch.cuda.is_available():                                                        # Hvis vi bruger CUDA GPU...
        torch.cuda.manual_seed_all(seed)                                                 # Så seed dertil også.


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None: # For at skrive lsite af dicts til csv fil
    path.parent.mkdir(parents=True, exist_ok=True)                                       # Hvis output mappe ikke allerede er oprettet.
    with path.open("w", newline="", encoding="utf-8") as f:                              # åbner filen til skrivning i den
        writer = csv.DictWriter(f, fieldnames=fieldnames)                                # Writer til CSV.
        writer.writeheader()                                                             # Laver en header række
        for row in rows:                                                                 # går gennem alle rækker i data
            writer.writerow(row)                                                         # Hver dictionary bliver en række.
