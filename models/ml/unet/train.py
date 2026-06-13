import argparse                                                                          # Læse parametere fra kommandolinjen
import csv
import json                                                                              # Gemme konfigurationen efter træning er done
from datetime import datetime, timezone                                                  # Hive tid og dato ned
from pathlib import Path                                                                 # Så vi kan lave robuste filstier og ikke behøver at hardcode.
from typing import Any                                                                   # Bruges som fleksibel type når vi ikke låser typen endnu.

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_ROOT = REPO_ROOT / "data" / "outputs" / "segmentation" / "runs"

def _soft_dice_loss_from_logits(logits: Any, targets: Any, eps: float = 1e-8) -> Any:
    import torch

    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def _loss_from_logits(logits: Any,                                                       # Hjælpefunktion som beregner loss ud fra den valgte loss type. Dette skal bruges i train og val, så derfor laver vi en funktion.
                      targets: Any, 
                      loss_name: str, 
                      criterion: Any
) -> Any: 
    import torch                                                                         # Vi importerer torch inde i funktionen så den kun importeres når den skal bruges. 

    if loss_name == "bce":                                                               # BCEWithLogitsLoss forventer rå logits direkte og er lavet til binære targets.
        return criterion(logits, targets)                                                # Beregner BCE loss direkte på model output og target mask.
    if loss_name == "bce_dice":
        bce_term = criterion["bce"](logits, targets)
        dice_term = _soft_dice_loss_from_logits(logits, targets)
        return criterion["bce_weight"] * bce_term + criterion["dice_weight"] * dice_term
    probs = torch.sigmoid(logits)                                                        # Hvis vi vælger MSE som fallback laver vi logits om til sandsynligheder først.
    return criterion(probs, targets)                                                     # Beregner så MSE mellem sandsynlighederne og target mask.


def _parse_threshold_list(value: str) -> list[float]:
    thresholds: list[float] = []
    for raw in (value or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        threshold = float(raw)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got: {threshold}")
        thresholds.append(threshold)
    return thresholds


def _threshold_metric_key(prefix: str, threshold: float) -> str:
    return f"{prefix}_{threshold:.2f}".replace(".", "_")


def _parse_history_rows(metrics_csv: Path) -> list[dict[str, float]]:
    if not metrics_csv.exists():
        return []

    rows: list[dict[str, float]] = []
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row: dict[str, float] = {}
            for key, value in raw_row.items():
                if value is None or value == "":
                    continue
                if key == "epoch":
                    row[key] = int(float(value))
                else:
                    row[key] = float(value)
            if row:
                rows.append(row)
    return rows


def _load_resume_bundle(
    resume_checkpoint: Path,
    device: Any,
    args: argparse.Namespace,
    model: Any,
    optimizer: Any,
) -> tuple[int, float, float, list[dict[str, float]], dict[str, Any]]:
    import torch

    ckpt = torch.load(resume_checkpoint, map_location=device)
    if "model_state_dict" not in ckpt:
        raise ValueError(f"Resume checkpoint is missing model_state_dict: {resume_checkpoint}")

    checkpoint_norm = str(ckpt.get("norm_type", args.norm))
    checkpoint_groups = int(ckpt.get("group_norm_groups", args.group_norm_groups))
    if checkpoint_norm != args.norm or checkpoint_groups != args.group_norm_groups:
        raise ValueError(
            "Resume checkpoint architecture does not match requested model config: "
            f"checkpoint norm={checkpoint_norm}, groups={checkpoint_groups}, "
            f"requested norm={args.norm}, groups={args.group_norm_groups}"
        )

    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    checkpoint_epoch = int(ckpt.get("epoch", 0))
    start_epoch = checkpoint_epoch + 1
    resume_run_dir = resume_checkpoint.resolve().parent.parent
    history = _parse_history_rows(resume_run_dir / "metrics.csv")

    best_dice = float(ckpt.get("best_dice", float("-inf")))
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    if history:
        best_history_row = max(
            history,
            key=lambda row: (
                float(row.get("val_dice", float("-inf"))),
                -float(row.get("val_loss", float("inf"))),
            ),
        )
        best_dice = max(best_dice, float(best_history_row.get("val_dice", float("-inf"))))
        if best_val_loss == float("inf"):
            best_val_loss = float(best_history_row.get("val_loss", float("inf")))
        elif abs(best_dice - float(best_history_row.get("val_dice", float("-inf")))) < 1e-12:
            best_val_loss = min(best_val_loss, float(best_history_row.get("val_loss", float("inf"))))

    resume_info = {
        "resume_checkpoint": str(resume_checkpoint.resolve()),
        "resume_run_dir": str(resume_run_dir),
        "resume_checkpoint_epoch": checkpoint_epoch,
        "resume_start_epoch": start_epoch,
        "resume_history_rows": len(history),
    }
    return start_epoch, best_dice, best_val_loss, history, resume_info


def evaluate(                                                                            # Funktion til validering
    model: Any,                                                                          # Model objektet (typisk en torch nn.Module).
    loader: Any,                                                                         # TIl at load data for validation
    loss_name: str,                                                                      # Til at beregne den rigtige loss med vores hjælpefunktion fra før (BCE bruges som default).
    criterion: Any,                                                                      # Selve loss funktionen (kunne være nn.BCEWithLogitsLoss()). Ja det er her den kaldes.
    device: Any,                                                                         # Hvor tensors og model skal ligge: cpu, cuda eller mps.
    threshold: float,                                                                    # Den threshold vi vil bruge til at lave binær maske ud fra vores probs.
    sweep_thresholds: list[float] | None = None,
) -> dict[str, float]:                                                                   # Out put er et dictionary med metrics i.
    import torch                                                                         # Kald pytorch lokalt så den kun hentes når den skal bruges. Metrics er eval mål som dice, IoU, precision og recall.
    from models.ml.unet.metrics import dice_score, iou_score, precision_score, recall_score # Import af metric funktioner

    model.eval()                                                                         # Sætter modellen i eval mode (fx dropout fra og stabil batchnorm adfærd).
    losses: list[float] = []                                                             # Samler loss pr. batch for gennemsnit.
    dices: list[float] = []                                                              # Samler dice scores ind. Dice er et overlap mål mellem prediction og maske.
    ious: list[float] = []                                                               # Samler IoU. IoU er intersection over union (et overlap mål).
    precisions: list[float] = []                                                         # Opsamler precision
    recalls: list[float] = []                                                            # Opsamler recall
    sweep_thresholds = list(dict.fromkeys(sweep_thresholds or []))
    threshold_dices: dict[float, list[float]] = {thr: [] for thr in sweep_thresholds}

    with torch.no_grad():                                                                # Vi tracker ikke gradienterne da de tager hukommelse og tid... kan slås til senere. 
        for images, masks, _ in loader:                                                  # Itererer baches... vi kigger på image, mask og så ignirere vi det tredje input fra datasetet (tror det var basename). basename er ikke vigtig i eval. 
            images = images.to(device)                                                   # Flytter input hen i det korrekte device så data og model matcher.
            masks = masks.to(device)                                                     # Flytter mask hen i det korrekte device så beregninger kan køres sammen.

            logits = model(images)                                                       # Det her er en forward pass: input gennem model og rå output (logits) pr. pixel.
            loss = _loss_from_logits(logits, masks, loss_name, criterion)                # Beregner loss med vores hjælpe funktion
            probs = torch.sigmoid(logits)                                                # Laver vores probs med sigmoid
            preds = (probs >= threshold).float()                                         # Lav en binær maske, 0.0 eller 1.0 som tensor. 

            losses.append(float(loss.item()))                                            # Vi gemmer loss som en float. 
            dices.append(float(dice_score(preds, masks).mean().item()))                  # Vi lave dice om til en mean pr. batch så vi ikke har per element.
            ious.append(float(iou_score(preds, masks).mean().item()))                    # Samme her for IoU
            precisions.append(float(precision_score(preds, masks).mean().item()))        # SAme for precision
            recalls.append(float(recall_score(preds, masks).mean().item()))              # Same for recall
            for thr in sweep_thresholds:
                thr_preds = (probs >= thr).float()
                threshold_dices[thr].append(float(dice_score(thr_preds, masks).mean().item()))

    def _mean(values: list[float]) -> float:                                             # lille hjælpe funktion så vi ike gør det samme 5 gange. 
        return float(sum(values) / max(len(values), 1))                                  # Gennemsnit - beskyttet mod div med nul.

    metrics = {
        "val_loss": _mean(losses),                                                       # Regn og print loss
        "val_dice": _mean(dices),                                                        # Regn og print dice
        "val_iou": _mean(ious),                                                          # Regn og print  IoU
        "val_precision": _mean(precisions),                                              # Regn og print precision
        "val_recall": _mean(recalls),                                                    # Regn og print recall
    }
    if sweep_thresholds:
        threshold_summary = {thr: _mean(vals) for thr, vals in threshold_dices.items()}
        best_threshold = max(threshold_summary, key=threshold_summary.get)
        metrics["val_dice_best_threshold"] = float(best_threshold)
        metrics["val_dice_best"] = float(threshold_summary[best_threshold])
        for thr, score in threshold_summary.items():
            metrics[_threshold_metric_key("val_dice_thr", thr)] = float(score)
    return metrics


def train_model(args: argparse.Namespace) -> dict[str, str]:                             # vores main funktion til træningen
    import matplotlib.pyplot as plt                                                      # Plot loss og dice
    import torch                                                                         # Træning og saving
    from torch import nn                                                                 # loss funktioner f.eks. nn.BCEWithLogitsLoss()
    from torch.optim import Adam                                                         # Optimizer klasse som opdaterer modelens vægte ud fra gradienter.
    from torch.utils.data import DataLoader                                              # Til batching og parallel loading (flere workers kan loade data samtidigt).
 
    from models.ml.unet.dataset import SegmentationDataset                               # Projektets dataset wrapper (læser/klargør data i det format modellen forventer).
    from models.ml.unet.model import UNet                                                # Henter UNET arkitekturen fra model.py
    from models.ml.unet.utils import select_device, set_seed, write_csv                  # Hjælpe funktioner til device, seet og csv output (hentes fra utils.py)

    set_seed(args.seed)                                                                  # Gør at vi kan reproduce det vi laver

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")         # Vi spørger om et run id, ellers laves et tidsstempel i UTC så hvert run får en unik mappe.
    out_dir = Path(args.out_dir).resolve() / run_id                                      # Output mappen.
    checkpoints_dir = out_dir / "checkpoints"                                            # Mappe til modellen vægte
    plots_dir = out_dir / "plots"                                                        # Mappe til modellens plots
    out_dir.mkdir(parents=True, exist_ok=True)                                           # Opretter output mappe
    checkpoints_dir.mkdir(parents=True, exist_ok=True)                                   # Opretter checkpoints mappe (gemte model-tilstande under træningen).
    plots_dir.mkdir(parents=True, exist_ok=True)                                         # opretter plots mappen

    device = select_device(args.device)                                                  # Vælger device ud fra hvad vi vil have. Ja det er hardwaren på maskinen (cpu/gpu/mps).
    print(f"Using device: {device}")

    train_dataset = SegmentationDataset(                                                 # Laver dataset objekt for træning
        csv_path=Path(args.train_csv),                                                   # Trænings data beskrives af en csv fil med stier til image/mask (og evt id). 
        img_size=args.img_size,                                                          # Laver resize hvis billederne ikke er 512x512
        augment=args.augment,                                                            # Augmentation er ekstra variation af data, og her bruges det i train hvis flag er slået til.
        photo_augment=args.photo_augment,                                                # Ekstra fotometrisk augmentation er opt-in, så baseline-adfærd bevares.
        require_mask=True,                                                               # Vi skal bruge en maske for at træne. Ja den markerer target (fx container) vs baggrund.
        strict_binary_masks=True,                                                        # Sørger for at masken kun indehodler 0 og 1 eller 0 og 255
        strict_shape_check=True,                                                         # Sørger for at image og mask passer i dimensioner.
    )
    val_dataset = SegmentationDataset(                                                   # same som train_dataset men bare for val...
        csv_path=Path(args.val_csv),
        img_size=args.img_size,
        augment=False,                                                                   # False her så val bliver stabil og fair at sammenligne mellem epochs.
        photo_augment=False,                                                             # Vi holder validation ren og deterministisk.
        require_mask=True,
        strict_binary_masks=True,
        strict_shape_check=True,
    )

    train_loader = DataLoader(                                                           # opretter loader som så laver batches
        train_dataset,                                                                   # Bruger vores train dataset (sættes senere, men lavet så vi kan teste felre ting i terminalen "--batch-size 4")
        batch_size=args.batch_size,                                                      # Batch sætter vores batch size 
        shuffle=True,                                                                    # Shuffle training så vi reducerer bias fra rækkefølge. 
        num_workers=args.num_workers,                                                    # Parallel data loading: antal worker processer der loader data samtidig.
    )
    val_loader = DataLoader(                                                             #Samme som train_loader
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,                                                                   # Ingen shuffle da det er eval, så målingen bliver stabil og reproducerbar.
        num_workers=args.num_workers,
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        norm_type=args.norm,
        group_norm_groups=args.group_norm_groups,
    ).to(device)                                                                         # Laver UNET. 3 input kanaler (RGB) 1 output kanal (Binær maske). Flyttes til valgt device.

    loss_name = args.loss.lower()                                                        # Normaliserer inputtet med lower(), fx BCE -> bce.
    if loss_name == "bce":                                                               # Tjek om vi bruger BCE som er vores standard loss til binær segmentering.
        criterion: Any = nn.BCEWithLogitsLoss()                                          # BCEWithLogitsLoss er stabil fordi sigmoid indgår inde i loss funktionen.
    elif loss_name == "bce_dice":
        criterion = {
            "bce": nn.BCEWithLogitsLoss(),
            "bce_weight": float(args.bce_weight),
            "dice_weight": float(args.dice_weight),
        }
    elif loss_name == "mse":                                                             # Vi beholder MSE som fallback hvis vi vil sammenligne med gamle runs.
        criterion = nn.MSELoss()                                                         # MSE bruger vi kun når vi bevidst vil køre den ældre opsætning.
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")                               # Hvis ukendt så stopper vi tydeligt i stedet for at træne forkert.

    optimizer = Adam(model.parameters(), lr=args.lr)                                     # Adam optimizer med learning rate 
    val_thresholds = _parse_threshold_list(args.val_thresholds)

    history: list[dict[str, float]] = []                                                 # gemmer de metris der blev brugt
    best_dice = float("-inf")                                                            # best dice starter lav så næste epoch altid kan forbedre det
    best_val_loss = float("inf")                                                         # best val loss høj for at starte
    start_epoch = 1
    resume_info: dict[str, Any] = {}

    resume_checkpoint = str(getattr(args, "resume_checkpoint", "") or "")

    if resume_checkpoint:
        (
            start_epoch,
            best_dice,
            best_val_loss,
            history,
            resume_info,
        ) = _load_resume_bundle(
            resume_checkpoint=Path(resume_checkpoint),
            device=device,
            args=args,
            model=model,
            optimizer=optimizer,
        )
        if start_epoch > args.epochs:
            raise ValueError(
                f"Resume checkpoint already reached epoch {start_epoch - 1}, "
                f"but requested epochs={args.epochs}. Increase --epochs to continue."
            )
        print(
            "Resuming training from "
            f"{resume_info['resume_checkpoint']} at epoch {resume_info['resume_start_epoch']} "
            f"(loaded {resume_info['resume_history_rows']} previous metric rows)"
        )

    for epoch in range(start_epoch, args.epochs + 1):                                    # Epoch starter fra 1 så vi nemmere kan logge.
        model.train()                                                                    # laver modelens mode til train
        train_losses: list[float] = []                                                   # Samler batch losses 

        for images, masks, _ in train_loader:                                            # Itererer gennem alle billeder og masks. ignorerer sidste index som var sådan id. 
            images = images.to(device)                                                   # Flytter images til vores device
            masks = masks.to(device)                                                     # Flytter masks til vores device

            optimizer.zero_grad(set_to_none=True)                                        # Nulstiller gradienter (kan også sætte til 0)
            logits = model(images)                                                       # Forward pass = vi sender batchen gennem modellen og får output.
            loss = _loss_from_logits(logits, masks, loss_name, criterion)                # BEregner loss 
            loss.backward()                                                              #Backpropagation - regner gradienter for hver parameter.
            optimizer.step()                                                             # Opdaterer parameter ud fra gradienterne.

            train_losses.append(float(loss.item()))                                      # Gemmer batch loss (loss værdien for den aktuelle minibatch).

        train_loss = float(sum(train_losses) / max(len(train_losses), 1))                # train loss for epoch
        val_metrics = evaluate(                                                          # Kalder evaluate med model, val loader, loss setup, device, threshold.
            model=model, 
            loader=val_loader, 
            loss_name=loss_name, 
            criterion=criterion,
            device=device,
            threshold=args.threshold,
            sweep_thresholds=val_thresholds,
        )

        row = {
            "epoch": epoch,                                                              # Gemmer epoch.
            "train_loss": train_loss,                                                    # Gemmer train loss.
            **val_metrics,                                                               # Merge val metrics ind.
        }
        history.append(row)                                                              # Gemmer det til senere csv og plots.

        improved = (                                                                     # Definerer kriterie for om denne epoch er bedre end den akutelle bedste
            val_metrics["val_dice"] > best_dice                                          # højere dice er bedre så det bruger vi som main criterie.
            or (
                abs(val_metrics["val_dice"] - best_dice) < 1e-12                         # hvis der er dice tie så bruger vi den med lavest loss.
                and val_metrics["val_loss"] < best_val_loss
            )
        )

        if improved:                                                                     # Hvis den er bedre så...
            best_dice = val_metrics["val_dice"]                                          # Opdater så det er registreret at den nye er bedst. 
            best_val_loss = val_metrics["val_loss"]                                      # Opdater best loss
            torch.save(                                                                  # Gem best checkpoint. Gemmer epoch, model weights, optimizer state, best metrics, img_size. Valgt så man kan resume eller reproducere inference opsætning.
                {
                    "epoch": epoch, 
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "best_val_loss": best_val_loss,
                    "img_size": args.img_size,
                    "norm_type": args.norm,
                    "group_norm_groups": args.group_norm_groups,
                    "resume_checkpoint": resume_info.get("resume_checkpoint", ""),
                },
                checkpoints_dir / "best.pt",
            )

        torch.save(
            {                                                                            # Gemmer altid last checkpoint, uanset forbedring.
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "best_val_loss": best_val_loss,
                "img_size": args.img_size,
                "norm_type": args.norm,
                "group_norm_groups": args.group_norm_groups,
                "resume_checkpoint": resume_info.get("resume_checkpoint", ""),
            },
            checkpoints_dir / "last.pt",                                                 # Gemmer sidste checkpoint
        )

        message = (
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['val_loss']:.6f} val_dice={val_metrics['val_dice']:.6f}"
        )
        if "val_dice_best_threshold" in val_metrics:
            message += (
                f" best_thr={val_metrics['val_dice_best_threshold']:.2f}"
                f" best_thr_dice={val_metrics['val_dice_best']:.6f}"
            )
        print(message)                                                                   # Udskriver epoch, train loss, val loss, val dice med 6 decimaler.

    metrics_csv = out_dir / "metrics.csv"                                                # Path til csv.
    base_fieldnames = ["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_precision", "val_recall"]
    extra_fieldnames = sorted({key for row in history for key in row.keys() if key not in base_fieldnames})
    write_csv(                                                                           # Fra utils
        metrics_csv,
        [{k: v for k, v in row.items()} for row in history],
        base_fieldnames + sorted(extra_fieldnames),
    )

    epochs = [int(row["epoch"]) for row in history]                                      # X akse.
    train_losses = [float(row["train_loss"]) for row in history]                         # Y serie.
    val_losses = [float(row["val_loss"]) for row in history]                             # Y serie.
    val_dices = [float(row["val_dice"]) for row in history]                              # Y serie.

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_dices, label="val_dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Validation Dice")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "val_dice.png", dpi=150)
    plt.close()

    config_out = {                                                                       # Bygger config dict til reproducering.
        "run_id": run_id,                                                                # Gem run id.
        "train_csv": str(Path(args.train_csv).resolve()),                                #Gem absolut sti så du kan se præcis hvilket input der blev brugt 
        "val_csv": str(Path(args.val_csv).resolve()),                                    # Samme 
        "img_size": args.img_size,                                                       # Gem
        "batch_size": args.batch_size,                                                   # Gem
        "lr": args.lr,                                                                   # Gem
        "loss": loss_name,                                                               # Gem
        "bce_weight": args.bce_weight,                                                   # Gem
        "dice_weight": args.dice_weight,                                                 # Gem
        "epochs": args.epochs,                                                           # Gem
        "threshold": args.threshold,                                                     # Gem
        "val_thresholds": val_thresholds,                                                # Gem
        "device_requested": args.device,                                                 # Gem
        "device_used": str(device),                                                      # Gem
        "seed": args.seed,                                                               # Gem# Gem# Gem# Gem# Gem# Gem# Gem# Gem
        "augment": args.augment,                                                         # Gem
        "photo_augment": args.photo_augment,                                             # Gem
        "norm": args.norm,                                                               # Gem
        "group_norm_groups": args.group_norm_groups,                                     # Gem
        "resume_checkpoint": resume_info.get("resume_checkpoint", ""),
        "resume_start_epoch": resume_info.get("resume_start_epoch", 1),
        "resumed": bool(resume_info),
    }
    (out_dir / "train_config.json").write_text(json.dumps(config_out, indent=2), encoding="utf-8") # Serialiserer config til json, pænt indrykket. Valgt så vi kan reproducere et run senere

    return {                                                                             # Returnerer output info til main
        "run_dir": str(out_dir),                                                         # Hvor alt ligger
        "best_checkpoint": str(checkpoints_dir / "best.pt"),                             # Best model fil
        "last_checkpoint": str(checkpoints_dir / "last.pt"),                             # Seneste model fil
        "metrics_csv": str(metrics_csv),                                                 # Metrics csv
    }


def parse_args() -> argparse.Namespace:                                                  # Definerer funktion der læser input argumenter fra command line og returnerer dem som Namespace objekt
    parser = argparse.ArgumentParser(description="Train U-Net on image/mask pairs.")     # Opretter argument parser med hjælpebeskrivelse til --help
    parser.add_argument("--train-csv", required=True)                                    # Sti til CSV med træningsdata; required=True betyder programmet fejler hvis den ikke gives
    parser.add_argument("--val-csv", required=True)                                      # Sti til CSV med validation data som bruges til evaluering efter hver epoch
    parser.add_argument("--img-size", type=int, default=512)                             # Input billedstørrelse som billeder resize/checkes til før de sendes i modellen
    parser.add_argument("--batch-size", type=int, default=1)                             # Antal billeder per batch; lav default fordi 512x512 segmentation bruger meget memory
    parser.add_argument("--lr", type=float, default=1e-4)                                # Learning rate for optimizeren; bestemmer hvor store gradient steps modellen tager
    parser.add_argument("--loss", choices=["bce", "mse", "bce_dice"], default="bce")    # Valg af loss funktion; BCE er default og MSE er kun med som fallback til sammenligning
    parser.add_argument("--bce-weight", type=float, default=1.0)                         # Vægt for BCE-delen i BCE+Dice loss.
    parser.add_argument("--dice-weight", type=float, default=1.0)                        # Vægt for Dice-delen i BCE+Dice loss.
    parser.add_argument("--epochs", type=int, default=30)                                # Hvor mange gange hele datasættet gennemløbes under træning
    parser.add_argument("--threshold", type=float, default=0.5)                          # Threshold til at konvertere sigmoid output til binær maske ved evaluering
    parser.add_argument("--val-thresholds", default="")                                  # Kommasepareret threshold-sweep til validation.
    parser.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")            # Hvilken hardware der bruges; auto vælger bedste tilgængelige device
    parser.add_argument("--seed", type=int, default=42)                                  # Random seed for reproducerbarhed af træning og dataloading
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)# Giver både --augment og --no-augment flags; styrer om data augmentation bruges i træning
    parser.add_argument("--photo-augment", action=argparse.BooleanOptionalAction, default=False) # Opt-in fotometrisk augmentation til robusthed mod kontrast/lys variationer.
    parser.add_argument("--num-workers", type=int, default=0)                            # Antal parallelle dataloader processer; 0 er mest stabilt på Mac/MPS
    parser.add_argument("--norm", choices=["batch", "group", "instance"], default="batch") # Normaliseringslag i U-Net.
    parser.add_argument("--group-norm-groups", type=int, default=8)                      # Antal grupper når GroupNorm bruges.
    parser.add_argument("--resume-checkpoint", default="")                               # Optional path til checkpoint hvis træningen skal fortsættes fra et tidligere run.
    parser.add_argument("--out-dir", default=str(DEFAULT_RUNS_ROOT))                     # Basis mappe hvor checkpoints, plots og metrics gemmes
    parser.add_argument("--run-id", default="")                                          # Valgfrit navn på run; hvis tom genereres timestamp automatisk
    return parser.parse_args()                                                           # Parser faktisk CLI input og returnerer Namespace med alle argumenter som attributter


def main() -> None:
    result = train_model(parse_args())
    print("Training complete")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
