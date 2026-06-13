# U-Net Segmentering

## Kontrakt

- `image`: `<basename>.png`
- `mask`: `<basename>_mask.png`
- Patch-navne (`P1`, `P2`, ...) skal indgå i samme basename for både image/mask.
- Masker skal være binære (`0/255` eller `0/1`).

## Kommandoer

```bash
python -m models.ml.unet.train --train-csv <train.csv> --val-csv <val.csv> --img-size 512 --batch-size 1 --lr 1e-4 --loss bce --epochs 30 --device mps
python -m models.ml.unet.infer --checkpoint <best.pt> --input-csv <test.csv> --out <out_dir> --threshold 0.5
python -m models.ml.unet.report --run-dir "<repo>/data/outputs/segmentation/runs/<run_id>"
```

Eksempel på eksperiment uden at ændre baseline-adfærden:

```bash
python -m models.ml.unet.train --train-csv <train.csv> --val-csv <val.csv> --img-size 512 --batch-size 1 --lr 1e-4 --loss bce_dice --bce-weight 1.0 --dice-weight 1.0 --norm group --group-norm-groups 8 --val-thresholds 0.2,0.3,0.4,0.5,0.6 --epochs 30 --device cuda
python -m models.ml.unet.infer --checkpoint <best.pt> --input-csv <val.csv> --out <out_dir> --threshold 0.5 --threshold-sweep 0.2,0.3,0.4,0.5,0.6
```

Fotometrisk augmentation er opt-in og påvirker kun træningsbillederne, ikke masks eller validation:

```bash
python -m models.ml.unet.train --train-csv <train.csv> --val-csv <val.csv> --img-size 512 --batch-size 1 --lr 1e-4 --loss bce_dice --bce-weight 1.0 --dice-weight 1.0 --norm group --group-norm-groups 8 --val-thresholds 0.2,0.3,0.4,0.5,0.6 --epochs 30 --device cuda --photo-augment
```

Resume af et tidligere run fra `last.pt` eller `best.pt`:

```bash
python -m models.ml.unet.train --train-csv <train.csv> --val-csv <val.csv> --img-size 512 --batch-size 1 --lr 1e-4 --loss bce_dice --bce-weight 1.0 --dice-weight 1.0 --norm group --group-norm-groups 8 --val-thresholds 0.2,0.3,0.4,0.5,0.6 --epochs 100 --device cuda --photo-augment --resume-checkpoint <run_dir>/checkpoints/last.pt --out-dir <same_or_new_out_root> --run-id <new_or_existing_run_id>
```

Hvis du vil fortsætte i samme run-mappe, så brug samme `--out-dir` og `--run-id`. Hvis du vil bevare det gamle run urørt, så giv et nyt `--run-id`.

Isoleret enkelt-eksperiment med samme outputs som overnight-suiten:

```bash
python -m models.ml.unet.run_single_experiment --run-id 04_bce_dice_group_photoaug_test --train-csv <train.csv> --val-csv <val.csv> --out-root <out_root> --img-size 512 --batch-size 1 --lr 1e-4 --loss bce_dice --bce-weight 1.0 --dice-weight 1.0 --epochs 30 --threshold 0.5 --val-thresholds 0.2,0.3,0.4,0.5,0.6 --device cuda --augment --photo-augment --num-workers 4 --norm group --group-norm-groups 8 --visual-sample-size 12 --report-max-samples 12
```

## Overnight Suite

```bash
python -m models.ml.unet.overnight_suite --train-csv <train.csv> --val-csv <val.csv> --suite-dir <suite_dir> --img-size 512 --batch-size 1 --epochs 30 --threshold 0.5 --val-thresholds 0.2,0.3,0.4,0.5,0.6 --device cuda --num-workers 4
```

Det kører 8 eksperimenter sekventielt og gemmer for hvert run:
- træningsoutput, checkpoints og plots
- inferens på val-sættet med threshold-sweep
- diverse visual checks med probability-maps og overlays
- HTML-report pr. run
- samlet `suite_summary.csv`, `suite_leaderboard.csv` og `suite_summary.md`

## ROI Postprocessing

ROI er opt-in og påvirker kun inferens/postprocessing. Uden ROI-flags er inferens-output uændret.

`--roi-patch-bboxes-path` kan pege på:
- en enkelt `patch_bboxes_*.txt`
- en `manifest.csv` med `bbox_minlon/...`
- en mappe, som så merges deterministisk over alle `patch_bboxes_*.txt` og relevante `manifest.csv`

Default ROI-kilden i koden er den kanoniske LA/LB-fil:
`data/inputs/patch_bboxes_final_49ports_lalb_20260527.txt`.

Eksempel på ROI-inferens:

```bash
python -m models.ml.unet.infer --checkpoint <best.pt> --input-csv <val.csv> --out <roi_out_dir> --threshold 0.5 --threshold-sweep 0.2,0.3,0.4,0.5,0.6 --apply-roi --roi-polygons-path <repo>/data/inputs/Havne_koor.txt --roi-patch-bboxes-path <repo>/data/inputs/patch_bboxes_final_49ports_lalb_20260527.txt --roi-buffer-px 10
```

Det gemmer bl.a.:
- `predictions.csv` med ROI-felter og pre/post-metrics
- `roi_summary.csv`
- `roi_summary.json`
- `roi_port_map_used.json`
- `pred_masks_pre_roi/`
- `pred_masks/` som final post-ROI prediction
- `roi_masks/`
- `prob_maps_pre_roi/` og `prob_maps_post_roi/`

Fuld visuel review-pakke for alle val-samples:

```bash
python -m models.ml.unet.build_roi_review --predictions-csv <roi_out_dir>/predictions.csv --out-dir <roi_review_dir>
```

Review-pakken organiseres som:
- `cases.csv`
- `port_summary.csv`
- `port_patch_summary.csv`
- `missing_roi_ports.csv`
- `ports/<port_slug>/README.md`
- `ports/<port_slug>/<patch_id>/panels/`
- `ports/<port_slug>/<patch_id>/roi_masks/`
- `ports/<port_slug>/<patch_id>/roi_overlays/`
- `ports/<port_slug>/<patch_id>/pred_pre_roi/`
- `ports/<port_slug>/<patch_id>/pred_post_roi/`
- `ports/<port_slug>/overview_pages/`
- `ports/<port_slug>/<patch_id>/overview_pages/`
