import torch


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-8) -> torch.Tensor: # For at lave sikker division... nok en meget god ide haha - gal de thar givet nogle fejl...
    return num / (den + eps) # Vi dividere tæller med nævner men smider en lille værdi oven i nævner så den ikker er 0


def batch_confusion(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: # Til at beregne confusion matrix (TP, FP, TN, FN). Dette gøres for hver batch
    pred_b = pred.bool() # Prediction tensor laves til bool (altså true hvis pixel lig 1)
    target_b = target.bool() # Laver så vores target mask til bool

    tp = (pred_b & target_b).sum(dim=(1, 2, 3)).float() # TP - prediction og target amsh var true
    fp = (pred_b & ~target_b).sum(dim=(1, 2, 3)).float() # FP - prediction true men target var false
    tn = (~pred_b & ~target_b).sum(dim=(1, 2, 3)).float() # TN - begge false
    fn = (~pred_b & target_b).sum(dim=(1, 2, 3)).float() # FN - gætter false men er true.
    return tp, fp, tn, fn 


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor: # Beregner Dice coefficient (det er vist ret stort i segmentation)
    tp, fp, _, fn = batch_confusion(pred, target) # Vi skal kun bruge TP, FP og FN
    return _safe_div(2 * tp, 2 * tp + fp + fn + eps) # Dice = 2TP / (2TP + FP + FN) altså (overlap mellem prediction og target)


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor: # Beregn Beregner Intersection over Union (IoU)
    tp, fp, _, fn = batch_confusion(pred, target) # same...
    return _safe_div(tp, tp + fp + fn + eps)  # IoU = TP / (TP + FP + FN) - altså hvor stor del af unionen mellem prediction og target der overlapper


def precision_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor: # beregn precision - altså hvor mange positive predictions er korrekte...
    tp, fp, _, _ = batch_confusion(pred, target) # den her gang kun TP og FP
    return _safe_div(tp, tp + fp + eps) # Precision = TP / (TP + FP) 


def recall_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor: # Regner recall - altså hvor amnge rigtige opjekter modellen finder.
    tp, _, _, fn = batch_confusion(pred, target) # kun TP og FN vi skal bruge
    return _safe_div(tp, tp + fn + eps) # Recall = TP / (TP + FN) 
