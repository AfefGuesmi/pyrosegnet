# Training loop for PyroSegNet.
import time
from pathlib import Path
import numpy as np
import torch
from pyrosegnet.models.losses import BCEDiceLoss
from pyrosegnet.evaluation.metrics import (
    confusion_from_pred_mask, metrics_from_confusion,
)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for batch in loader:
        pre = batch["pre"].to(device, non_blocking=True) / 255.0
        post = batch["post"].to(device, non_blocking=True) / 255.0
        mask = batch["mask"].to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(pre, post)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(pre, post)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * pre.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    TP = FP = TN = FN = 0
    for batch in loader:
        pre = batch["pre"].to(device, non_blocking=True) / 255.0
        post = batch["post"].to(device, non_blocking=True) / 255.0
        mask = batch["mask"].to(device, non_blocking=True).unsqueeze(1)
        logits = model(pre, post)
        loss = criterion(logits, mask)
        total_loss += loss.item() * pre.size(0)
        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int8)
        tgt = mask.cpu().numpy().astype(np.int8)
        for i in range(pred.shape[0]):
            tp, fp, tn, fn = confusion_from_pred_mask(pred[i, 0], tgt[i, 0])
            TP += tp; FP += fp; TN += tn; FN += fn
    metrics = metrics_from_confusion(TP, FP, TN, FN)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def fit(model, train_loader, val_loader, config, device,
        checkpoint_path, max_epochs=None, patience=None, verbose=True):
    max_epochs = max_epochs or config["training"]["num_epochs"]
    patience = patience or config["training"]["patience"]

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs,
    )
    pos_weight = torch.tensor([config["training"]["pos_weight"]]).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5, pos_weight=pos_weight)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": []}
    best_iou = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                      criterion, device, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_f1"].append(val_metrics["f1"])

        if verbose:
            print(f"Epoch {epoch:3d}/{max_epochs} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f} | "
                  f"val_IoU={val_metrics['iou']:.4f} | "
                  f"val_F1={val_metrics['f1']:.4f} | "
                  f"{time.time()-t0:.1f}s")

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, checkpoint_path)
            if verbose:
                print(f"  -> new best IoU={best_iou:.4f}, saved to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping after {epoch} epochs "
                          f"(best epoch {best_epoch}, best IoU {best_iou:.4f})")
                break

    return history, best_iou, best_epoch
