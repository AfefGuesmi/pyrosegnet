# Train PyroSegNet on CaBuAr. Works on CPU (smoke) and GPU (full).
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import pandas as pd
from torch.utils.data import DataLoader

from pyrosegnet.config import load_config
from pyrosegnet.data.splits import get_split_samples
from pyrosegnet.data.dataset import CaBuArDataset
from pyrosegnet.models.pyrosegnet import PyroSegNet
from pyrosegnet.training.train import fit, evaluate
from pyrosegnet.models.losses import BCEDiceLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Tiny CPU smoke test (2 samples, 2 epochs)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers

    hdf5_path = config["data"]["hdf5_file"]
    ckpt_dir = Path(config["outputs"]["checkpoint_dir"])
    result_dir = Path(config["outputs"]["result_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading splits...")
    splits = get_split_samples(hdf5_path)

    if args.smoke:
        print("*** SMOKE MODE ***")
        splits["train"] = splits["train"][:2]
        splits["val"] = splits["val"][:2]
        splits["test"] = splits["test"][:2]
        config["training"]["batch_size"] = 1
        config["training"]["num_epochs"] = 2
        config["training"]["patience"] = 2
        config["training"]["num_workers"] = 0

    print(f"  train: {len(splits['train'])} samples")
    print(f"  val:   {len(splits['val'])} samples")
    print(f"  test:  {len(splits['test'])} samples")

    bs = config["training"]["batch_size"]
    nw = config["training"]["num_workers"]
    train_ds = CaBuArDataset(hdf5_path, splits["train"])
    val_ds = CaBuArDataset(hdf5_path, splits["val"])
    test_ds = CaBuArDataset(hdf5_path, splits["test"])

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=(device.type == "cuda"))

    print("\nBuilding model...")
    model = PyroSegNet(in_channels=12, out_channels=1, base_ch=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    ckpt_path = str(ckpt_dir / "pyrosegnet_best.pth")
    epochs = args.epochs or config["training"]["num_epochs"]

    print(f"\nTraining for up to {epochs} epochs...")
    history, best_iou, best_epoch = fit(
        model, train_loader, val_loader, config, device,
        checkpoint_path=ckpt_path, max_epochs=epochs,
    )

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(result_dir / "pyrosegnet_training_history.csv", index=False)

    print(f"\nLoading best checkpoint (epoch {best_epoch}, val IoU={best_iou:.4f})...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pos_weight = torch.tensor([config["training"]["pos_weight"]]).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5, pos_weight=pos_weight)
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\n=== TEST SET RESULTS (PyroSegNet) ===")
    print(f"  Precision (burned): {test_metrics['precision']:.4f}")
    print(f"  Recall (burned):    {test_metrics['recall']:.4f}")
    print(f"  F1 (burned):        {test_metrics['f1']:.4f}")
    print(f"  IoU (burned):       {test_metrics['iou']:.4f}")
    print(f"  Dice (burned):      {test_metrics['dice']:.4f}")
    print(f"  Accuracy:           {test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}")

    row = {"model": "PyroSegNet", "best_val_iou": best_iou,
           "best_epoch": best_epoch, **test_metrics}
    pd.DataFrame([row]).to_csv(result_dir / "pyrosegnet_test_metrics.csv", index=False)
    print(f"\nCheckpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
