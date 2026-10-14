"""Local CPU training with resized images."""
import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import pandas as pd
from torch.utils.data import DataLoader

from pyrosegnet.config import load_config
from pyrosegnet.data.splits import get_split_samples
from pyrosegnet.data.dataset import CaBuArDataset
from pyrosegnet.models.pyrosegnet import PyroSegNet
from pyrosegnet.models.losses import BCEDiceLoss
from pyrosegnet.training.train import fit, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    config = load_config()
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    config["training"]["batch_size"] = args.batch_size
    config["training"]["num_workers"] = args.num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Config: img_size=%d, epochs=%d, batch_size=%d, lr=%s" % (
        args.img_size, args.epochs, args.batch_size,
        config["training"]["learning_rate"]))

    hdf5_path = config["data"]["hdf5_file"]
    ckpt_dir = Path(config["outputs"]["checkpoint_dir"])
    result_dir = Path(config["outputs"]["result_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("Loading splits...")
    splits = get_split_samples(hdf5_path)
    print("  train:", len(splits["train"]), "samples")
    print("  val:  ", len(splits["val"]), "samples")
    print("  test: ", len(splits["test"]), "samples")

    bs = args.batch_size
    nw = args.num_workers
    train_ds = CaBuArDataset(hdf5_path, splits["train"], img_size=args.img_size)
    val_ds = CaBuArDataset(hdf5_path, splits["val"], img_size=args.img_size)
    test_ds = CaBuArDataset(hdf5_path, splits["test"], img_size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=False)

    print()
    print("Building model...")
    model = PyroSegNet(in_channels=12, out_channels=1, base_ch=32)
    n_params = sum(p.numel() for p in model.parameters())
    print("  Parameters: {:,}".format(n_params))

    ckpt_path = str(ckpt_dir / "pyrosegnet_local_best.pth")

    print()
    print("Training {} epochs...".format(args.epochs))
    t_start = time.time()
    history, best_iou, best_epoch = fit(
        model, train_loader, val_loader, config, device,
        checkpoint_path=ckpt_path, max_epochs=args.epochs,
        patience=args.epochs + 1,
    )
    total_time = time.time() - t_start
    print()
    print("Total training time: {:.1f} min".format(total_time / 60))

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(result_dir / "pyrosegnet_local_history.csv", index=False)

    print()
    print("Loading best checkpoint (epoch {}, val IoU={:.4f})...".format(
        best_epoch, best_iou))
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pos_weight = torch.tensor([config["training"]["pos_weight"]]).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5, pos_weight=pos_weight)
    test_metrics = evaluate(model, test_loader, criterion, device)

    print()
    print("=== TEST SET RESULTS (PyroSegNet, local) ===")
    print("  Precision (burned): {:.4f}".format(test_metrics["precision"]))
    print("  Recall (burned):    {:.4f}".format(test_metrics["recall"]))
    print("  F1 (burned):        {:.4f}".format(test_metrics["f1"]))
    print("  IoU (burned):       {:.4f}".format(test_metrics["iou"]))
    print("  Dice (burned):      {:.4f}".format(test_metrics["dice"]))
    print("  Accuracy:           {:.4f}".format(test_metrics["accuracy"]))
    print("  Balanced Accuracy:  {:.4f}".format(test_metrics["balanced_accuracy"]))

    print()
    print("=== COMPARISON WITH THRESHOLD BASELINE ===")
    baseline = {"precision": 0.4855, "recall": 0.6706, "f1": 0.5633,
                "iou": 0.3920, "dice": 0.5633}
    print("  {:<12} {:<12} {:<12}".format("Metric", "Threshold", "PyroSegNet"))
    print("  " + "-" * 36)
    for k in ["precision", "recall", "f1", "iou", "dice"]:
        print("  {:<12} {:<12.4f} {:<12.4f}".format(
            k, baseline[k], test_metrics[k]))

    row = {"model": "PyroSegNet_local", "img_size": args.img_size,
           "epochs": args.epochs, "best_val_iou": best_iou,
           "best_epoch": best_epoch, "train_time_min": total_time / 60}
    row.update(test_metrics)
    pd.DataFrame([row]).to_csv(
        result_dir / "pyrosegnet_local_test_metrics.csv", index=False)
    print()
    print("Checkpoint:", ckpt_path)
    print("History:   ", result_dir / "pyrosegnet_local_history.csv")
    print("Metrics:   ", result_dir / "pyrosegnet_local_test_metrics.csv")


if __name__ == "__main__":
    main()
