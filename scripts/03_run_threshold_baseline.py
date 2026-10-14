"""Run the dNBR threshold baseline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from pyrosegnet.config import load_config
from pyrosegnet.data.splits import get_split_samples
from pyrosegnet.traditional.threshold import sweep_thresholds, load_dnbr_and_mask
from pyrosegnet.evaluation.metrics import metrics_from_confusion
from pyrosegnet.visualization.plots import plot_prediction

if __name__ == "__main__":
    config = load_config()
    hdf5_path = config["data"]["hdf5_file"]
    result_dir = Path(config["outputs"]["result_dir"])
    fig_dir = Path(config["outputs"]["figure_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load splits from official folds ----
    print("Loading official folds...")
    splits = get_split_samples(hdf5_path)
    print(f"  train: {len(splits['train'])} samples")
    print(f"  val:   {len(splits['val'])} samples")
    print(f"  test:  {len(splits['test'])} samples")

    # ---- 2. Sweep thresholds on validation ----
    thresholds = np.round(np.arange(
        config["threshold"]["dnbr_range"][0],
        config["threshold"]["dnbr_range"][1] + 1e-9,
        config["threshold"]["dnbr_step"],
    ), 3)
    print(f"\nSweeping {len(thresholds)} thresholds on validation set...")
    val_results = sweep_thresholds(hdf5_path, splits["val"], thresholds, verbose=True)

    val_rows = []
    for t in thresholds:
        r = val_results[float(t)]
        m = metrics_from_confusion(r["TP"], r["FP"], r["TN"], r["FN"])
        val_rows.append({"threshold": float(t),
                         "TP": r["TP"], "FP": r["FP"],
                         "TN": r["TN"], "FN": r["FN"], **m})
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(result_dir / "threshold_selection.csv", index=False)

    # ---- 3. Pick best threshold by F1 ----
    best_row = val_df.loc[val_df["f1"].idxmax()]
    best_t = float(best_row["threshold"])
    print(f"\nBest validation threshold: dNBR >= {best_t:.3f}")
    print(f"  Val precision: {best_row['precision']:.4f}")
    print(f"  Val recall:    {best_row['recall']:.4f}")
    print(f"  Val F1:        {best_row['f1']:.4f}")
    print(f"  Val IoU:       {best_row['iou']:.4f}")

    # ---- 4. Evaluate on test with fixed threshold ----
    print(f"\nEvaluating on test set with threshold={best_t:.3f}...")
    test_results = sweep_thresholds(hdf5_path, splits["test"], [best_t], verbose=True)
    r = test_results[float(best_t)]
    test_metrics = metrics_from_confusion(r["TP"], r["FP"], r["TN"], r["FN"])

    print("\n=== TEST SET RESULTS ===")
    print(f"  Threshold: {best_t:.3f}")
    print(f"  TP={r['TP']:,}  FP={r['FP']:,}  TN={r['TN']:,}  FN={r['FN']:,}")
    print(f"  Precision (burned): {test_metrics['precision']:.4f}")
    print(f"  Recall (burned):    {test_metrics['recall']:.4f}")
    print(f"  F1 (burned):        {test_metrics['f1']:.4f}")
    print(f"  IoU (burned):       {test_metrics['iou']:.4f}")
    print(f"  Dice (burned):      {test_metrics['dice']:.4f}")
    print(f"  Accuracy:           {test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Specificity:        {test_metrics['specificity']:.4f}")

    # Save test metrics
    test_row = {"model": "dNBR_threshold", "threshold": best_t,
                "TP": r["TP"], "FP": r["FP"], "TN": r["TN"], "FN": r["FN"],
                **test_metrics}
    pd.DataFrame([test_row]).to_csv(result_dir / "threshold_test_metrics.csv", index=False)

    # ---- 5. Visualization on 3 test samples ----
    print("\nGenerating prediction visualizations for 3 test samples...")
    # Sort by burned area to get diversity
    sample_burn = []
    for key in splits["test"]:
        _, mask = load_dnbr_and_mask(hdf5_path, key)
        sample_burn.append((float(mask.mean()), key))
    sample_burn.sort()

    picks = [sample_burn[0][1],
             sample_burn[len(sample_burn) // 2][1],
             sample_burn[-1][1]]

    for i, key in enumerate(picks):
        dnbr, mask = load_dnbr_and_mask(hdf5_path, key)
        pred = (dnbr >= best_t).astype(np.int8)
        plot_prediction(
            hdf5_path, key, pred,
            save_path=str(fig_dir / f"threshold_pred_{i}.png"),
            title_extra=f"| t={best_t:.2f}",
        )
        print(f"  saved threshold_pred_{i}.png (burn ratio={mask.mean():.3f})")

    print(f"\nResults saved to {result_dir}")
    print(f"Figures saved to {fig_dir}")
