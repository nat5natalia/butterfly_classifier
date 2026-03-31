from __future__ import annotations
import sys, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

sys.path.append(str(Path(__file__).parent.parent))
from src.config import CFG
from src.preprocessing import create_dataset
from src.utils import load_json, save_json, set_seed, sha256sum

def main() -> None:
    CFG.ensure_dirs()
    set_seed(CFG.seed, deterministic=True)

    class_map = load_json(CFG.class_map_json)
    class_names = list(class_map.keys())
    n_cls = len(class_names)

    model_path = CFG.models_dir / "best_model.keras"
    if not model_path.exists():
        model_path = CFG.models_dir / "butterfly_model.keras"
    model = tf.keras.models.load_model(model_path)

    test_df = pd.read_csv(CFG.test_csv)
    test_ds = create_dataset(CFG.test_csv, class_map, shuffle=False, augment=False)

    # Predict
    y_true, y_prob = [], []
    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist()); y_prob.extend(p.tolist())
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    y_pred = np.argmax(y_prob, axis=1)

    # Metrics
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    y_true_bin = label_binarize(y_true, classes=np.arange(n_cls))
    roc_auc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
    pr_auc = average_precision_score(y_true_bin, y_prob, average="macro")

    best_metrics = {
        "model_path": str(model_path),
        "seed": int(CFG.seed),
        "n_test": int(len(test_df)),
        "test_split_sha256": sha256sum(CFG.test_csv),
        "accuracy": float(rep["accuracy"]),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "weighted_f1": float(rep["weighted avg"]["f1-score"]),
        "roc_auc_macro_ovr": float(roc_auc),
        "pr_auc_macro": float(pr_auc),
    }

    # Save reports
    save_json(best_metrics, CFG.metrics_dir / "best_metrics.json")
    save_json(rep, CFG.metrics_dir / "classification_report.json")
    pd.DataFrame(rep).transpose().to_csv(CFG.metrics_dir / "classification_report.csv", index=True)

    # results_summary.csv (append-style)
    out_sum = CFG.metrics_dir / "results_summary.csv"
    row = pd.DataFrame([best_metrics])
    row.to_csv(out_sum, mode="a", header=not out_sum.exists(), index=False)

    # sample_predictions.csv
    idx_to_label = {int(v): str(k) for k, v in class_map.items()}
    pred_conf = y_prob[np.arange(len(y_prob)), y_pred]
    sp = pd.DataFrame({
        "filepath": test_df["filepath"].astype(str),
        "true_label": test_df["label"].astype(str),
        "pred_label": [idx_to_label[int(i)] for i in y_pred],
        "pred_confidence": pred_conf,
    })
    sp.to_csv(CFG.metrics_dir / "sample_predictions.csv", index=False)

    # confusion_matrix.png
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CFG.figures_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # misclassified_examples/
    mis_dir = CFG.reports_dir / "misclassified_examples"
    mis_dir.mkdir(parents=True, exist_ok=True)
    mis_idx = np.where(y_true != y_pred)[0][:25]
    for i in mis_idx:
        src = Path(str(test_df.loc[i, "filepath"]))
        if src.exists():
            dst = mis_dir / f"idx_{i:04d}__true_{test_df.loc[i,'label']}__pred_{idx_to_label[int(y_pred[i])]}{src.suffix}"
            shutil.copy2(src, dst)

    print("Saved: best_metrics.json, results_summary.csv, confusion_matrix.png, misclassified_examples, sample_predictions.csv")

if __name__ == "__main__":
    main()
