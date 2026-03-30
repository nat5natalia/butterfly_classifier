from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CFG
from src.preprocessing import create_dataset
from src.utils import load_json, save_json


def load_trained_model(model_path: Path = CFG.models_dir / "butterfly_model.keras") -> tf.keras.Model:
    """Load a saved Keras model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
) -> tuple[float, float, dict, np.ndarray]:
    """
    Evaluate model on test dataset.
    Returns: test_loss, test_accuracy, classification_report_dict, confusion_matrix_array
    """
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Get predictions for all test samples
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    return test_loss, test_acc, report, cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot confusion matrix and optionally save to file."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")

    return fig


def save_evaluation_results(
    test_loss: float,
    test_acc: float,
    report: dict,
    cm: np.ndarray,
    output_dir: Path = CFG.metrics_dir,
) -> None:
    """Save evaluation metrics to JSON and confusion matrix to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "classification_report": report
    }
    save_json(metrics, output_dir / "test_metrics.json")

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(output_dir / "confusion_matrix.csv", index=False)
    print(f"Evaluation results saved to {output_dir}")