from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

# Add project root to path if necessary
sys.path.append(str(Path(__file__).parent.parent))

from src.config import CFG
from src.preprocessing import create_dataset
from src.evaluate import load_trained_model, evaluate_model, plot_confusion_matrix, save_evaluation_results
from src.utils import load_json


def main() -> None:
    # 1. Load class map
    class_map = load_json(CFG.class_map_json)
    num_classes = len(class_map)
    class_names = list(class_map.keys())
    print("Class names:", class_names)

    # 2. Create test dataset
    test_ds = create_dataset(
        CFG.test_csv,
        class_map,
        shuffle=False,
        augment=False,
        batch_size=CFG.batch_size,
    )

    # 3. Load trained model
    model = load_trained_model(CFG.models_dir / "butterfly_model.keras")

    # 4. Evaluate
    test_loss, test_acc, report, cm = evaluate_model(model, test_ds)

    # 5. Save results
    save_evaluation_results(test_loss, test_acc, report, cm, CFG.metrics_dir)

    # 6. Plot confusion matrix
    fig = plot_confusion_matrix(
        cm,
        class_names,
        save_path=CFG.figures_dir / "confusion_matrix.png"
    )
    plt.show()


if __name__ == "__main__":
    main()