from __future__ import annotations

import tensorflow as tf

from src.config import CFG
from src.preprocessing import create_dataset
from src.train import create_model, compile_model, train_model, save_model
from src.utils import load_json


def main() -> None:
    # Load class map
    class_map = load_json(CFG.class_map_json)
    num_classes = len(class_map)

    # Create datasets
    train_ds = create_dataset(CFG.train_csv, class_map, shuffle=True, augment=True)
    val_ds = create_dataset(CFG.val_csv, class_map, shuffle=False, augment=False)

    # Build and compile model
    model = create_model(num_classes)
    compile_model(model)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            CFG.models_dir / "best_model.keras",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]

    # Train
    history = train_model(model, train_ds, val_ds, callbacks=callbacks)
    import json
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(CFG.metrics_dir / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    # Save final model
    save_model(model, CFG.models_dir / "butterfly_model.keras")

    print("Training completed.")


if __name__ == "__main__":
    main()