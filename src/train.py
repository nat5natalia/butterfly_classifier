from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

from src.config import CFG


def create_model(num_classes: int, input_shape: tuple[int, int, int] = (*CFG.img_size, 3)) -> tf.keras.Model:
    """
    Build a CNN model using transfer learning (MobileNetV2).
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # freeze base

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = CFG.learning_rate) -> None:
    """Compile the model with Adam optimizer and sparse categorical crossentropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = CFG.epochs,
    callbacks: list | None = None,
) -> tf.keras.callbacks.History:
    """Train the model and return history."""
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks or []
    )
    return history


def save_model(model: tf.keras.Model, model_path: Path = CFG.models_dir / "butterfly_model.keras") -> None:
    """Save trained model in .keras format."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")