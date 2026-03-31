from __future__ import annotations
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import CFG

def create_model(num_classes: int, input_shape: tuple[int,int,int]=(*CFG.img_size,3)) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False
    inp = tf.keras.Input(shape=input_shape, name="image")
    x = layers.Rescaling(2.0, offset=-1.0, name="mobilenet_preprocess")(inp)  # [0,1] -> [-1,1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return models.Model(inp, out, name=CFG.model_name)

def compile_model(model: tf.keras.Model, lr: float = CFG.learning_rate) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy")],
    )

def save_model(model: tf.keras.Model, path: Path = CFG.models_dir / "butterfly_model.keras") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"Saved model to {path}")
