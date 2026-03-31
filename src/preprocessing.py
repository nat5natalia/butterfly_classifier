from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import tensorflow as tf
from src.config import CFG

def _decode(path: tf.Tensor) -> tf.Tensor:
    b = tf.io.read_file(path)
    img = tf.io.decode_image(b, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, CFG.img_size)
    img = tf.cast(img, tf.float32) / 255.0  # [0,1]
    return img

def create_dataset(
    csv_path: Path,
    class_map: Dict[str, int],
    batch_size: int = CFG.batch_size,
    shuffle: bool = True,
    augment: bool = False,
    seed: int = CFG.seed,
) -> tf.data.Dataset:
    df = pd.read_csv(csv_path)
    if not {"filepath", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: filepath, label")

    labels = df["label"].astype(str).map(class_map)
    if labels.isna().any():
        unknown = sorted(df.loc[labels.isna(), "label"].astype(str).unique().tolist())
        raise ValueError(f"Unknown labels (not in class_map): {unknown}")

    paths = df["filepath"].astype(str).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels.astype(int).values))

    opt = tf.data.Options()
    opt.experimental_deterministic = True
    ds = ds.with_options(opt)

    ds = ds.map(lambda p, y: (_decode(p), tf.cast(y, tf.int32)),
                num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def _augment(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, 0.1)
    x = tf.image.random_contrast(x, 0.9, 1.1)
    return x, y
