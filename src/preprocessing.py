from __future__ import annotations

import tensorflow as tf
import pandas as pd
from pathlib import Path

from src.config import CFG


def decode_image(filepath: str, label: tf.int32, img_size: tuple[int, int]) -> tuple[tf.Tensor, tf.Tensor]:
    """Load and preprocess an image."""
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0  # normalize to [0,1]
    return img, label


def create_dataset(
    csv_path: Path,
    class_map: dict,
    batch_size: int = CFG.batch_size,
    img_size: tuple[int, int] = CFG.img_size,
    shuffle: bool = True,
    augment: bool = False,
) -> tf.data.Dataset:
    """
    Create tf.data.Dataset from CSV file.
    CSV must have columns 'filepath' and 'label'.
    """
    df = pd.read_csv(csv_path)
    filepaths = df["filepath"].values
    labels = df["label"].map(class_map).values

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(lambda x, y: decode_image(x, y, img_size),
                          num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))

    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def augment_image(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply random augmentations to an image."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label