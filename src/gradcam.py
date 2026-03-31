from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from typing import Any, Dict, List, Tuple, Optional
from PIL import Image

from src.config import CFG


class GradCAM:
    """Grad-CAM implementation for a Keras model."""

    def __init__(
            self,
            model: tf.keras.Model,
            layer_name: Optional[str] = None,
    ) -> None:
        """
        Initialize Grad-CAM for a given model.

        Args:
            model: Trained Keras model.
            layer_name: Name of the last convolutional layer. If None, will try to
                        find the last Conv2D layer automatically.
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        if not self.layer_name:
            raise ValueError("Could not find a convolutional layer in the model.")

        # Build a sub-model that outputs both the target layer activations and the model predictions
        self.grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(self.layer_name).output, model.output]
        )

    def _find_last_conv_layer(self) -> Optional[str]:
        """Find the name of the last convolutional layer in the model."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                return layer.name
        return None

    def compute_heatmap(
            self,
            image: np.ndarray,
            class_idx: int,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a given image and class.

        Args:
            image: Preprocessed image (shape: 1, H, W, 3) normalized to [0,1].
            class_idx: Index of the class to visualize.

        Returns:
            Heatmap as a 2D numpy array (H, W) in range [0, 1].
        """
        with tf.GradientTape() as tape:
            conv_output, predictions = self.grad_model(image)
            loss = predictions[:, class_idx]

        # Gradients of the loss w.r.t. the conv output
        grads = tape.gradient(loss, conv_output)
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weighted activations
        conv_output = conv_output[0]  # remove batch dimension
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = tf.maximum(heatmap, 0)  # ReLU
        heatmap /= tf.reduce_max(heatmap)  # Normalize to [0, 1]

        return heatmap.numpy()

    def overlay_heatmap(
            self,
            image: np.ndarray,
            heatmap: np.ndarray,
            alpha: float = 0.4,
            colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.

        Args:
            image: Original image (H, W, 3) with values in [0, 255] (uint8) or [0, 1] (float).
            heatmap: 2D heatmap array (H, W) in range [0, 1].
            alpha: Transparency factor for the overlay.
            colormap: OpenCV colormap constant.

        Returns:
            Overlaid image as uint8 (H, W, 3).
        """
        # Convert image to uint8 if it's in [0, 1]
        if image.max() <= 1.0:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.astype(np.uint8)

        # Resize heatmap to image size if needed
        if heatmap.shape != img.shape[:2]:
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Apply colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)

        # Overlay
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        return overlay

    def generate_and_save(
            self,
            image: np.ndarray,
            class_idx: int,
            original_img: np.ndarray,
            save_path: str,
            alpha: float = 0.4,
    ) -> None:
        """
        Convenience method: compute heatmap, overlay, and save.

        Args:
            image: Preprocessed input (1, H, W, 3).
            class_idx: Target class.
            original_img: Original image (H, W, 3) in [0, 255] or [0, 1].
            save_path: Path to save the output.
            alpha: Overlay transparency.
        """
        heatmap = self.compute_heatmap(image, class_idx)
        overlay = self.overlay_heatmap(original_img, heatmap, alpha=alpha)
        plt.imsave(save_path, overlay)


def preprocess_for_gradcam(img: np.ndarray, target_size: Tuple[int, int] = CFG.img_size) -> np.ndarray:
    """
    Preprocess a PIL image or numpy array for Grad-CAM input.

    Args:
        img: Input image (H, W, 3) as uint8 or float [0, 1].
        target_size: Size to resize to (width, height).

    Returns:
        Batch of size 1 with normalized values in [0, 1].
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    # Resize
    img = cv2.resize(img, target_size)
    # Normalize to [0, 1]
    if img.max() > 1:
        img = img / 255.0
    # Add batch dimension
    return np.expand_dims(img, axis=0).astype(np.float32)