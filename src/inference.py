from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import CFG
from src.utils import load_json


class ButterflyClassifier:
    """A simple wrapper for the trained butterfly classifier."""

    def __init__(
        self,
        model_path: Path = CFG.models_dir / "butterfly_model.keras",
        class_map_path: Path = CFG.class_map_json,
        img_size: Tuple[int, int] = CFG.img_size,
    ) -> None:
        self.img_size = img_size
        self.class_map = load_json(class_map_path)
        self.idx_to_label = {v: k for k, v in self.class_map.items()}
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        print(f"Classes: {list(self.class_map.keys())}")

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Convert PIL image to normalized tensor ready for the model."""
        # Resize to target size
        image = image.resize(self.img_size)
        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)

    def predict(self, image: Image.Image, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Run inference on a PIL image.
        Returns a list of top_k predictions, each with 'label' and 'confidence'.
        """
        input_tensor = self.preprocess_image(image)
        predictions = self.model.predict(input_tensor, verbose=0)[0]
        # Get top-k indices
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            results.append({
                "label": self.idx_to_label[idx],
                "confidence": float(predictions[idx])
            })
        return results