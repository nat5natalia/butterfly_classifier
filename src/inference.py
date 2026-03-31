from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import tensorflow as tf
from PIL import Image
from src.config import CFG
from src.utils import load_json

class ButterflyClassifier:
    def __init__(self, model_path: Path, class_map_path: Path, img_size: Tuple[int,int]=CFG.img_size) -> None:
        self.img_size = img_size
        self.class_map = load_json(class_map_path)  # label -> idx
        self.idx_to_label = {int(v): str(k) for k, v in self.class_map.items()}
        self.model = tf.keras.models.load_model(model_path)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        arr = np.asarray(image)
        x = tf.image.resize(arr, self.img_size)
        x = tf.cast(x, tf.float32) / 255.0
        return tf.expand_dims(x, 0).numpy()

    def predict(self, image: Image.Image, top_k: int = 3) -> List[Dict[str, Any]]:
        probs = self.model.predict(self.preprocess(image), verbose=0)[0]
        idx = np.argsort(probs)[-top_k:][::-1]
        return [{"label": self.idx_to_label[int(i)], "confidence": float(probs[int(i)])} for i in idx]
