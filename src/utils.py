from __future__ import annotations
import hashlib, json, os, random
from pathlib import Path
from typing import Any, Dict
import numpy as np

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Best-effort reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        if deterministic:
            try: tf.config.experimental.enable_op_determinism()
            except Exception: pass
    except Exception:
        pass

def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()
