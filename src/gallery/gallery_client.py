import requests
import numpy as np
from typing import Tuple, Optional, Any


class GalleryClient:
    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def find_or_create(self, feature: Any, threshold: float = 0.75) -> Tuple[int, float, bool]:
        # Accept numpy arrays, lists, or torch tensors
        try:
            if hasattr(feature, "detach"):
                feat = feature.detach().cpu().numpy().ravel().tolist()
            else:
                feat = np.asarray(feature).ravel().tolist()
        except Exception:
            feat = list(feature)

        resp = self.session.post(f"{self.url}/find_or_create", json={"feature": feat, "threshold": float(threshold)}, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return int(data.get("id")), float(data.get("score", 0.0)), bool(data.get("is_new", False))

    def update(self, id: int, feature: Any) -> None:
        try:
            if hasattr(feature, "detach"):
                feat = feature.detach().cpu().numpy().ravel().tolist()
            else:
                feat = np.asarray(feature).ravel().tolist()
        except Exception:
            feat = list(feature)
        resp = self.session.post(
            f"{self.url}/update",
            json={"id": int(id), "feature": feat},
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def save(self) -> None:
        resp = requests.post(f"{self.url}/save", timeout=self.timeout)
        resp.raise_for_status()

    def reset(self) -> None:
        resp = requests.post(f"{self.url}/reset", timeout=self.timeout)
        resp.raise_for_status()

    def health(self) -> dict:
        resp = requests.get(f"{self.url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
