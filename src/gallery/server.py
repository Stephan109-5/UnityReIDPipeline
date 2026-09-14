from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import os
import threading
import pickle
import uvicorn
import logging


class FeaturePayload(BaseModel):
    feature: List[float]
    threshold: float = 0.75


class UpdatePayload(BaseModel):
    id: int
    feature: List[float]


app = FastAPI()
# structured gallery: id -> { first_observed: np.ndarray, most_recent: np.ndarray, top_k: [ {feature: np.ndarray, confidence: float}, ... ] }
gallery: Dict[int, Dict[str, Any]] = {}
next_id = 1
lock = threading.Lock()
TOP_K = 5
MIN_CONFIDENCE_FOR_TOP_K = 0.8
save_path = os.getenv("REID_GALLERY_PATH", "reid_gallery_service.pkl")


def _normalize(vec: List[float]) -> np.ndarray:
    a = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(a)
    return a / norm if norm > 0 else a


def _insert_into_top_k(top_k: List[dict], feature: np.ndarray, confidence: float) -> None:
    """Insert (feature, confidence) into the top_k list if it qualifies.
    Maintains descending order by confidence and caps length at TOP_K.
    """
    if confidence <= 0:
        return
    entry = {"feature": feature, "confidence": float(confidence)}
    # find insert position to keep descending order
    pos = 0
    while pos < len(top_k) and top_k[pos]["confidence"] >= confidence:
        pos += 1
    top_k.insert(pos, entry)
    # trim
    if len(top_k) > TOP_K:
        top_k.pop()


def _representative_embedding(entry: Dict[str, Any]) -> np.ndarray | None:
    """Build a normalized representative embedding from the stored profile vectors."""
    vectors = []

    first_observed = entry.get("first_observed")
    if first_observed is not None and np.asarray(first_observed).size > 0:
        vectors.append(np.asarray(first_observed, dtype=float))

    most_recent = entry.get("most_recent")
    if most_recent is not None and np.asarray(most_recent).size > 0:
        vectors.append(np.asarray(most_recent, dtype=float))

    for item in entry.get("top_k", []) or []:
        feature = item.get("feature")
        if feature is None:
            continue
        feature_arr = np.asarray(feature, dtype=float)
        if feature_arr.size > 0:
            vectors.append(feature_arr)

    if not vectors:
        return None

    representative = np.mean(np.stack(vectors, axis=0), axis=0)
    norm = np.linalg.norm(representative)
    return representative / norm if norm > 0 else representative


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load gallery on startup
    _load()
    yield
    # no-op on shutdown


def _load():
    global gallery, next_id
    gallery = {}
    next_id = 1

    # Code for loading existing gallery from disk if file exists. Unused 
    # if os.path.exists(save_path):
    #     try:
    #         data = pickle.load(open(save_path, "rb"))
    #         raw_gallery = data.get("gallery", {})
    #         gallery = {}
    #         for k, v in raw_gallery.items():
    #             try:
    #                 fo = np.asarray(v.get("first_observed", []), dtype=float)
    #                 mr = np.asarray(v.get("most_recent", []), dtype=float)
    #                 top_k_raw = v.get("top_k", []) or []
    #                 top_k = []
    #                 for t in top_k_raw:
    #                     top_k.append({
    #                         "feature": np.asarray(t.get("feature", []), dtype=float),
    #                         "confidence": float(t.get("confidence", 0.0)),
    #                     })
    #                 gallery[int(k)] = {"first_observed": fo, "most_recent": mr, "top_k": top_k}
    #             except Exception:
    #                 continue
    #         next_id = int(data.get("next_id", 1))
    #     except Exception:
    #         gallery = {}
    #         next_id = 1


@app.post("/find_or_create")
def find_or_create(payload: FeaturePayload):
    global next_id
    feat = _normalize(payload.feature)
    best_id = None
    best_score = -1.0
    with lock:  # ensure thread safety for gallery access and modification
        for pid, entry in gallery.items():
            emb = _representative_embedding(entry)
            if emb is None:
                continue
            score = float(np.dot(feat, emb))
            if score > best_score:
                best_score = score
                best_id = pid

        # if we have a match above the threshold, update the existing entry; otherwise create a new one
        if best_id is not None and best_score >= float(payload.threshold):
            entry = gallery[best_id]
            entry["most_recent"] = feat

            if best_score >= MIN_CONFIDENCE_FOR_TOP_K:
                _insert_into_top_k(entry["top_k"], feat, best_score)

            return {"id": int(best_id), "score": float(round(best_score, 6)), "is_new": False}

        # create new identity
        new_id = next_id
        next_id += 1
        top_k = []
        # _insert_into_top_k(top_k, feat, 1.0) # initial feature gets a confidence of 1.0 in top_k
        gallery[new_id] = {"first_observed": feat, "most_recent": feat, "top_k": top_k}
        print(f"Created new identity {new_id} with initial confidence 1.0")
        return {"id": int(new_id), "score": 1.0, "is_new": True}


@app.post("/update")
def update(payload: UpdatePayload):
    feat = _normalize(payload.feature)
    with lock:
        current = gallery.get(int(payload.id))
        current["most_recent"] = feat
        rep = _representative_embedding(current)
        if rep is not None:
            sim = float(np.dot(feat, rep))
            if sim >= MIN_CONFIDENCE_FOR_TOP_K:
                _insert_into_top_k(current["top_k"], feat, sim)
    return {"ok": True}


@app.post("/save")
def save():
    with lock:
        serial = {}
        for k, v in gallery.items():
            serial[k] = {
                "first_observed": v["first_observed"].tolist(),
                "most_recent": v["most_recent"].tolist(),
                "top_k": [{"feature": t["feature"].tolist(), "confidence": t["confidence"]} for t in v["top_k"]],
            }
        data = {"next_id": next_id, "gallery": serial}
        tmp = save_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(data, f)
        os.replace(tmp, save_path)
    return {"saved": True, "path": save_path}


@app.post("/reset")
def reset_gallery():
    """Clear the in-memory gallery and reset ID counter. Use with caution."""
    global gallery, next_id
    with lock:
        gallery = {}
        next_id = 1
        # remove persisted file if exists
        try:
            if os.path.exists(save_path):
                os.remove(save_path)
        except Exception:
            logging.getLogger("gallery_service").exception("Failed to remove persisted gallery file during reset")
    return {"reset": True}


if __name__ == "__main__":
    uvicorn.run("src.reid.gallery_service.server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
