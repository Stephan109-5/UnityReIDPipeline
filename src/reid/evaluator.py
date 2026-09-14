from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.metrics.base_evaluator import BaseEvaluator
from src.tracking.utils import match_detections


# Potential utility function for cropping GT objects for ReID feature extraction or visualization.
def _crop_bbox(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]


class ReIDEvaluator(BaseEvaluator):
    """Evaluator for Person Re-Identification using online ID metrics."""
    def __init__(
        self,
        save_results: bool = True,
        output_dir: str = "./reid_evaluation_results",
        camera_id: str = "camera_0",
        reid_model_name: str = "reid_model",
        reid_weights_path: str = None,
        min_area: int = 0,      
        iou_threshold: float = 0.5,
        verbose: bool = False,
        match_mode: str = "iou",
    ):
        super().__init__(
            save_results=save_results,
            output_dir=output_dir,
            camera_id=camera_id,
            iou_threshold=iou_threshold,
            match_mode=match_mode,
            verbose=verbose,
        )
        self.reid_model_name = reid_model_name
        self.reid_weights_path = reid_weights_path
        self.min_area = max(0, int(min_area))
        # self.query_crops_dir = self.run_dir / "query_crops"

        # ReID-specific metrics
        self.total_gt_instances = 0
        self.total_pred_instances = 0
        self.idtp = 0
        self.idfp = 0
        self.idfn = 0
        self.id_switches = 0

        # Tracks the last assigned ReID id per GT identity for switch counting.
        self.gt_to_last_reid_id: Dict[str, str] = {}

    def _get_result_dir_name(self) -> str:
        """Return the result directory name for ReID evaluation."""
        return "reid_results"

    @property
    def seen_identities(self) -> int:
        """Get the number of unique GT identities seen."""
        return len(self.gt_to_last_reid_id)

    @property
    def total_instances(self) -> Tuple[int, int]:
        """Get total GT and prediction instances."""
        return (self.total_gt_instances, self.total_pred_instances)

    def _save_image(self, path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)

    def evaluate_frame(
        self,
        frame_id: int,
        frame: np.ndarray,
        gt_objects: List[Dict],
        reid_assigned_objects: List[Dict],
        packet_timestamp_utc_ms: int | None = None,
        source_frame_id: int | None = None,
    ) -> Dict:
        """Evaluate a single frame for ReID metrics."""
        # frame_path = self._save_frame(frame_id, frame) if self.save_results else ""
        self.total_frames += 1
        self.total_gt_instances += len(gt_objects)
        self.total_pred_instances += len(reid_assigned_objects)

        valid_gt_objects: List[Dict] = []
        for gt_object in gt_objects:
            identity = str(gt_object["name"])
            bbox = tuple(int(value) for value in gt_object["bbox"])
            
            crop_area = max(0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

            # Skip GT objects with zero-area or below the minimum area threshold
            if crop_area == 0 or crop_area < self.min_area:
                if self.verbose:
                    print(f"  ReID eval frame {frame_id}: skipping GT {identity} because crop area {crop_area} < min_area {self.min_area}")
                continue

            valid_gt_objects.append({"name": identity, "bbox": bbox})

        frame_record: Dict = {
            "camera_id": self.camera_id,
            "frame_id": frame_id,
            "source_frame_id": source_frame_id,
            "packet_timestamp_utc_ms": packet_timestamp_utc_ms,
            "num_gt": len(valid_gt_objects),
            "num_reid_predictions": len(reid_assigned_objects),
            "gt_objects": [
                {
                    "name": str(gt_object["name"]),
                    "bbox": tuple(int(value) for value in gt_object["bbox"]),
                }
                for gt_object in valid_gt_objects
            ],
            "pred_objects": [
                {
                    "reid_id": str(pred_object["reid_id"]),
                    "track_id": int(pred_object.get("track_id", -1)),
                    "bbox": tuple(int(value) for value in pred_object["bbox"]),
                }
                for pred_object in reid_assigned_objects
            ],
            "matches": [],
        }

        matches, unmatched_gt, unmatched_pred = match_detections(
            valid_gt_objects,
            reid_assigned_objects,
            iou_threshold=self.iou_threshold,
            match_mode="iou",
        )

        for gt_idx, pred_idx, iou_score in matches:
            gt_identity = str(valid_gt_objects[gt_idx]["name"])
            pred_reid_id = str(reid_assigned_objects[pred_idx]["reid_id"])

            previous_reid_id = self.gt_to_last_reid_id.get(gt_identity)
            is_id_switch = previous_reid_id is not None and previous_reid_id != pred_reid_id

            if is_id_switch:
                self.id_switches += 1
                # self.idfp += 1
                # self.idfn += 1
            # else:
            #     self.idtp += 1

            self.gt_to_last_reid_id[gt_identity] = pred_reid_id

            frame_record["matches"].append(
                {
                    "gt_identity": gt_identity,
                    "reid_id": pred_reid_id,
                    "gt_bbox": tuple(int(value) for value in valid_gt_objects[gt_idx]["bbox"]),
                    "pred_bbox": tuple(int(value) for value in reid_assigned_objects[pred_idx]["bbox"]),
                    "iou": round(float(iou_score), 6),
                    "id_switch": is_id_switch,
                }
            )

        self.idfn += len(unmatched_gt)
        self.idfp += len(unmatched_pred)

        frame_record["unmatched_gt"] = [str(valid_gt_objects[index]["name"]) for index in unmatched_gt]
        frame_record["unmatched_pred"] = [str(reid_assigned_objects[index]["reid_id"]) for index in unmatched_pred]
        frame_record["unmatched_gt_objects"] = [
            {
                "name": str(valid_gt_objects[index]["name"]),
                "bbox": tuple(int(value) for value in valid_gt_objects[index]["bbox"]),
            }
            for index in unmatched_gt
        ]
        frame_record["unmatched_pred_objects"] = [
            {
                "reid_id": str(reid_assigned_objects[index]["reid_id"]),
                "track_id": int(reid_assigned_objects[index].get("track_id", -1)),
                "bbox": tuple(int(value) for value in reid_assigned_objects[index]["bbox"]),
            }
            for index in unmatched_pred
        ]
        self.frame_records.append(frame_record)

        reid_info = {
            # "idtp": self.idtp,
            "idfp": self.idfp,
            "idfn": self.idfn,
            "id_switches": self.id_switches,
            "num_gt": len(valid_gt_objects),
            "num_reid_predictions": len(reid_assigned_objects),
            }

        return frame_record, reid_info

    def save_results_to_file(self, timing: Dict | None = None) -> Path:
        """Save ReID frame records and metadata to JSON file."""
        if not self.save_results:
            return None
        
        results = {
            "camera_id": self.camera_id,
            "reid_model": self.reid_model_name,
            "reid_weights_path": self.reid_weights_path,
            "reid_iou_threshold": self.iou_threshold,
            "reid_min_area": self.min_area,
            "saved_timestamp": self.timestamp,
        }
        if timing:
            results["processing_time_sec"] = float(timing.get("processing_time_sec", 0.0))
            results["processed_frames"] = int(timing.get("processed_frames", 0))
            results["processing_fps"] = float(timing.get("processing_fps", 0.0))
        results["frames"] = self.frame_records

        filename = f"reid_metrics_{self.camera_id}.json"
        saved_path = self._save_metrics_to_json(filename, results)
        return saved_path

        