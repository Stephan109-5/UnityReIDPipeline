from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import motmetrics as mm
import numpy as np

from src.metrics.base_evaluator import BaseEvaluator
from src.tracking.utils import match_detections, calculate_iou, calculate_diou


class TrackingEvaluator(BaseEvaluator):
    """Evaluator for multiple object tracking evaluation metrics."""

    def __init__(
        self,
        save_results: bool = False,
        output_dir: str = "./tracking_results",
        iou_threshold: float = 0.5,
        match_mode: str = "iou",
        verbose: bool = False,
        tracker_type: str = None,
        camera_id: str = None,
        compute_metrics_online: bool = True,
    ):
        super().__init__(
            save_results=save_results,
            output_dir=output_dir,
            camera_id=camera_id or "camera_0",
            iou_threshold=iou_threshold,
            match_mode=match_mode,
            verbose=verbose,
        )
        self.tracker_type = tracker_type
        # If False, evaluator will only collect per-frame records and NOT
        # update the motmetrics accumulator during the run. Metrics can be
        # computed later by calling `build_accumulator_from_records`.
        self.compute_metrics_online = compute_metrics_online

        # Tracking-specific metrics
        self.total_gt = 0
        self.total_pred = 0
        self.total_matches = 0
        self.id_switches = defaultdict(int)
        self.false_positives = 0
        self.false_negatives = 0
        self.gt_id_to_pred_id = {}

        # Motmetrics accumulator for overall MOTChallenge-style metrics.
        self.accumulator = mm.MOTAccumulator(auto_id=True)

        # For MOTP and IOU calculations
        self.sum_match_iou = 0.0

        # Tracking frame data structure
        self.per_frame_gt = []  # list of dicts: {gt_id: bbox}
        self.per_frame_pred = []  # list of dicts: {pred_id: bbox}

    def _get_result_dir_name(self) -> str:
        """Return the result directory name for tracking evaluation."""
        return "tracking_results"

    def _build_motmetrics_distance_matrix(
        self, gt_objects: List[Dict], pred_objects: List[Dict]
    ) -> np.ndarray:
        """Build a distance matrix compatible with motmetrics.

        Distances are 1 - similarity and invalid pairs are marked as NaN.
        """
        if not gt_objects or not pred_objects:
            return np.empty((len(gt_objects), len(pred_objects)))

        dist = np.full((len(gt_objects), len(pred_objects)), np.nan, dtype=float)
        for i, gt_obj in enumerate(gt_objects):
            for j, pred_obj in enumerate(pred_objects):
                if self.match_mode == "diou":
                    score = calculate_diou(gt_obj["bbox"], pred_obj["bbox"])
                else:
                    score = calculate_iou(gt_obj["bbox"], pred_obj["bbox"])
                if score >= self.iou_threshold:
                    dist[i, j] = 1.0 - float(score)
        return dist

    def _update_motmetrics_accumulator(
        self, gt_objects: List[Dict], pred_objects: List[Dict]
    ) -> None:
        """Push a frame into the motmetrics accumulator."""
        gt_ids = [str(obj["name"]) for obj in gt_objects]
        pred_ids = [str(obj["track_id"]) for obj in pred_objects]
        dist = self._build_motmetrics_distance_matrix(gt_objects, pred_objects)
        self.accumulator.update(gt_ids, pred_ids, dist)

    # NOT USED anymore
    # def _compute_frame_metrics_tracking(
    #     self,
    #     frame_id: int,
    #     tp: int,
    #     fp: int,
    #     fn: int,
    #     gt_count: int,
    #     pred_count: int,
    #     id_switches: int,
    #     iou_sum: float,
    #     matches: List[Tuple],
    #     gt_objects: List[Dict],
    #     pred_objects: List[Dict],
    # ) -> Dict:
    #     """Compute frame-level tracking metrics."""
    #     # Basic metrics
    #     base_metrics = self._compute_frame_metrics(tp, fp, fn)

    #     # Tracking-specific metrics
    #     mota = (
    #         1.0 - (fn + fp + id_switches) / gt_count if gt_count > 0 else None
    #     )
    #     motp = (iou_sum / tp) if tp > 0 else None
    #     det_a = (
    #         tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    #     )
    #     ass_a = (
    #         max(0.0, 1.0 - (id_switches / tp)) if tp > 0 else 0.0
    #     )

    #     return {
    #         "frame_id": frame_id,
    #         "gt_count": gt_count,
    #         "pred_count": pred_count,
    #         "tp": tp,
    #         "fp": fp,
    #         "fn": fn,
    #         "id_switches": id_switches,
    #         "precision": base_metrics["precision"],
    #         "recall": base_metrics["recall"],
    #         "f1_score": base_metrics["f1_score"],
    #         "mota": round(mota, 4) if mota is not None else None,
    #         "motp": round(motp, 4) if motp is not None else None,
    #         "det_a": round(det_a, 4),
    #         "ass_a": round(ass_a, 4),
    #         "avg_iou": round(motp, 4) if motp is not None else None,
    #         "matched_pairs": [
    #             {
    #                 "gt_id": gt_objects[i]["name"],
    #                 "pred_id": pred_objects[j]["track_id"],
    #                 "iou": round(float(iou), 4),
    #             }
    #             for i, j, iou in matches
    #         ],
    #     }

    def _print_frame_debug_info(
        self,
        frame_id: int,
        gt_objects: List[Dict],
        pred_objects: List[Dict],
        matches: List[Tuple],
        unmatched_gt: List[int],
        unmatched_pred: List[int],
    ) -> None:
        """Print verbose debug information for a frame."""
        print(f"\n--- Frame {frame_id} Evaluation ---")
        print(f"Ground Truth Objects ({len(gt_objects)}):")
        for gt_obj in gt_objects:
            name = gt_obj["name"]
            x1, y1, x2, y2 = gt_obj["bbox"]
            w, h = x2 - x1, y2 - y1
            print(f"  ID={name:12s} | x={x1:6.1f} y={y1:6.1f} w={w:6.1f} h={h:6.1f}")

        print(f"Predicted Objects ({len(pred_objects)}):")
        for pred_obj in pred_objects:
            track_id = pred_obj["track_id"]
            x1, y1, x2, y2 = pred_obj["bbox"]
            w, h = x2 - x1, y2 - y1
            print(f"  ID={track_id:12d} | x={x1:6.1f} y={y1:6.1f} w={w:6.1f} h={h:6.1f}")

        print(f"Matches ({len(matches)}):")
        for gt_idx, pred_idx, iou in matches:
            gt_name = gt_objects[gt_idx]["name"]
            pred_id = pred_objects[pred_idx]["track_id"]
            print(f"  GT {gt_name:12s} <-> Pred {pred_id:12d} | IoU={iou:.4f}")
        if unmatched_gt:
            print(f"Unmatched GT: {[gt_objects[i]['name'] for i in unmatched_gt]}")
        if unmatched_pred:
            print(
                f"Unmatched Pred: {[pred_objects[i]['track_id'] for i in unmatched_pred]}"
            )

    def evaluate_frame(
        self, frame_id: int, gt_objects: List[Dict], pred_objects: List[Dict], packet_timestamp_utc_ms: int = None, source_frame_id: int = None
    ) -> Dict:
        """Evaluate a single frame for tracking metrics."""
        self.total_frames += 1
        self.total_gt += len(gt_objects)
        self.total_pred += len(pred_objects)

        matches, unmatched_gt, unmatched_pred = match_detections(
            gt_objects, pred_objects, self.iou_threshold, self.match_mode
        )

        # build per-frame maps for later MOT accumulator and metrics calculations
        gt_map = {str(obj["name"]): tuple(obj["bbox"]) for obj in gt_objects}
        pred_map = {str(obj["track_id"]): tuple(obj["bbox"]) for obj in pred_objects}
        self.per_frame_gt.append(gt_map)
        self.per_frame_pred.append(pred_map)

        frame_tp = len(matches)
        frame_fp = len(unmatched_pred)
        frame_fn = len(unmatched_gt)
        frame_gt = len(gt_objects)
        frame_pred = len(pred_objects)
        frame_id_switches = 0
        # frame_iou_sum = 0.0

        # Verbose logging
        if self.verbose:
            self._print_frame_debug_info(
                frame_id, gt_objects, pred_objects, matches, unmatched_gt, unmatched_pred
            )

        self.total_matches += len(matches)
        self.false_negatives += len(unmatched_gt)
        self.false_positives += len(unmatched_pred)

        # Track ID switches within local frame
        for gt_idx, pred_idx, _ in matches:
            gt_name = gt_objects[gt_idx]["name"]
            pred_id = pred_objects[pred_idx]["track_id"]

            if gt_name in self.gt_id_to_pred_id:
                if self.gt_id_to_pred_id[gt_name] != pred_id:
                    self.id_switches[gt_name] += 1
                    frame_id_switches += 1
            self.gt_id_to_pred_id[gt_name] = pred_id

        # Build a single flattened frame record
        frame_record = {
            "tp": frame_tp,
            "fp": frame_fp,
            "fn": frame_fn,
            "gt": frame_gt,
            "pred": frame_pred,
            "id_switches": frame_id_switches,
            "source_frame_id": source_frame_id,
            "packet_timestamp_utc_ms": packet_timestamp_utc_ms,
            "gt_objects": gt_objects,
            "pred_objects": pred_objects,
            "matched_pairs": [
                {
                    "gt_id": gt_objects[i]["name"],
                    "pred_id": pred_objects[j]["track_id"],
                    "iou": round(float(iou), 4),
                }
                for i, j, iou in matches
            ],
        }

        self.frame_records.append(frame_record)
        return frame_record

    def save_results_to_file(self, timing: Dict | None = None) -> Path:
        """Save tracking frame data and metadata to JSON file."""
        if not self.save_results:
            return None

        results = {
            "camera_id": self.camera_id,
            "tracker_type": self.tracker_type,
            "tracking_threshold": self.iou_threshold,
            "track_mode": self.match_mode,
            "saved_timestamp": self.timestamp
        }
        
        if timing:
            results["processing_time_sec"] = float(timing.get("processing_time_sec", 0.0))
            results["processed_frames"] = int(timing.get("processed_frames", 0))
            results["processing_fps"] = float(timing.get("processing_fps", 0.0))

        results["frames"] = self.frame_records

        filename = f"tracking_results_{self.camera_id}.json"
        save_path = self._save_metrics_to_json(filename, results)
        return save_path