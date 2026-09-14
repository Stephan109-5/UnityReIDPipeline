"""
Base evaluator class with shared logic for both tracking and ReID evaluation.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List
import json


class BaseEvaluator:
    """Base class for evaluation metrics collection."""

    def __init__(
        self,
        save_results: bool = True,
        output_dir: str = "./eval_results",
        camera_id: str = "camera_0",
        iou_threshold: float = 0.5,
        match_mode: str = "iou",
        verbose: bool = False,
    ):
        self.save_results = save_results
        self.output_dir = Path(output_dir)
        self.camera_id = camera_id
        self.iou_threshold = iou_threshold
        self.match_mode = (match_mode or "iou").lower()
        self.verbose = verbose

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"{self._get_result_dir_name()}"
        if self.save_results:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        self.frame_records: List[Dict] = []
        self.total_frames = 0


    def _get_result_dir_name(self) -> str:
        """Override in subclass to return appropriate directory name."""
        return "eval_results"

    def _compute_frame_metrics(
        self, tp: int, fp: int, fn: int
    ) -> Dict[str, float]:
        """Compute common frame-level metrics."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
        }

    def get_frame_records(self) -> List[Dict]:
        """Get all collected frame records for aggregation."""
        return self.frame_records

    def print_header(self, title: str) -> None:
        """Print evaluation results header."""
        separator = "=" * 60
        print(f"\n{separator}")
        print(f"{title}")
        print(separator)

    def print_footer(self) -> None:
        """Print evaluation results footer."""
        print("=" * 60 + "\n")

    def _save_metrics_to_json(
        self, filename: str, payload: Dict
    ) -> Path:
        """Save metrics to JSON file."""
        if not self.save_results:
            return None

        metrics_path = self.run_dir / filename
        with metrics_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)

        print(f"Results saved to {metrics_path}")
        return metrics_path

    def evaluate_frame(self, frame_id: int, *args, **kwargs) -> Dict:
        """Override in subclass to implement frame evaluation logic."""
        raise NotImplementedError("Subclass must implement evaluate_frame")

    def save_results_to_file(self) -> Path:
        """Override in subclass to implement result saving."""
        raise NotImplementedError("Subclass must implement save_results_to_file")
