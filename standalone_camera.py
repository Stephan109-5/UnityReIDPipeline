import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

from src.detection.utils import load_detect_model
from src.tracking.utils import match_detections
from src.unitycam.camera_listener import CamListener


def is_bbox_too_close_to_edge(
    bbox: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    margin_px: int,
) -> bool:
    if margin_px <= 0:
        return False

    x1, y1, x2, y2 = bbox
    return (
        x1 <= margin_px
        or y1 <= margin_px
        or x2 >= frame_width - margin_px
        or y2 >= frame_height - margin_px
    )


def draw_gt_boxes(frame: np.ndarray, gt_objects: List[Dict]) -> np.ndarray:
    output = frame.copy()
    for gt_object in gt_objects:
        x1, y1, x2, y2 = gt_object['bbox']
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            str(gt_object.get('name', 'gt')),
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return output


def draw_detection_boxes(
    frame: np.ndarray,
    gt_objects: List[Dict],
    pred_objects: List[Dict],
    matches: List[Tuple[int, int, float]],
) -> np.ndarray:
    output = draw_gt_boxes(frame, gt_objects)
    matched_pred_to_iou = {pred_idx: iou for _, pred_idx, iou in matches}

    for pred_object in pred_objects:
        x1, y1, x2, y2 = pred_object['bbox']
        pred_idx = int(pred_object['track_id'])
        iou = matched_pred_to_iou.get(pred_idx)
        color = (0, 0, 255) if iou is not None else (255, 0, 0)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"det {pred_idx}"
        if iou is not None:
            label = f"{label} IoU {iou:.2f}"
        else:
            label = f"{label} unmatched"

        cv2.putText(
            output,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    return output


def bbox_to_xywh(bbox: Tuple[int, int, int, int]) -> Dict[str, int]:
    x1, y1, x2, y2 = map(int, bbox)
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return {
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'width': width,
        'height': height,
        'area': width * height,
    }


def build_capture_metadata(
    *,
    camera_id: str,
    frame_count: int,
    packet,
    gt_objects: List[Dict],
    pred_objects: List[Dict],
    matches: List[Tuple[int, int, float]],
    summary_text: str,
) -> Dict:
    gt_match_map: Dict[int, Tuple[int, float]] = {}
    pred_match_map: Dict[int, Tuple[int, float]] = {}
    for gt_idx, pred_idx, iou in matches:
        gt_match_map[int(gt_idx)] = (int(pred_idx), float(iou))
        pred_match_map[int(pred_idx)] = (int(gt_idx), float(iou))

    gt_bbox_cache = [bbox_to_xywh(tuple(gt_object['bbox'])) for gt_object in gt_objects]
    pred_bbox_cache = [bbox_to_xywh(tuple(pred_object['bbox'])) for pred_object in pred_objects]

    gt_records: List[Dict] = []
    for gt_idx, gt_object in enumerate(gt_objects):
        bbox_info = gt_bbox_cache[gt_idx]
        match = gt_match_map.get(gt_idx)
        gt_records.append(
            {
                'index': gt_idx,
                'name': gt_object.get('name'),
                'bbox': bbox_info,
                'matched_pred_index': match[0] if match is not None else None,
                'iou': round(match[1], 6) if match is not None else None,
            }
        )

    pred_records: List[Dict] = []
    for pred_idx, pred_object in enumerate(pred_objects):
        bbox_info = pred_bbox_cache[pred_idx]
        match = pred_match_map.get(pred_idx)
        pred_records.append(
            {
                'index': pred_idx,
                'track_id': pred_object.get('track_id'),
                'confidence': pred_object.get('confidence'),
                'bbox': bbox_info,
                'matched_gt_index': match[0] if match is not None else None,
                'iou': round(match[1], 6) if match is not None else None,
            }
        )

    return {
        'camera_id': camera_id,
        'frame_count': frame_count,
        'frame_id': getattr(packet, 'frame_id', None),
        'timestamp_utc_ms': getattr(packet, 'timestamp_utc_ms', None),
        'summary': summary_text,
        'gt_objects': gt_records,
        'pred_objects': pred_records,
        'match_count': len(matches),
        'matches': [
            {
                'gt_index': int(gt_idx),
                'pred_index': int(pred_idx),
                'gt_name': gt_objects[int(gt_idx)].get('name'),
                'pred_track_id': pred_objects[int(pred_idx)].get('track_id'),
                'iou': round(float(iou), 6),
                'gt_bbox_area': int(gt_bbox_cache[int(gt_idx)]['area']),
                'pred_bbox_area': int(pred_bbox_cache[int(pred_idx)]['area']),
                'pred_to_gt_area_ratio': (
                    round(
                        float(pred_bbox_cache[int(pred_idx)]['area']) / float(gt_bbox_cache[int(gt_idx)]['area']),
                        6,
                    )
                    if gt_bbox_cache[int(gt_idx)]['area'] > 0
                    else None
                ),
            }
            for gt_idx, pred_idx, iou in matches
        ],
    }


def save_capture(
    output_dir: Path,
    camera_id: str,
    frame_count: int,
    combined_view: np.ndarray,
    metadata: Dict,
) -> Path:
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    base_name = f"{timestamp}"
    image_path = output_dir / f"{base_name}.png"
    json_path = output_dir / f"{base_name}.json"

    if not cv2.imwrite(str(image_path), combined_view):
        raise RuntimeError(f"Failed to save capture image to {image_path}")

    with json_path.open('w', encoding='utf-8') as file_handle:
        json.dump(metadata, file_handle, indent=2)

    return image_path


def load_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as file_handle:
        return yaml.safe_load(file_handle) or {}


def resolve_camera_entry(config: Dict, camera_id: Optional[str], camera_index: int) -> Dict:
    global_cfg = dict(config.get('global', {}))
    cameras = config.get('cameras', []) or []

    camera_entry: Dict = {}
    if cameras:
        if camera_id is not None:
            for entry in cameras:
                if str(entry.get('camera_id')) == str(camera_id):
                    camera_entry = dict(entry)
                    break
            else:
                raise ValueError(f"Camera id '{camera_id}' was not found in config.")
        else:
            if camera_index < 0 or camera_index >= len(cameras):
                raise IndexError(f"camera_index {camera_index} is out of range for {len(cameras)} configured cameras.")
            camera_entry = dict(cameras[camera_index])

    worker_cfg = dict(global_cfg)
    worker_cfg.update(camera_entry)

    worker_cfg.setdefault('camera_id', camera_entry.get('camera_id', camera_id if camera_id is not None else camera_index))
    worker_cfg.setdefault('host', global_cfg.get('host', '0.0.0.0'))
    worker_cfg.setdefault('port', int(global_cfg.get('port', 5005)))
    worker_cfg.setdefault('yolo_model_path', global_cfg.get('yolo_model_path', 'yolov8n.pt'))
    worker_cfg.setdefault('yolo_conf_threshold', float(global_cfg.get('yolo_conf_threshold', 0.25)))
    worker_cfg.setdefault('display_frames', bool(global_cfg.get('display_frames', True)))
    worker_cfg.setdefault('display_scale', float(global_cfg.get('display_scale', 0.5)))
    worker_cfg.setdefault('save_frames_dir', global_cfg.get('save_frames_dir', 'saved_frames'))
    worker_cfg.setdefault('iou_threshold', float(config.get('evaluation', {}).get('iou_threshold', global_cfg.get('iou_threshold', 0.5))))
    worker_cfg.setdefault('edge_margin_px', int(global_cfg.get('edge_margin_px', 20)))

    return worker_cfg


class StandaloneYoloCameraWorker:
    def __init__(self, camera_id: str, config: Dict):
        self.camera_id = camera_id
        self.config = config
        self.logger = logging.getLogger(f"camera_yolo.{camera_id}")
        self.conf_threshold = float(config.get('yolo_conf_threshold', 0.25))
        self.iou_threshold = float(config.get('iou_threshold', 0.5))
        self.edge_margin_px = int(config.get('reid_edge_margin_px', config.get('edge_margin_px', 20)))
        self.display_frames = bool(config.get('display_frames', True))
        self.display_scale = float(config.get('display_scale', 0.5))
        self.save_frames_dir = Path(config.get('save_frames_dir', 'saved_frames'))
        self.save_frames_dir.mkdir(parents=True, exist_ok=True)

        model_path = str(config.get('yolo_model_path', 'yolov8n.pt'))
        self.detect_model = load_detect_model(model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detect_model.to(device)
        self.logger.info("Loaded detection model on device %s from %s", device, model_path)

    def _parse_gt_objects(self, object_ids: List[str], frame_width: int, frame_height: int) -> List[Dict]:
        gt_objects: List[Dict] = []

        for entry in object_ids:
            if not entry:
                continue

            parts = entry.split('|')
            if len(parts) != 5:
                continue

            name, x, y, bw, bh = parts
            ix = int(float(x) * frame_width)
            iy = int((1.0 - float(y) - float(bh)) * frame_height)
            iw = int(float(bw) * frame_width)
            ih = int(float(bh) * frame_height)
            gt_bbox = (ix, iy, ix + iw, iy + ih)

            gt_objects.append({'name': name, 'bbox': gt_bbox})

        return gt_objects

    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        results = self.detect_model(frame, classes=[0], verbose=False, conf=self.conf_threshold)
        detections: List[Dict] = []

        if not results or results[0].boxes is None:
            return detections

        for det_index, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    'track_id': det_index,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(box.conf[0]) if box.conf is not None else None,
                }
            )

        return detections

    def run(self):
        frame_count = 0
        total_processing_time_sec = 0.0
        window_name = f"YOLO Camera {self.camera_id}"
        window_created = False

        with CamListener(host=self.config['host'], port=self.config['port']) as listener:
            self.logger.info("Listening on %s:%s", self.config['host'], self.config['port'])

            while True:
                self.logger.info("Waiting for Unity camera connection...")
                addr = listener.accept()
                self.logger.info("Accepted connection from %s", addr)

                try:
                    while True:
                        packet = listener.recv_packet()
                        frame_received_time = time.perf_counter()

                        img_array = np.frombuffer(packet.image_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if frame is None:
                            self.logger.warning("Failed to decode image")
                            continue

                        frame_height, frame_width, _ = frame.shape
                        gt_objects = self._parse_gt_objects(packet.object_ids, frame_width, frame_height)

                        detections = self._detect_objects(frame)
                        matches, unmatched_gt, unmatched_pred = match_detections(
                            gt_objects,
                            detections,
                            iou_threshold=self.iou_threshold,
                            match_mode='iou',
                        )

                        detection_frame = draw_detection_boxes(frame, gt_objects, detections, matches)
                        groundtruth_frame = draw_gt_boxes(frame, gt_objects)

                        matched_ious = [match[2] for match in matches]
                        mean_iou = (sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0
                        summary_text = (
                            f"GT:{len(gt_objects)} DET:{len(detections)} MATCH:{len(matches)} "
                            f"UM_GT:{len(unmatched_gt)} UM_DET:{len(unmatched_pred)} IoU:{mean_iou:.2f}"
                        )
                        # cv2.putText(
                        #     detection_frame,
                        #     summary_text,
                        #     (10, 30),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.7,
                        #     (0, 255, 255),
                        #     2,
                        # )

                        if matches:
                            match_details = ", ".join(
                                f"gt{gt_idx}-det{pred_idx}:{iou:.2f}"
                                for gt_idx, pred_idx, iou in matches
                            )
                        else:
                            match_details = "no IoU matches"

                        self.logger.info("Frame %s -> %s | %s", frame_count, summary_text, match_details)

                        if self.display_frames:
                            combined_view = np.vstack((groundtruth_frame, detection_frame))

                            if not window_created:
                                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                                try:
                                    init_height, init_width = combined_view.shape[:2]
                                    init_w = max(1, int(init_width * self.display_scale))
                                    init_h = max(1, int(init_height * self.display_scale))
                                    cv2.resizeWindow(window_name, init_w, init_h)
                                except Exception:
                                    pass
                                window_created = True

                            if 0.0 < self.display_scale < 1.0:
                                display_width = max(1, int(combined_view.shape[1] * self.display_scale))
                                display_height = max(1, int(combined_view.shape[0] * self.display_scale))
                                display_img = cv2.resize(combined_view, (display_width, display_height), interpolation=cv2.INTER_AREA)
                            else:
                                display_img = combined_view

                            cv2.imshow(window_name, display_img)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('s'):
                                metadata = build_capture_metadata(
                                    camera_id=self.camera_id,
                                    frame_count=frame_count,
                                    packet=packet,
                                    gt_objects=gt_objects,
                                    pred_objects=detections,
                                    matches=matches,
                                    summary_text=summary_text,
                                )
                                try:
                                    image_path = save_capture(
                                        self.save_frames_dir,
                                        self.camera_id,
                                        frame_count,
                                        detection_frame,
                                        metadata,
                                    )
                                    self.logger.info("Saved capture image and metadata at %s", image_path)
                                except Exception as exc:
                                    self.logger.warning("Failed to save capture: %s", exc)
                            if key == ord('q'):
                                break

                        total_processing_time_sec += max(0.0, float(time.perf_counter() - frame_received_time))
                        frame_count += 1

                except Exception as exc:
                    self.logger.exception("Error in StandaloneYoloCameraWorker: %s", exc)
                    break
                finally:
                    if window_created:
                        try:
                            cv2.destroyWindow(window_name)
                        except Exception:
                            pass

                cv2.destroyAllWindows()
                break

        processing_fps = (float(frame_count) / float(total_processing_time_sec)) if total_processing_time_sec > 0 else 0.0
        self.logger.info(
            "Finished. frames=%s processing_time_sec=%.6f processing_fps=%.4f",
            frame_count,
            total_processing_time_sec,
            processing_fps,
        )


def main():
    parser = argparse.ArgumentParser(description='Run a single Unity camera worker with YOLO detection only.')
    parser.add_argument('--config', default='standalone.yaml', help='Path to config file')
    parser.add_argument('--camera-id', default=None, help='Camera id to launch from the config file')
    parser.add_argument('--camera-index', type=int, default=0, help='Fallback camera index when camera-id is not provided')
    args = parser.parse_args()

    config = load_config(args.config)
    worker_cfg = resolve_camera_entry(config, args.camera_id, args.camera_index)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    worker = StandaloneYoloCameraWorker(camera_id=str(worker_cfg.get('camera_id')), config=worker_cfg)
    worker.run()


if __name__ == '__main__':
    main()