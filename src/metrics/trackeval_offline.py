from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from trackeval.metrics import clear as trackeval_clear
from trackeval.metrics import hota as trackeval_hota
from trackeval.metrics import identity as trackeval_identity


@dataclass(frozen=True)
class LoadedResultFile:
    path: Path
    camera_id: str
    frames: List[Dict[str, Any]]
    metadata: Dict[str, Any]


def _load_json_file(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as file_handle:
        return json.load(file_handle)


def _to_tlwh(bbox: Sequence[Any]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.asarray([float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1)], dtype=float)


def  _iou_matrix(gt_boxes: Sequence[np.ndarray], tracker_boxes: Sequence[np.ndarray]) -> np.ndarray:
    if not gt_boxes or not tracker_boxes:
        return np.empty((len(gt_boxes), len(tracker_boxes)), dtype=float)

    gt = np.asarray(gt_boxes, dtype=float)
    tracker = np.asarray(tracker_boxes, dtype=float)

    gt_x1 = gt[:, 0][:, np.newaxis]
    gt_y1 = gt[:, 1][:, np.newaxis]
    gt_x2 = (gt[:, 0] + gt[:, 2])[:, np.newaxis]
    gt_y2 = (gt[:, 1] + gt[:, 3])[:, np.newaxis]

    tr_x1 = tracker[:, 0][np.newaxis, :]
    tr_y1 = tracker[:, 1][np.newaxis, :]
    tr_x2 = (tracker[:, 0] + tracker[:, 2])[np.newaxis, :]
    tr_y2 = (tracker[:, 1] + tracker[:, 3])[np.newaxis, :]

    inter_x1 = np.maximum(gt_x1, tr_x1)
    inter_y1 = np.maximum(gt_y1, tr_y1)
    inter_x2 = np.minimum(gt_x2, tr_x2)
    inter_y2 = np.minimum(gt_y2, tr_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    gt_area = np.maximum(0.0, gt[:, 2]) * np.maximum(0.0, gt[:, 3])
    tr_area = np.maximum(0.0, tracker[:, 2]) * np.maximum(0.0, tracker[:, 3])
    union = gt_area[:, np.newaxis] + tr_area[np.newaxis, :] - intersection

    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(union > 0.0, intersection / union, 0.0)
    return iou.astype(float)


def _extract_result_files(results_dir: Path, pattern: str) -> List[Path]:
    return sorted(results_dir.glob(pattern))


def _load_tracking_files(tracking_results_dir: Path) -> List[LoadedResultFile]:
    loaded: List[LoadedResultFile] = []
    for path in _extract_result_files(tracking_results_dir, 'tracking_results_*.json'):
        try:
            payload = _load_json_file(path)
            frames = payload.get('frames', [])
            if not isinstance(frames, list):
                frames = []
            loaded.append(
                LoadedResultFile(
                    path=path,
                    camera_id=str(payload.get('camera_id', path.stem.replace('tracking_results_', ''))),
                    frames=frames,
                    metadata=payload,
                )
            )
        except Exception as exc:
            loaded.append(
                LoadedResultFile(
                    path=path,
                    camera_id=str(path.stem.replace('tracking_results_', '')),
                    frames=[],
                    metadata={'error': str(exc)},
                )
            )
    return loaded


def _load_reid_files(reid_results_dir: Path) -> List[LoadedResultFile]:
    loaded: List[LoadedResultFile] = []
    for path in _extract_result_files(reid_results_dir, 'reid_metrics_*.json'):
        try:
            payload = _load_json_file(path)
            frames = payload.get('frames', [])
            if not isinstance(frames, list):
                frames = []
            loaded.append(
                LoadedResultFile(
                    path=path,
                    camera_id=str(payload.get('camera_id', path.stem.replace('reid_metrics_', ''))),
                    frames=frames,
                    metadata=payload,
                )
            )
        except Exception as exc:
            loaded.append(
                LoadedResultFile(
                    path=path,
                    camera_id=str(path.stem.replace('reid_metrics_', '')),
                    frames=[],
                    metadata={'error': str(exc)},
                )
            )
    return loaded


def _build_sequence_data(frames: Sequence[Dict[str, Any]], pred_id_field: str) -> Dict[str, Any]:
    gt_id_map: Dict[str, int] = {}
    tracker_id_map: Dict[str, int] = {}
    gt_ids: List[np.ndarray] = []
    tracker_ids: List[np.ndarray] = []
    gt_dets: List[np.ndarray] = []
    tracker_dets: List[np.ndarray] = []
    similarity_scores: List[np.ndarray] = []
    total_gt_dets = 0
    total_tracker_dets = 0

    for frame in frames:
        gt_objects = frame.get('gt_objects', []) or []
        tracker_objects = frame.get('pred_objects', []) or []

        frame_gt_ids: List[int] = []
        frame_tracker_ids: List[int] = []
        frame_gt_boxes: List[np.ndarray] = []
        frame_tracker_boxes: List[np.ndarray] = []

        for gt_obj in gt_objects:
            gt_name = gt_obj.get('name')
            gt_bbox = gt_obj.get('bbox')
            if gt_name is None or gt_bbox is None:
                continue
            gt_key = str(gt_name)
            if gt_key not in gt_id_map:
                gt_id_map[gt_key] = len(gt_id_map)
            frame_gt_ids.append(gt_id_map[gt_key])
            frame_gt_boxes.append(_to_tlwh(gt_bbox))

        for tracker_obj in tracker_objects:
            tracker_id = tracker_obj.get(pred_id_field)
            tracker_bbox = tracker_obj.get('bbox')
            if tracker_id is None or tracker_bbox is None:
                continue
            tracker_key = str(tracker_id)
            if tracker_key not in tracker_id_map:
                tracker_id_map[tracker_key] = len(tracker_id_map)
            frame_tracker_ids.append(tracker_id_map[tracker_key])
            frame_tracker_boxes.append(_to_tlwh(tracker_bbox))

        gt_ids.append(np.asarray(frame_gt_ids, dtype=int))
        tracker_ids.append(np.asarray(frame_tracker_ids, dtype=int))
        gt_dets.append(np.asarray(frame_gt_boxes, dtype=float).reshape(-1, 4) if frame_gt_boxes else np.empty((0, 4), dtype=float))
        tracker_dets.append(np.asarray(frame_tracker_boxes, dtype=float).reshape(-1, 4) if frame_tracker_boxes else np.empty((0, 4), dtype=float))
        similarity_scores.append(_iou_matrix(frame_gt_boxes, frame_tracker_boxes))

        total_gt_dets += len(frame_gt_ids)
        total_tracker_dets += len(frame_tracker_ids)

    return {
        'num_timesteps': len(frames),
        'num_gt_ids': len(gt_id_map),
        'num_tracker_ids': len(tracker_id_map),
        'num_gt_dets': total_gt_dets,
        'num_tracker_dets': total_tracker_dets,
        'gt_ids': gt_ids,
        'tracker_ids': tracker_ids,
        'gt_dets': gt_dets,
        'tracker_dets': tracker_dets,
        'similarity_scores': similarity_scores,
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (int, np.integer)):
        return float(value)
    try:
        value = float(value)
    except Exception:
        return None
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def _extract_processing_metrics(metadata: Dict[str, Any]) -> Dict[str, float | None]:
    timing_source = metadata.get('timing') if isinstance(metadata.get('timing'), dict) else metadata

    return {
        'processing_time_sec': _safe_float(timing_source.get('processing_time_sec')),
        'processed_frames': _safe_float(timing_source.get('processed_frames')),
        'processing_fps': _safe_float(timing_source.get('processing_fps')),
    }


def _average_processing_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, float | None]:
    averages: Dict[str, float | None] = {}
    for field in ('processing_time_sec', 'processed_frames', 'processing_fps'):
        values = [result[field] for result in results if result.get(field) is not None]
        averages[f'avg_{field}'] = round(float(np.mean(values)), 6) if values else None
    return averages


def _serialize_metric_result(metric: Any, raw_result: Dict[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}

    for field in getattr(metric, 'integer_array_fields', []):
        if field not in raw_result:
            continue
        values = np.asarray(raw_result[field], dtype=float)
        serialized[field] = int(np.nansum(values))
        serialized[f'{field}_curve'] = [int(round(float(item))) for item in values.tolist()]

    for field in getattr(metric, 'float_array_fields', []):
        if field not in raw_result:
            continue
        values = np.asarray(raw_result[field], dtype=float)
        serialized[field] = round(float(np.nanmean(values)), 6) if values.size else None
        serialized[f'{field}_curve'] = [round(float(item), 6) for item in values.tolist()]

    for field in getattr(metric, 'integer_fields', []):
        if field in raw_result:
            serialized[field] = int(raw_result[field])

    for field in getattr(metric, 'float_fields', []):
        if field in raw_result:
            value = _safe_float(raw_result[field])
            serialized[field] = round(value, 6) if value is not None else None

    return serialized


def _aggregate_metric_results(metric: Any, raw_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not raw_results:
        return {}

    aggregated: Dict[str, Any] = {}

    for field in getattr(metric, 'integer_array_fields', []):
        arrays = [np.asarray(result[field], dtype=float) for result in raw_results if field in result]
        if arrays:
            aggregated[field] = np.nansum(np.stack(arrays, axis=0), axis=0)

    for field in getattr(metric, 'float_array_fields', []):
        arrays = [np.asarray(result[field], dtype=float) for result in raw_results if field in result]
        if arrays:
            aggregated[field] = np.nanmean(np.stack(arrays, axis=0), axis=0)

    for field in getattr(metric, 'integer_fields', []):
        values = [result[field] for result in raw_results if field in result]
        if values:
            aggregated[field] = int(np.nansum(np.asarray(values, dtype=float)))

    for field in getattr(metric, 'float_fields', []):
        values = [result[field] for result in raw_results if field in result]
        if values:
            aggregated[field] = float(np.nanmean(np.asarray(values, dtype=float)))

    return aggregated


def _evaluate_tracking_sequence(loaded_file: LoadedResultFile) -> Dict[str, Any]:
    tracking_threshold = float(loaded_file.metadata.get('tracking_threshold', 0.5))
    sequence_data = _build_sequence_data(loaded_file.frames, pred_id_field='track_id')
    clear_metric = trackeval_clear.CLEAR({'THRESHOLD': tracking_threshold, 'PRINT_CONFIG': False})
    identity_metric = trackeval_identity.Identity({'THRESHOLD': float(tracking_threshold), 'PRINT_CONFIG': False})
    hota_metric = trackeval_hota.HOTA({'PRINT_CONFIG': False})
    processing_metrics = _extract_processing_metrics(loaded_file.metadata)

    clear_result = clear_metric.eval_sequence(sequence_data)
    identity_result = identity_metric.eval_sequence(sequence_data)
    hota_result = hota_metric.eval_sequence(sequence_data)

    return {
        'camera_id': loaded_file.camera_id,
        'source_file': str(loaded_file.path),
        'tracking_threshold': tracking_threshold,
        'track_mode': loaded_file.metadata.get('track_mode'),
        'frame_count': sequence_data['num_timesteps'],
        'num_gt_ids': sequence_data['num_gt_ids'],
        'num_tracker_ids': sequence_data['num_tracker_ids'],
        'num_gt_dets': sequence_data['num_gt_dets'],
        'num_tracker_dets': sequence_data['num_tracker_dets'],
        **processing_metrics,
        'raw_clear': clear_result,
        'raw_identity': identity_result,
        'raw_hota': hota_result,
        'clear': _serialize_metric_result(clear_metric, clear_result),
        'identity': _serialize_metric_result(identity_metric, identity_result),
        'hota': _serialize_metric_result(hota_metric, hota_result),
    }


def _evaluate_reid_global_sequence(loaded_files: Sequence[LoadedResultFile]) -> Dict[str, Any]:
    all_frames: List[Dict[str, Any]] = []
    source_files: List[str] = []
    global_gt_history: Dict[str, List[tuple[int, str, int, str]]] = {}
    global_reid_history: Dict[str, List[tuple[int, str, int, str]]] = {}
    reid_iou_threshold = None
    reid_min_area = None
    reid_model = None
    reid_weights_path = None

    for loaded_file in loaded_files:
        source_files.append(str(loaded_file.path))
        # print(loaded_file.metadata)
        if reid_iou_threshold is None:
            reid_iou_threshold = loaded_file.metadata.get('reid_iou_threshold', 0.35)
        if reid_min_area is None:
            reid_min_area = loaded_file.metadata.get('reid_min_area')
        if reid_model is None:
            reid_model = loaded_file.metadata.get('reid_model')
        if reid_weights_path is None:
            reid_weights_path = loaded_file.metadata.get('reid_weights_path')

        for frame in loaded_file.frames:
            frame_copy = dict(frame)
            frame_copy['_camera_id'] = str(frame_copy.get('camera_id', loaded_file.camera_id))
            all_frames.append(frame_copy)

            timestamp = int(frame_copy.get('packet_timestamp_utc_ms', 0) or 0)
            camera_id = str(frame_copy.get('_camera_id', loaded_file.camera_id))
            frame_id = int(frame_copy.get('frame_id', frame_copy.get('source_frame_id', 0)) or 0)
            for match in frame_copy.get('matches', []) or []:
                gt_identity = match.get('gt_identity')
                reid_id = match.get('reid_id')
                if gt_identity is None or reid_id is None:
                    continue
                gt_key = str(gt_identity)
                global_gt_history.setdefault(gt_key, []).append((timestamp, camera_id, frame_id, str(reid_id)))
                global_reid_history.setdefault(str(reid_id), []).append((timestamp, camera_id, frame_id, gt_key))

    all_frames.sort(
        key=lambda record: (
            int(record.get('packet_timestamp_utc_ms', 0) or 0),
            str(record.get('_camera_id', '')),
            int(record.get('frame_id', record.get('source_frame_id', 0)) or 0),
        )
    )

    sequence_data = _build_sequence_data(all_frames, pred_id_field='reid_id')
    if reid_iou_threshold is None:
        reid_iou_threshold = 0.35
    
    clear_metric = trackeval_clear.CLEAR({'THRESHOLD': float(reid_iou_threshold), 'PRINT_CONFIG': False})
    identity_metric = trackeval_identity.Identity({'THRESHOLD': float(reid_iou_threshold), 'PRINT_CONFIG': False})
    hota_metric = trackeval_hota.HOTA({'PRINT_CONFIG': False})

    clear_result = clear_metric.eval_sequence(sequence_data)
    identity_result = identity_metric.eval_sequence(sequence_data)
    hota_result = hota_metric.eval_sequence(sequence_data)

    id_switches = int(clear_result.get('IDSW', 0))
    gt_breakdown: List[Dict[str, Any]] = []
    reid_breakdown: List[Dict[str, Any]] = []
    for gt_identity, events in sorted(global_gt_history.items(), key=lambda item: item[0]):
        ordered_events = sorted(events, key=lambda item: (item[0], item[1], item[2], item[3]))
        observed_reid_ids: List[str] = []
        timeline: List[Dict[str, Any]] = []
        id_switch_count = 0
        previous_reid_id = None

        for timestamp, camera_id, frame_id, reid_id in ordered_events:
            if previous_reid_id is not None and reid_id != previous_reid_id:
                id_switch_count += 1
            previous_reid_id = reid_id

            if reid_id not in observed_reid_ids:
                observed_reid_ids.append(reid_id)
            timeline.append(
                {
                    'timestamp_utc_ms': timestamp,
                    'camera_id': camera_id,
                    'frame_id': frame_id,
                    'reid_id': reid_id,
                }
            )

        gt_breakdown.append(
            {
                'gt_identity': gt_identity,
                'observations': len(ordered_events),
                'id_switch_count': id_switch_count,
                'unique_reid_id_count': len(observed_reid_ids),
                'observed_reid_ids': observed_reid_ids,
                'timeline': timeline,
            }
        )

    # ReID-centric breakdown (each ReID ID's timeline of matched GT identities, with ID switch count and unique GT count)
    for reid_id, events in sorted(global_reid_history.items(), key=lambda item: item[0]):
        ordered_events = sorted(events, key=lambda item: (item[0], item[1], item[2], item[3]))
        observed_gt_ids: List[str] = []
        timeline: List[Dict[str, Any]] = []
        id_switch_count = 0
        previous_gt_id = None

        for timestamp, camera_id, frame_id, gt_id in ordered_events:
            if previous_gt_id is not None and gt_id != previous_gt_id:
                id_switch_count += 1
            previous_gt_id = gt_id

            if gt_id not in observed_gt_ids:
                observed_gt_ids.append(gt_id)
            timeline.append(
                {
                    'timestamp_utc_ms': timestamp,
                    'camera_id': camera_id,
                    'frame_id': frame_id,
                    'gt_identity': gt_id,
                }
            )

        reid_breakdown.append(
            {
                'reid_id': reid_id,
                'observations': len(ordered_events),
                'id_switch_count': id_switch_count,
                'unique_gt_id_count': len(observed_gt_ids),
                'observed_gt_ids': observed_gt_ids,
                'timeline': timeline,
            }
        )

    reid_id_switches = sum(record['id_switch_count'] for record in gt_breakdown) # Total ID switches from the GT perspective
    gt_id_switches = sum(record['id_switch_count'] for record in reid_breakdown) # Total ID switches from the ReID perspective

    return {
        'source_files': source_files,
        'frame_count': sequence_data['num_timesteps'],
        'num_gt_ids': sequence_data['num_gt_ids'],
        'num_tracker_ids': sequence_data['num_tracker_ids'],
        'num_gt_dets': sequence_data['num_gt_dets'],
        'num_tracker_dets': sequence_data['num_tracker_dets'],
        'reid_iou_threshold': float(reid_iou_threshold),
        'reid_min_area': int(reid_min_area) if reid_min_area is not None else None,
        'reid_model': reid_model,
        'reid_weights_path': reid_weights_path,
        'id_switches': id_switches,
        'gt_id_switches': gt_id_switches,
        'reid_id_switches': reid_id_switches,  # Should be same as IDSW
        'global_gt_breakdown': gt_breakdown,
        'global_reid_breakdown': reid_breakdown,
        'clear': _serialize_metric_result(clear_metric, clear_result),
        'raw_identity': identity_result,
        'raw_hota': hota_result,
        'raw_clear': clear_result,
        'identity': _serialize_metric_result(identity_metric, identity_result),
        'hota': _serialize_metric_result(hota_metric, hota_result),
    }


def evaluate_run_folder(input_folder: Path | str) -> Dict[str, Any]:
    input_folder = Path(input_folder)
    tracking_results_dir = input_folder / 'tracking_results'
    reid_results_dir = input_folder / 'reid_results'

    summary: Dict[str, Any] = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'input_folder': str(input_folder),
        'tracking': {
            'camera_results': [],
            'aggregate': {},
        },
        'reid': {
            'global_result': {},
        },
    }

    tracking_files = _load_tracking_files(tracking_results_dir) if tracking_results_dir.exists() else []
    tracking_files = [loaded for loaded in tracking_files if loaded.frames and not loaded.metadata.get('error')]

    if tracking_files:
        per_camera_results = [_evaluate_tracking_sequence(loaded_file) for loaded_file in tracking_files]
        summary['tracking']['camera_results'] = [
            {key: value for key, value in result.items() if not key.startswith('raw_')}
            for result in per_camera_results
        ]
        iou_threshold = per_camera_results[0].get('tracking_threshold')

        clear_metric_template = trackeval_clear.CLEAR({'THRESHOLD': iou_threshold, 'PRINT_CONFIG': False})
        identity_metric = trackeval_identity.Identity({'THRESHOLD': iou_threshold, 'PRINT_CONFIG': False})
        hota_metric_template = trackeval_hota.HOTA({'PRINT_CONFIG': False})
        
        clear_raw_results = [result['raw_clear'] for result in per_camera_results]
        identity_raw_results = [result['raw_identity'] for result in per_camera_results]
        hota_raw_results = [result['raw_hota'] for result in per_camera_results]
        processing_averages = _average_processing_metrics(per_camera_results)
        summary['tracking']['aggregate'] = {
            'camera_count': len(per_camera_results),
            **processing_averages,
            'clear': _serialize_metric_result(
                clear_metric_template,
                _aggregate_metric_results(clear_metric_template, clear_raw_results),
            ),
            'identity': _serialize_metric_result(
                identity_metric,
                _aggregate_metric_results(identity_metric, identity_raw_results),
            ),
            'hota': _serialize_metric_result(
                hota_metric_template,
                _aggregate_metric_results(hota_metric_template, hota_raw_results),
            ),
        }
    else:
        summary['tracking']['aggregate'] = {'camera_count': 0, 'error': 'No tracking result files were found.'}

    reid_files = _load_reid_files(reid_results_dir) if reid_results_dir.exists() else []
    reid_files = [loaded for loaded in reid_files if loaded.frames and not loaded.metadata.get('error')]

    if reid_files:
        global_result = _evaluate_reid_global_sequence(reid_files)
        summary['reid']['global_result'] = {key: value for key, value in global_result.items() if not key.startswith('raw_')}
    else:
        summary['reid']['global_result'] = {'error': 'No ReID result files were found.'}

    summary_path = input_folder / 'trackeval_summary.json'
    with summary_path.open('w', encoding='utf-8') as file_handle:
        json.dump(summary, file_handle, indent=2)

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Evaluate saved tracking and ReID results with TrackEval.')
    parser.add_argument('input_folder', type=Path, help='Folder containing tracking_results and reid_results subfolders.')
    args = parser.parse_args(argv)

    summary = evaluate_run_folder(args.input_folder)
    print(json.dumps({
        'input_folder': summary['input_folder'],
        'tracking_cameras': summary['tracking'].get('aggregate', {}).get('camera_count', 0),
        'has_reid_result': 'error' not in summary['reid'].get('global_result', {}),
        'summary_path': str(Path(summary['input_folder']) / 'trackeval_summary.json'),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())