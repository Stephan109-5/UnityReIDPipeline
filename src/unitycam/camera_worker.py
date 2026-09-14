from src.unitycam.camera_listener import CamListener
from src.detection.utils import load_detect_model
from src.tracking.tracker_factory import init_tracker
from typing import List, Dict, Tuple
import cv2
import numpy as np
from src.unitycam.camera_utils import build_tracking_view
from src.tracking.evaluator import TrackingEvaluator
from src.reid.reid import init_reid_feature_extractor
from src.gallery.gallery_client import GalleryClient
from src.reid.evaluator import ReIDEvaluator
import logging
import time
import torch


def crop_bbox(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]


def is_bbox_too_close_to_edge(
    bbox: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    margin_px: int,
) -> bool:
    """Return True when a bbox touches or nearly touches the image boundary."""
    if margin_px <= 0:
        return False

    x1, y1, x2, y2 = bbox
    return (
        x1 <= margin_px
        or y1 <= margin_px
        or x2 >= frame_width - margin_px
        or y2 >= frame_height - margin_px
    )



class CameraWorker:
    def __init__(self, camera_id, config, result_queue):
        self.camera_id = camera_id
        self.config = config
        self.result_queue = result_queue
        self.logger = logging.getLogger(f"camera.{camera_id}")
        self.detect_model = load_detect_model(config['yolo_model_path']) if config.get('yolo_model_path') else None
        self.tracker_model = init_tracker(config['tracker_type']) if config.get('tracker_type') else None
        self.reid_feature_extractor = init_reid_feature_extractor(config['reid_model_name'], config['reid_model_path']) if config.get('reid', False) else None
        self.reid_threshold = float(config.get('reid_threshold', 0.75))
        self.reid_edge_margin_px = int(config.get('reid_edge_margin_px', 20))

        self.detect_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loaded detection model on device: {self.detect_model.device}")

    def run(self):
        frame_count = 0
        total_processing_time_sec = 0.0
        track_evaluator = None
        reid_service_url = self.config.get('reid_service_url') # Use FastAPI gallery service
        gallery_client = None
        track_to_reid: Dict[int, Dict[str, int]] = {}
        reid_track_memory_frames = int(self.config.get('reid_track_memory_frames', 45))

        if reid_service_url and self.config.get('reid', False):
            try:
                gallery_client = GalleryClient(reid_service_url)
            except Exception:
                print(f"Warning: failed to connect to ReID gallery service at {reid_service_url}.")
                gallery_client = None

        if self.config['eval_tracking']:
            track_evaluator = TrackingEvaluator(save_results=self.config['save_eval_results'], 
                                                output_dir=self.config['eval_output_dir'], 
                                                iou_threshold=self.config['iou_threshold'],
                                                match_mode=self.config['match_mode'],
                                                tracker_type=self.config['tracker_type'],
                                                camera_id=self.camera_id
                                                )
            
        if self.config['eval_reid']:
            reid_evaluator = ReIDEvaluator(save_results=self.config['save_eval_results'], 
                                            output_dir=self.config['eval_reid_output_dir'],
                                            camera_id=self.camera_id, 
                                            reid_model_name=self.config['reid_model_name'],
                                            reid_weights_path=self.config['reid_model_path'],
                                            min_area=self.config['reid_min_area'],
                                            iou_threshold=self.config['iou_threshold'],
                                            verbose=self.config['reid_eval_verbose']
                                            )

        with CamListener(host=self.config['host'], port=self.config['port']) as listener:
            self.logger.info(f"listening on {self.config['host']}:{self.config['port']}")
            while True:
                self.logger.info("Waiting for Unity camera connection...")
                addr = listener.accept()
                self.logger.info(f"Accepted connection from {addr}")
                # prepare display window variables per connection
                window_name = f"Camera {self.camera_id}"
                window_created = False
                # use a single scale factor (default 0.5) to shrink the combined view while keeping aspect ratio
                display_scale = float(self.config.get('display_scale', 0.5))

                try:
                    while True:
                        packet = listener.recv_packet()
                        self.logger.debug(f"received packet metadata={packet.metadata} image_bytes={packet.image_length}")
                        frame_received_time = time.perf_counter()
                        object_ids = packet.object_ids
                        source_frame_id = packet.frame_id
                        timestamp_utc_ms = packet.timestamp_utc_ms

                        img_bytes = packet.image_bytes
                        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                        if frame is None:
                            self.logger.warning("Failed to decode image")
                            continue

                        h, w, _ = frame.shape
                        gt_objects: List[Dict] = []
                        reid_gt_objects: List[Dict] = []
                        for entry in object_ids:
                            if not entry:
                                continue

                            self.logger.debug(entry)
                            parts = entry.split('|')
                            if len(parts) != 5:
                                continue

                            name, x, y, bw, bh = parts
                            ix = int(float(x) * w)
                            iy = int((1.0 - float(y) - float(bh)) * h)
                            iw = int(float(bw) * w)
                            ih = int(float(bh) * h)
                            gt_bbox = (ix, iy, ix + iw, iy + ih)

                            # GT object for tracking evaluation can be any size and even close to edge, but will be included in tracking GT
                            gt_objects.append({'name': name, 'bbox': gt_bbox})

                            # GT object for ReID evaluation must meet minimum area and not be too close to edge to be included in ReID gallery
                            if iw * ih < self.config['reid_min_area'] or is_bbox_too_close_to_edge(gt_bbox, w, h, self.reid_edge_margin_px):
                                continue
                            else:
                                reid_gt_objects.append({'name': name, 'bbox': gt_bbox})

                        groundtruth_frame, tracking_frame, track_list = build_tracking_view(
                            frame=frame,
                            gt_objects=gt_objects,
                            tracker_type=self.config['tracker_type'],
                            tracker=self.tracker_model,
                            yolo_model=self.detect_model,
                            frame_count=frame_count,
                            camera_id=self.camera_id
                        )

                        if self.config['eval_tracking']:
                            pred_objects = [
                                {
                                    'track_id': int(track_data[4]),
                                    'bbox': (int(track_data[0]), int(track_data[1]), int(track_data[2]), int(track_data[3])), # x1, y1, x2, y2
                                }
                                for track_data in track_list
                            ]
                            frame_result = track_evaluator.evaluate_frame(frame_count, gt_objects, pred_objects, timestamp_utc_ms, source_frame_id)
                            metrics_text = f"Track: TP:{frame_result['tp']} FP:{frame_result['fp']} FN:{frame_result['fn']}"
                            cv2.putText(
                                tracking_frame,
                                metrics_text,
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2,
                            )

                        reid_assigned_objects: List[Dict] = []
                        if self.config['reid']:
                            active_track_ids = {int(track_data[4]) for track_data in track_list}

                            # Remove cached ReID assignments for tracks that have disappeared for more than reid_track_memory_frames
                            for cached_track_id, cached_data in list(track_to_reid.items()):
                                if cached_track_id in active_track_ids:
                                    continue
                                last_seen = int(cached_data.get('last_seen_frame', -1))
                                if frame_count - last_seen > reid_track_memory_frames:
                                    del track_to_reid[cached_track_id]

                            for track_data in track_list:
                                x1, y1, x2, y2, track_id = map(int, track_data[:5])
                                x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
                                if is_bbox_too_close_to_edge((x1_i, y1_i, x2_i, y2_i), w, h, self.reid_edge_margin_px):
                                    continue
                                track_crop = crop_bbox(frame, (x1_i, y1_i, x2_i, y2_i))

                                # skip small crops 
                                if (track_crop.shape[0] * track_crop.shape[1]) < (self.config['reid_min_area'] * 0.55):
                                    continue
                        
                                # Convert BGR (OpenCV) -> RGB for torchreid FeatureExtractor which expects RGB images
                                rgb_crop = cv2.cvtColor(track_crop, cv2.COLOR_BGR2RGB)
                                features = self.reid_feature_extractor(rgb_crop)

                                if gallery_client is not None:
                                    # Check if this track_id already has a cached ReID assignment in the gallery, and if so, use it instead of querying the gallery again. 
                                    # This helps maintain ReID ID consistency for tracks across frames, even if the gallery matching is not 100% stable, while also reducing the number of gallery queries which can be a bottleneck.
                                    cached_assignment = track_to_reid.get(int(track_id))
                                    if cached_assignment is None:
                                        reid_id, similarity, is_new = gallery_client.find_or_create(
                                            features,
                                            threshold=self.reid_threshold,
                                        )
                                        if similarity >= 0.85: # only cache if similarity is reasonably above threshold to avoid caching noisy assignments
                                            track_to_reid[int(track_id)] = {
                                                'reid_id': int(reid_id),
                                                'last_seen_frame': int(frame_count),
                                            }
                                    else:
                                        reid_id = int(cached_assignment['reid_id'])
                                        similarity = 1.0
                                        is_new = False
                                        cached_assignment['last_seen_frame'] = int(frame_count)
                                        try:
                                            gallery_client.update(reid_id, features)
                                        except Exception:
                                            self.logger.warning(f"Failed to update ReID gallery for ID {reid_id}")
                                            pass

                                    label = f"ReID {reid_id}"
                                    if is_new:
                                        label += " new"
                                    else:
                                        label += f" {similarity:.2f}"

                                    reid_assigned_objects.append(
                                        {
                                            'track_id': int(track_id),
                                            'reid_id': int(reid_id),
                                            'bbox': (x1_i, y1_i, x2_i, y2_i),
                                            'similarity': float(similarity),
                                            'is_new': bool(is_new),
                                        }
                                    )

                                    cv2.putText(
                                        tracking_frame,
                                        label,
                                        (x1_i, max(0, y1_i - 28)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55,
                                        (0, 255, 255),
                                        2,
                                    )
                        
                        if self.config['eval_reid']:
                            _, info = reid_evaluator.evaluate_frame(frame_count, frame, reid_gt_objects, reid_assigned_objects, timestamp_utc_ms, source_frame_id)

                            reid_text = f"ReID: IDFP:{info['idfp']} IDFN:{info['idfn']} IDSW:{info['id_switches']}"
                            cv2.putText(
                                tracking_frame,
                                reid_text,
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 0),
                                2,
                            )

                        if self.config['display_frames']:
                            combined_view = np.vstack((groundtruth_frame, tracking_frame))

                            # create a resizable window once per connection
                            if not window_created:
                                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                                # set an initial window size based on the scale factor
                                try:
                                    h_c_init, w_c_init = combined_view.shape[:2]
                                    init_w = max(1, int(w_c_init * display_scale))
                                    init_h = max(1, int(h_c_init * display_scale))
                                    cv2.resizeWindow(window_name, init_w, init_h)
                                except Exception:
                                    pass
                                window_created = True

                            # scale combined view using the single scale factor while preserving aspect ratio
                            h_c, w_c = combined_view.shape[:2]
                            if display_scale > 0 and display_scale < 1.0:
                                new_w = max(1, int(w_c * display_scale))
                                new_h = max(1, int(h_c * display_scale))
                                display_img = cv2.resize(combined_view, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            else:
                                display_img = combined_view

                            cv2.imshow(window_name, display_img)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                            # do not record display timing here; keep rendering only

                        total_processing_time_sec += max(0.0, float(time.perf_counter() - frame_received_time))
                        frame_count += 1

                except Exception as e:
                    self.logger.exception(f"Error in CameraWorker: {e}")
                    # on exception make sure to close the window for this connection
                    if window_created:
                        try:
                            cv2.destroyWindow(window_name)
                        except Exception:
                            pass
                    break
                finally:
                    # cleanup per-connection window if it was created
                    if self.config['display_frames'] and window_created:
                        try:
                            cv2.destroyWindow(window_name)
                        except Exception:
                            pass

                cv2.destroyAllWindows()
                break

        result_item_queue = {
            'camera_id': self.camera_id,
            'status': 'ok'
        }

        processing_fps = (float(frame_count) / float(total_processing_time_sec)) if total_processing_time_sec > 0 else 0.0

        result_item_queue['timing'] = {
            'processed_frames': int(frame_count),
            'processing_time_sec': round(total_processing_time_sec, 6),
            'processing_fps': round(processing_fps, 4),
        }

        timing_snapshot = result_item_queue['timing']

        if self.config['eval_tracking'] and self.config['save_eval_results']:
            track_evaluator.save_results_to_file(timing=timing_snapshot)

        if self.config['eval_reid']:
            if self.config['save_eval_results']:
                reid_results_path = reid_evaluator.save_results_to_file(timing=timing_snapshot)
                if reid_results_path is not None:
                    result_item_queue['reid_results_path'] = str(reid_results_path)
        
        self.result_queue.put(result_item_queue)

        
