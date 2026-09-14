from typing import Optional, List, Dict, Tuple
from supervision import Detections
import cv2
import numpy as np

def build_tracking_view(
    frame: np.ndarray,
    gt_objects: List[Dict],
    tracker_type: str,
    tracker,
    yolo_model,
    camera_id: str,
    frame_count: int,
    # conf: float = 0.25 # default confidence threshold for YOLO detections
):
    groundtruth_frame = frame.copy()
    tracking_frame = frame.copy()
    track_list: List[Tuple[int, int, int, int, int, float]] = []  # (x1, y1, x2, y2, track_id, conf)

    # Draw ground truth boxes
    for gt_object in gt_objects:
        x1, y1, x2, y2 = gt_object['bbox']
        cv2.rectangle(groundtruth_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            groundtruth_frame,
            gt_object['name'],
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    # Run YOLO detection
    cam_id = int(camera_id)
    # results = yolo_model(frame, classes=[0], verbose=False, conf=conf)
    # num_detections = 0
    # if results and results[0].boxes is not None:
    #     num_detections = len(results[0].boxes)
    # print(f"Frame {frame_count}: GT objects: {len(gt_objects)}, Raw detections: {num_detections}")

    if tracker_type == 'botsort' or tracker_type == 'bytetrack':
        results = yolo_model.track(frame, classes=[0], verbose=False, persist=True, tracker=f"{tracker_type}.yaml")
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            # print(f"  YOLO: Found {len(boxes)} tracked objects")
            for box in boxes:
                # Exclude non-confirmed tracks (those without an assigned ID)
                if box.id is None:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                # w_box = x2 - x1
                # h_box = y2 - y1
                track_id = int(box.id[0]) + (cam_id * 10000)
                track_list.append((x1, y1, x2, y2, track_id, conf))

    # else:
    #     # Deep SORT
    #     detections = []
    #     if results and results[0].boxes is not None:
    #         boxes = results[0].boxes
    #         for box in boxes:
    #             x1, y1, x2, y2 = box.xyxy[0].tolist()
    #             conf = float(box.conf[0])
    #             w_box = x2 - x1
    #             h_box = y2 - y1
    #             detections.append(([x1, y1, w_box, h_box], conf, 'person'))

    #     print(f"  YOLO: Found {len(detections)} filtered detections")
    #     tracks = tracker.update_tracks(detections, frame=frame)
    #     for track in tracks:
    #         if not track.is_confirmed():
    #             continue
    #         ltrb = track.to_ltrb()
    #         tx1, ty1, tx2, ty2 = map(int, ltrb)
    #         track_id = int(track.track_id) + (cam_id * 10000)
    #         track_list.append((tx1, ty1, tx2, ty2, track_id, 0.0))

    # print(f" {tracker_type} Final track_list size: {len(track_list)}\n")

    for tx1, ty1, tx2, ty2, track_id, _ in track_list:
        tx1, ty1, tx2, ty2 = int(tx1), int(ty1), int(tx2), int(ty2)
        cv2.rectangle(tracking_frame, (tx1, ty1), (tx2, ty2), (255, 0, 0), 2)
        cv2.putText(
            tracking_frame,
            f"ID {track_id}",
            (tx1, max(0, ty1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    return groundtruth_frame, tracking_frame, track_list