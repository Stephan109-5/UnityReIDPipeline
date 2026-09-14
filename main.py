import argparse
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
from typing import List
import yaml
import sys
import logging
import logging.handlers
import re
from src.unitycam.camera_worker import CameraWorker
from src.gallery.gallery_client import GalleryClient


def worker_process(camera_id, worker_cfg, result_queue, log_queue):
    """Construct and run a CameraWorker inside the child process.

    Configure a `QueueHandler` so child process log records are sent to
    the parent's `QueueListener` for serialized console output.
    """
    # Configure logging to send records to the parent's log queue
    qh = logging.handlers.QueueHandler(log_queue)
    root_logger = logging.getLogger()
    # Remove existing handlers to avoid duplicate output in child
    root_logger.handlers = []
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(qh)

    worker = CameraWorker(
        camera_id=camera_id,
        config=worker_cfg,
        result_queue=result_queue,
    )
    worker.run()

def load_config(path):
    """Safely loads the YAML file."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{path}' not found.")
        sys.exit(1)


def build_eval_run_dir(base_dir: str, run_name: str = None) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_run_name = re.sub(r'[^A-Za-z0-9._-]+', '_', str(run_name).strip()) if run_name else ''
    folder_name = f"{safe_run_name}_{timestamp}" if safe_run_name else timestamp
    return Path(base_dir) / folder_name


def main(config):
    global_cfg = config.get('global', {})
    cameras = config.get('cameras', [])
    eval_cfg = config.get('evaluation', {})

    if not cameras:
        print("Error: No cameras configured. Add entries under 'cameras' in config.yaml.")
        sys.exit(1)

    reid_enabled = any(bool(camera.get('reid', global_cfg.get('reid', False))) for camera in cameras)
    reid_service_url = global_cfg.get('reid_service_url', 'http://127.0.0.1:8000')
    gallery_client = None
    
    if reid_enabled and reid_service_url:
        try:
            gallery_client = GalleryClient(reid_service_url, timeout=10.0)
            gallery_client.reset()
            print(f"Reset ReID gallery service at {reid_service_url}")
        except Exception as exc:
            print(f"Warning: failed to reset ReID gallery service at {reid_service_url}: {exc}")

    eval_enabled = bool(eval_cfg.get('enabled'))
    eval_output_dir = Path(eval_cfg.get('output_dir'))
    eval_run_dir = build_eval_run_dir(eval_output_dir, eval_cfg.get('run_name')) if eval_enabled else eval_output_dir

    result_queue: mp.Queue = mp.Queue()
    processes: List[mp.Process] = []

    # Setup a multiprocessing-safe logging queue and listener so that
    # child process logs don't interleave on stdout.
    log_queue: mp.Queue = mp.Queue()
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    listener = logging.handlers.QueueListener(log_queue, console_handler)
    listener.start()

    for idx, camera_entry in enumerate(cameras):
        camera_id = camera_entry.get('camera_id', f'camera_{idx}')

        worker_cfg = dict(global_cfg)
        worker_cfg.update({
            'host': camera_entry.get('host', global_cfg.get('host', '0.0.0.0')),
            'port': int(camera_entry.get('port', global_cfg.get('port', 5005))),
            'tracker_type': camera_entry.get('tracker_type', global_cfg.get('tracker_type', 'bytetrack')),
            'yolo_model_path': camera_entry.get('yolo_model_path', global_cfg.get('yolo_model_path', 'yolov8n.pt')),
            'yolo_conf_threshold': float(camera_entry.get('yolo_conf_threshold', global_cfg.get('yolo_conf_threshold', 0.25))),
            'display_frames': bool(camera_entry.get('display_frames', global_cfg.get('display_frames', True))),
            'eval_tracking': eval_enabled,
            'save_eval_results': bool(eval_cfg.get('save_eval_results', False)),
            'eval_output_dir': str(eval_run_dir),
            'iou_threshold': float(eval_cfg.get('iou_threshold', global_cfg.get('iou_threshold', 0.5))),
            'match_mode': eval_cfg.get('match_mode', global_cfg.get('match_mode', 'iou')),
            'reid': bool(worker_cfg.get('reid', False)),
            'reid_model_name': worker_cfg.get('reid_model_name', global_cfg.get('reid_model_name', 'osnet_x1_0')),
            'reid_model_path': worker_cfg.get('reid_model_path', global_cfg.get('reid_model_path', None)),
            'reid_threshold': float(worker_cfg.get('reid_threshold', 0.5)),
            'eval_reid': bool(eval_cfg.get('eval_reid', global_cfg.get('eval_reid', False))),
            'eval_reid_trackeval': bool(eval_cfg.get('eval_reid_trackeval', global_cfg.get('eval_reid_trackeval', True))),
            'eval_reid_output_dir': str(eval_run_dir),
            'reid_min_area': int(eval_cfg.get('reid_min_area', global_cfg.get('reid_min_area', 1000))),
            'reid_iou_threshold': float(eval_cfg.get('reid_iou_threshold', global_cfg.get('reid_iou_threshold', 0.5))),
            'reid_eval_verbose': bool(eval_cfg.get('reid_eval_verbose', global_cfg.get('reid_eval_verbose', False))),
            'reid_service_url': global_cfg.get('reid_service_url', 'http://127.0.0.1:8000'),
        })

        process = mp.Process(
            target=worker_process,
            args=(camera_id, worker_cfg, result_queue, log_queue),
            name=f"worker-{camera_id}",
        )
        process.start()
        processes.append(process)
        print(f"Started worker for {camera_id} on {worker_cfg['host']}:{worker_cfg['port']}")

    results_received = 0
    while results_received < len(processes):
        result_item = result_queue.get()
        results_received += 1
        camera_id = result_item.get('camera_id', 'unknown')
        status = result_item.get('status', 'unknown')
        print(f"Camera {camera_id} finished with status: {status}")

    for process in processes:
        process.join()

    print('All camera workers have ended.')

    if gallery_client is not None:
        try:
            gallery_client.save()
            print(f"Saved ReID gallery service state at {reid_service_url}")
        except Exception as exc:
            print(f"Warning: failed to save ReID gallery service state at {reid_service_url}: {exc}")

    # Stop the logging listener after workers have finished
    try:
        listener.stop()
    except Exception:
        pass


if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    main(config)