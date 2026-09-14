from ultralytics import YOLO

def load_detect_model(model_path: str):
    if model_path.startswith('yolo'):
        return load_yolo_model(model_path)
    else:
        try:
            return load_yolo_model(model_path)
        except Exception as e:
            print(f"Failed to load model with path {model_path}: {e}")
        # finally:
        #     raise ValueError(f"Unsupported model type in path: {model_path}") 

def load_yolo_model(model_path: str = 'yolov8n.pt') -> YOLO:
    return YOLO(model_path)
