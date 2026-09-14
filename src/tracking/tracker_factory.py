from deep_sort_realtime.deepsort_tracker import DeepSort


def init_tracker(tracker_type: str = 'botsort'):
    tracker_type = tracker_type.lower()

    if tracker_type == 'botsort': # Ultralytics BotSort
        return 'botsort'
    if tracker_type == 'bytetrack': # Ultralytics ByteTrack
        return 'bytetrack'
    
    # NOT USED
    if tracker_type == 'deepsort':
        return DeepSort(max_age=30, n_init=3, max_cosine_distance=0.2)

    raise ValueError(
        f"Unknown tracker type: {tracker_type}. Choose from: 'botsort', 'bytetrack'"
    )