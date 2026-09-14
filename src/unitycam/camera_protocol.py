import struct
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class UnityCameraPacket:
    metadata_length_bytes: bytes
    metadata_length: int
    metadata: Dict[str, Any]
    camera_id: Optional[str]
    frame_id: Optional[int]
    timestamp_utc_ms: Optional[int]
    object_ids: List[str]
    image_length_bytes: bytes
    image_length: int
    image_bytes: bytes


def recv_all(sock, length):
    view = memoryview(bytearray(length)) # Pre-allocate memory
    pos = 0
    while pos < length:
        read = sock.recv_into(view[pos:], length - pos)
        if not read:
            raise EOFError('Socket closed')
        pos += read
    return view.tobytes()


def read_unity_camera_packet(sock) -> UnityCameraPacket:
    metadata_length_bytes = recv_all(sock, 4)
    metadata_length = struct.unpack('<I', metadata_length_bytes)[0]

    metadata_bytes = recv_all(sock, metadata_length)
    metadata_text = metadata_bytes.decode('utf-8')

    camera_id = None
    frame_id = None
    timestamp_utc_ms = None

    try:
        metadata = json.loads(metadata_text) if metadata_text else {}

        if isinstance(metadata, dict):
            camera_id = metadata.get('camera_id')
            frame_id = metadata.get('frame_id')
            timestamp_utc_ms = metadata.get('timestamp_utc_ms')
            object_ids = metadata.get('object_ids') or []
        else:
            metadata = {'raw': metadata_text}
            object_ids = []
    except json.JSONDecodeError:
        object_ids = metadata_text.split(',') if metadata_text else []
        metadata = {
            'raw': metadata_text,
            'object_ids': object_ids,
        }

    image_length_bytes = recv_all(sock, 4)
    image_length = struct.unpack('<I', image_length_bytes)[0]
    image_bytes = recv_all(sock, image_length)

    return UnityCameraPacket(
        metadata_length_bytes=metadata_length_bytes,
        metadata_length=metadata_length,
        metadata=metadata,
        camera_id=camera_id,
        frame_id=frame_id,
        timestamp_utc_ms=timestamp_utc_ms,
        object_ids=object_ids,
        image_length_bytes=image_length_bytes,
        image_length=image_length,
        image_bytes=image_bytes,
    )