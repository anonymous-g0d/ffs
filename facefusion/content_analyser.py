from typing import Any
from functools import lru_cache
from time import sleep
import cv2
import numpy
import onnxruntime
from tqdm import tqdm

import facefusion.globals
from facefusion import process_manager, wording
from facefusion.thread_helper import thread_lock, conditional_thread_semaphore
from facefusion.typing import VisionFrame, ModelSet, Fps
from facefusion.execution import apply_execution_provider_options
from facefusion.vision import get_video_frame, count_video_frame_total, read_image, detect_video_fps
from facefusion.filesystem import resolve_relative_path, is_file
from facefusion.download import conditional_download

CONTENT_ANALYSER = None
MODELS: ModelSet = {
    'open_nsfw': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/open_nsfw.onnx',
        'path': resolve_relative_path('../.assets/models/open_nsfw.onnx')
    }
}
PROBABILITY_LIMIT = 0.80
RATE_LIMIT = 10
STREAM_COUNTER = 0

def get_content_analyser() -> Any:
    return None

def clear_content_analyser() -> None:
    pass

def pre_check() -> bool:
    return True

def analyse_stream(vision_frame: VisionFrame, video_fps: Fps) -> bool:
    return False

def analyse_frame(vision_frame: VisionFrame) -> bool:
    return False

def prepare_frame(vision_frame: VisionFrame) -> VisionFrame:
    return vision_frame

@lru_cache(maxsize=None)
def analyse_image(image_path: str) -> bool:
    return False

@lru_cache(maxsize=None)
def analyse_video(video_path: str, start_frame: int, end_frame: int) -> bool:
    video_frame_total = count_video_frame_total(video_path)
    frame_range = range(start_frame or 0, end_frame or video_frame_total)

    with tqdm(total=len(frame_range), desc=wording.get('analysing'), unit='frame', ascii=' =', disable=facefusion.globals.log_level in ['warn', 'error']) as progress:
        for frame_number in frame_range:
            progress.update()
    return False
