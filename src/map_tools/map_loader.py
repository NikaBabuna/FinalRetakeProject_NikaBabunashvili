import os
import cv2
from src.utils.io_utils import get_project_root
from src import config


def get_map_path() -> str:
    """
    Returns absolute path to the map image.
    """
    project_root = get_project_root()
    return os.path.join(project_root, "data", config.MAP_FILENAME)


def load_map_image():
    """
    Loads the map image using OpenCV.

    Returns:
        image (numpy array BGR)
    """
    map_path = get_map_path()

    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Map not found at: {map_path}")

    img = cv2.imread(map_path)
    if img is None:
        raise RuntimeError("Failed to load map image.")

    return img


def get_map_shape():
    """
    Returns (height, width) of map.
    """
    img = load_map_image()
    h, w = img.shape[:2]
    return h, w


def get_A_B():
    """
    Returns A and B pixel coordinates from config.
    """
    if config.A_PIX is None or config.B_PIX is None:
        raise RuntimeError(
            "A_PIX and B_PIX not set in config.py. Run click tool first."
        )

    return config.A_PIX, config.B_PIX
