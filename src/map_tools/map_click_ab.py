# src/map_tools/map_click_ab.py

import os
import json
import cv2

from src.map_tools.map_loader import load_map_image
from src.utils.io_utils import get_project_root
from src import config


def _ab_json_path() -> str:
    project_root = get_project_root()
    return os.path.join(project_root, "data", "ab_points.json")


def _load_ab_from_json():
    path = _ab_json_path()
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        A = tuple(data.get("A", None))
        B = tuple(data.get("B", None))
        if A is None or B is None:
            return None
        if len(A) != 2 or len(B) != 2:
            return None
        return (int(A[0]), int(A[1])), (int(B[0]), int(B[1]))
    except Exception:
        return None


def _save_ab_to_json(A, B):
    path = _ab_json_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"A": [int(A[0]), int(A[1])], "B": [int(B[0]), int(B[1])]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _click_points_ui():
    """
    Opens OpenCV window and lets user click A and B.
    Returns:
        A (x,y), B (x,y)
    """
    print("======================================")
    print("Select A and B on map")
    print("Click start point A, then end point B")
    print("Press ESC to cancel")
    print("======================================")

    img = load_map_image()
    display = img.copy()

    points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Clicked: ({x}, {y})")
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow("Select A and B", display)

    cv2.imshow("Select A and B", display)
    cv2.setMouseCallback("Select A and B", mouse_callback)

    while True:
        cv2.imshow("Select A and B", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        if len(points) >= 2:
            break

    cv2.destroyAllWindows()

    if len(points) < 2:
        raise RuntimeError("A/B selection cancelled or incomplete.")

    return points[0], points[1]


def ensure_AB_points():
    """
    Ensures config.A_PIX and config.B_PIX exist.
    Priority:
      - If FORCE_AB_RESELECT=True: ALWAYS launch UI (ignore config + ignore json)
      - Else:
          1) use config.A_PIX/B_PIX if already set
          2) else load from data/ab_points.json
          3) else launch UI and save to json
    Returns:
        A, B
    """

    # 0) FORCE: always reselect, ignoring config + JSON
    if getattr(config, "FORCE_AB_RESELECT", False):
        print("[INFO] FORCE_AB_RESELECT=True -> launching selection UI (ignoring saved A/B).")
        A, B = _click_points_ui()

        config.A_PIX = A
        config.B_PIX = B

        # IMPORTANT: turn off force mode so we don't reopen the UI on the next call
        config.FORCE_AB_RESELECT = False
        print("[INFO] FORCE_AB_RESELECT auto-reset to False (won't ask again this run).")

        try:
            _save_ab_to_json(A, B)
            print(f"[INFO] Saved A/B to: {_ab_json_path()}")
        except Exception as e:
            print(f"[WARN] Could not save A/B json: {e}")

        print(f"[INFO] Stored A = {A}")
        print(f"[INFO] Stored B = {B}")
        return A, B

    # 1) Use existing from config
    if config.A_PIX is not None and config.B_PIX is not None:
        print("[INFO] Using existing A/B from config")
        return config.A_PIX, config.B_PIX

    # 2) Try load from JSON
    loaded = _load_ab_from_json()
    if loaded is not None:
        A, B = loaded
        config.A_PIX = A
        config.B_PIX = B
        print(f"[INFO] Loaded A/B from {os.path.basename(_ab_json_path())}: A={A}, B={B}")
        return A, B

    # 3) Fall back to UI
    print("[INFO] A/B not set. Launching selection UI...")
    A, B = _click_points_ui()

    config.A_PIX = A
    config.B_PIX = B

    try:
        _save_ab_to_json(A, B)
        print(f"[INFO] Saved A/B to: {_ab_json_path()}")
    except Exception as e:
        print(f"[WARN] Could not save A/B json: {e}")

    print(f"[INFO] Stored A = {A}")
    print(f"[INFO] Stored B = {B}")
    return A, B
