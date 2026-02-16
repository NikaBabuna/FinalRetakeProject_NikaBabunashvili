import cv2
import numpy as np

from src.map_tools.map_loader import load_map_image
from src.map_tools.map_click_ab import ensure_AB_points


# ============================================================
# WALKABLE REGION EXTRACTION (FLOOD FILL)
# ============================================================

def extract_path_mask(debug: bool = False):
    """
    Computes walkable region by flood-filling from A.
    Walls are detected as dark pixels.

    Returns:
        mask (H,W) bool
    If debug=True also returns:
        gray image
        wall_mask
    """

    img = load_map_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------
    # Detect walls (dark pixels)
    # ------------------------------------------------
    wall_mask = gray < 100

    # Dilate walls slightly to close gaps
    kernel = np.ones((3, 3), np.uint8)
    wall_mask = cv2.dilate(wall_mask.astype(np.uint8) * 255, kernel, iterations=2)
    wall_mask = wall_mask > 0

    # ------------------------------------------------
    # Flood fill from A (walkable region)
    # ------------------------------------------------
    A, B = ensure_AB_points()

    h, w = gray.shape
    visited = np.zeros((h, w), dtype=bool)

    stack = [(int(A[0]), int(A[1]))]

    while stack:
        x, y = stack.pop()

        # bounds
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        # already visited
        if visited[y, x]:
            continue

        # wall
        if wall_mask[y, x]:
            continue

        visited[y, x] = True

        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))

    mask = visited

    # ------------------------------------------------
    # Debug return
    # ------------------------------------------------
    if debug:
        return mask, gray, wall_mask

    return mask


# ============================================================
# VALIDATION
# ============================================================

def validate_A_B_inside_mask(mask):
    """
    Ensures A and B lie inside walkable region.
    """
    A, B = ensure_AB_points()

    if not mask[int(A[1]), int(A[0])]:
        raise RuntimeError("A not inside walkable region")

    if not mask[int(B[1]), int(B[0])]:
        raise RuntimeError("B not inside walkable region")

    return True
