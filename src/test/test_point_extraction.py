import numpy as np
import cv2

from src.map_tools.map_loader import load_map_image
from src.map_tools.map_click_ab import ensure_AB_points
from src.test._test_utils import TestContext
from src.path.path_extraction import extract_path_mask, validate_A_B_inside_mask
from src.path.centerline import extract_centerline_points


TEST_ID = "P3_STEP1_001"
TEST_NAME = "Map + A/B functional validation"
TEST_DESCRIPTION = "Ensures map loads and A/B points exist and are valid."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir=results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    # -------------------------------------------------
    # TEST 1 — Map loads
    # -------------------------------------------------
    try:
        img = load_map_image()
        h, w = img.shape[:2]
        ctx.pass_(f"Map loaded successfully ({w}x{h})")
    except Exception as e:
        ctx.fail(f"Map failed to load: {e}")
        return False

    # -------------------------------------------------
    # TEST 2 — Map dimensions valid
    # -------------------------------------------------
    if h > 0 and w > 0:
        ctx.pass_("Map dimensions valid")
    else:
        ctx.fail("Map dimensions invalid")
        ok = False

    # -------------------------------------------------
    # TEST 3 — Ensure A/B exist (pipeline-safe)
    # -------------------------------------------------
    try:
        A, B = ensure_AB_points()
        A = np.array(A, dtype=int)
        B = np.array(B, dtype=int)

        ctx.pass_(f"A obtained: {A}")
        ctx.pass_(f"B obtained: {B}")

    except Exception as e:
        ctx.fail(f"Failed to obtain A/B: {e}")
        return False

    # -------------------------------------------------
    # TEST 4 — A/B inside map bounds
    # -------------------------------------------------
    def inside(p):
        return 0 <= int(p[0]) < w and 0 <= int(p[1]) < h

    if inside(A):
        ctx.pass_("A inside map bounds")
    else:
        ctx.fail("A outside map bounds")
        ok = False

    if inside(B):
        ctx.pass_("B inside map bounds")
    else:
        ctx.fail("B outside map bounds")
        ok = False

    # -------------------------------------------------
    # TEST 5 — Save debug visualization (A/B)
    # -------------------------------------------------
    debug = img.copy()
    cv2.circle(debug, (int(A[0]), int(A[1])), 8, (0, 0, 255), -1)   # red = A
    cv2.circle(debug, (int(B[0]), int(B[1])), 8, (255, 0, 0), -1)   # blue = B

    save_path = ctx.path("map_with_A_B.png")
    cv2.imwrite(save_path, debug)
    ctx.info(f"Saved debug image: {save_path}")

    # -------------------------------------------------
    # TEST 6 — Path mask extraction
    # -------------------------------------------------
    try:
        mask, gray, wall_mask = extract_path_mask(debug=True)
        ctx.pass_("Path mask extracted")
    except Exception as e:
        ctx.fail(f"Path mask extraction failed: {e}")
        return False

    # mask size check
    if mask.shape[0] == h and mask.shape[1] == w:
        ctx.pass_("Mask size matches map")
    else:
        ctx.fail("Mask size mismatch")
        ok = False

    # mask not empty
    if np.sum(mask) > 0:
        ctx.pass_("Mask contains walkable area")
    else:
        ctx.fail("Mask is empty")
        ok = False

    # A/B inside mask
    try:
        validate_A_B_inside_mask(mask)
        ctx.pass_("A and B lie inside walkable path")
    except Exception as e:
        ctx.fail(str(e))
        ok = False

    # -------------------------------------------------
    # DEBUG VISUALIZATION — mask overlay
    # -------------------------------------------------
    overlay = img.copy()
    overlay[mask] = (0, 255, 0)  # green mask overlay
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    save_path = ctx.path("path_mask_overlay.png")
    cv2.imwrite(save_path, blended)
    ctx.info(f"Saved mask overlay: {save_path}")

    # -------------------------------------------------
    # TEST 7 — Centerline extraction (A->B)
    # -------------------------------------------------
    try:
        pts = extract_centerline_points(mask=mask, A=A, B=B)
        ctx.pass_(f"Centerline extracted with {len(pts)} points")
    except Exception as e:
        ctx.fail(f"Centerline extraction failed: {e}")
        return False

    # centerline sanity
    if len(pts) > 10:
        ctx.pass_("Centerline has sufficient points")
    else:
        ctx.fail("Centerline too short")
        ok = False

    # -------------------------------------------------
    # DEBUG VISUALIZATION — centerline points overlay
    # -------------------------------------------------
    debug2 = img.copy()
    for p in pts.astype(int):
        x, y = int(p[0]), int(p[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(debug2, (x, y), 1, (0, 0, 255), -1)

    save_path2 = ctx.path("centerline_debug.png")
    cv2.imwrite(save_path2, debug2)
    ctx.info(f"Saved centerline debug: {save_path2}")

    return ok
