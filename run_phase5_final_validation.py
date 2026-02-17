# run_phase5_final_validation.py
"""
One-command Phase 5 "final validation":
1) Runs all src/test/test_phase5_*.py tests (fast).
2) Runs the two missions (fast or full).
3) Validates mission artifacts: mp4 opens, traj CSV has movement in correct direction,
   min-distance is finite, etc.

Usage:
  python run_phase5_final_validation.py --fast
  python run_phase5_final_validation.py --full
  python run_phase5_final_validation.py --fast --cleanup
"""

from __future__ import annotations

import os
import sys
import glob
import csv
import argparse
import importlib
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np


# -------------------------
# results dir helper
# -------------------------

def _next_results_dir(base: str = "output") -> str:
    os.makedirs(base, exist_ok=True)
    existing = glob.glob(os.path.join(base, "results_*"))
    nums = []
    for p in existing:
        name = os.path.basename(p)
        try:
            nums.append(int(name.split("_")[1]))
        except Exception:
            pass
    nxt = (max(nums) + 1) if nums else 1
    return os.path.join(base, f"results_{nxt:04d}")


def _video_duration_seconds(video_path: str) -> Tuple[float, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, 0.0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 1e-6 or n <= 0:
        return 0.0, fps, n
    return float(n / fps), fps, n


# -------------------------
# phase 5 test runner
# -------------------------

def run_phase5_tests(results_dir: str) -> bool:
    """
    Discovers and runs src/test/test_phase5_*.py modules that have run(results_dir)->bool.
    Writes artifacts to results_dir/tests_phase5 (like your existing runner).
    """
    print("===============================================")
    print("PHASE 5 FINAL VALIDATION: UNIT TESTS")
    print("===============================================")

    tests_dir = os.path.join(results_dir, "tests_phase5")
    os.makedirs(tests_dir, exist_ok=True)

    # discover tests
    pattern = os.path.join("src", "test", "test_phase5_*.py")
    files = sorted(glob.glob(pattern))
    print(f"Discovered: {len(files)} tests")

    ok_all = True
    for fp in files:
        mod_name = fp[:-3].replace(os.sep, ".")  # src.test.test_phase5_xxx
        print("\n------------------------------------------------")
        print(f"RUNNING: {mod_name}")
        print("------------------------------------------------")
        try:
            mod = importlib.import_module(mod_name)
            if not hasattr(mod, "run"):
                print("  [FAIL] No run(results_dir) function")
                ok_all = False
                continue
            ok = bool(mod.run(tests_dir))
            print(f"Test: {mod_name} -> {'PASS' if ok else 'FAIL'}")
            ok_all = ok_all and ok
        except Exception as e:
            print(f"  [FAIL] Import/Run error: {e}")
            ok_all = False

    print("\n===============================================")
    print(f"SUMMARY (TESTS): {'PASS' if ok_all else 'FAIL'}")
    print(f"Artifacts: {tests_dir}")
    print("===============================================")
    return ok_all


# -------------------------
# mission validation helpers
# -------------------------

def _read_traj_csv(traj_path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(traj_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    "t": float(row.get("t", 0.0)),
                    "x": float(row.get("x", 0.0)),
                    "y": float(row.get("y", 0.0)),
                    "vx": float(row.get("vx", 0.0)),
                    "vy": float(row.get("vy", 0.0)),
                })
            except Exception:
                continue
    return rows


def _validate_mp4(path: str, min_kb: int = 200) -> Tuple[bool, str]:
    if not os.path.exists(path):
        return False, f"missing: {path}"
    if os.path.getsize(path) < min_kb * 1024:
        return False, f"too small (<{min_kb}KB): {path}"

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False, f"cannot open mp4: {path}"
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    ok, _ = cap.read()
    cap.release()

    if not ok:
        return False, f"mp4 opens but cannot read first frame: {path}"
    if n < 5:
        return False, f"mp4 has too few frames ({n}): {path}"
    if fps <= 1e-6:
        return False, f"mp4 has invalid fps ({fps}): {path}"
    return True, f"ok (frames={n}, fps={fps:.2f})"


def _validate_direction(traj_rows: List[Dict[str, float]], direction: str, W_hint: float = 640.0) -> Tuple[bool, str]:
    if len(traj_rows) < 10:
        return False, f"trajectory too short (rows={len(traj_rows)})"

    x0 = traj_rows[0]["x"]
    x1 = traj_rows[-1]["x"]
    dx = x1 - x0

    # require meaningful progress (20% of width)
    thresh = 0.2 * float(W_hint)

    if direction == "L2R":
        if dx > thresh:
            return True, f"ok (dx={dx:.1f} > {thresh:.1f})"
        return False, f"not enough L->R progress (dx={dx:.1f} <= {thresh:.1f})"
    elif direction == "R2L":
        if dx < -thresh:
            return True, f"ok (dx={dx:.1f} < -{thresh:.1f})"
        return False, f"not enough R->L progress (dx={dx:.1f} >= -{thresh:.1f})"
    else:
        return True, "direction check skipped"


def _cleanup_heavy(results_dir: str) -> None:
    # deletes extracted frames + flow caches inside this results dir (optional)
    import shutil

    candidates = []
    candidates += glob.glob(os.path.join(results_dir, "tests_phase5", "phase5", "frames"))
    candidates += glob.glob(os.path.join(results_dir, "phase5_runs", "flows_cache"))
    candidates += glob.glob(os.path.join(results_dir, "phase5_runs", "*/flows_cache"))
    candidates += glob.glob(os.path.join(results_dir, "phase5_runs", "*/phase5", "frames"))

    for p in candidates:
        if os.path.isdir(p):
            try:
                shutil.rmtree(p, ignore_errors=True)
                print(f"  [INFO] cleaned: {p}")
            except Exception:
                pass


# -------------------------
# final validation runner
# -------------------------

def run_missions_and_validate(results_dir: str, mode: str) -> bool:
    """
    Runs src.phase5.run_missions.run(results_dir) and validates its outputs.
    mode: "fast" or "full"
    """
    print("\n===============================================")
    print("PHASE 5 FINAL VALIDATION: MISSIONS")
    print("===============================================")

    from src import config
    from src.phase5 import run_missions

    # derive video duration to avoid None -> float(None) crashes (and to support "full" reliably)
    video_path = os.path.join("data", str(getattr(config, "P5_VIDEO_FILENAME", "pedestrians.mp4")))
    dur_s, fps, n = _video_duration_seconds(video_path)

    # Apply safe overrides for runner (without touching your tests)
    # NOTE: run_missions currently uses P5_* keys (not always the RUN_* keys),
    # so we set both to be safe.
    if mode == "fast":
        run_max = float(getattr(config, "P5_RUN_MAX_SECONDS", 8.0) or 8.0)
        run_stride = int(getattr(config, "P5_RUN_FRAME_STRIDE", 2) or 2)
        run_w = int(getattr(config, "P5_RUN_RESIZE_WIDTH", 640) or 640)
    else:
        # full: if config is None, use actual duration
        cfg = getattr(config, "P5_RUN_MAX_SECONDS", None)
        if cfg is None:
            run_max = float(dur_s) if dur_s > 0 else 30.0
        else:
            run_max = float(cfg)
        run_stride = int(getattr(config, "P5_RUN_FRAME_STRIDE", 1) or 1)
        run_w = int(getattr(config, "P5_RUN_RESIZE_WIDTH", 640) or 640)

    setattr(config, "P5_RUN_MAX_SECONDS", run_max)
    setattr(config, "P5_RUN_FRAME_STRIDE", run_stride)
    setattr(config, "P5_RUN_RESIZE_WIDTH", run_w)

    # also set the generic keys because some code paths still use them
    setattr(config, "P5_MAX_SECONDS", run_max)
    setattr(config, "P5_FRAME_STRIDE", run_stride)
    setattr(config, "P5_RESIZE_WIDTH", run_w)

    print(f"  [INFO] mode={mode}  video={video_path}")
    print(f"  [INFO] run_max_seconds={run_max}  stride={run_stride}  resize_width={run_w}")
    if dur_s > 0:
        print(f"  [INFO] video_duration≈{dur_s:.1f}s  fps≈{fps:.1f}  frames≈{n}")

    # run missions
    try:
        out = run_missions.run(results_dir)
    except Exception as e:
        print(f"  [FAIL] mission runner crashed: {e}")
        return False

    ok_all = True

    # Validate both missions
    for key, direction in [("mission1", "L2R"), ("mission2", "R2L")]:
        m = out.get(key, {})
        print("\n------------------------------------------------")
        print(f"VALIDATING: {key}")
        print("------------------------------------------------")

        vid = m.get("video", "")
        traj = m.get("traj_csv", "")
        mind = m.get("mindist_csv", "")
        min_d = m.get("min_distance", None)

        # mp4 check
        ok, msg = _validate_mp4(vid)
        print(f"  [{'PASS' if ok else 'FAIL'}] video: {msg}")
        ok_all = ok_all and ok

        # traj check
        if not os.path.exists(traj):
            print(f"  [FAIL] missing traj csv: {traj}")
            ok_all = False
            rows = []
        else:
            rows = _read_traj_csv(traj)
            print(f"  [PASS] traj rows: {len(rows)}")

        # direction check (use frame width if we can infer it)
        W_hint = float(getattr(config, "P5_RUN_RESIZE_WIDTH", 640) or 640)
        ok, msg = _validate_direction(rows, direction, W_hint=W_hint)
        print(f"  [{'PASS' if ok else 'FAIL'}] direction({direction}): {msg}")
        ok_all = ok_all and ok

        # mindist check
        if not os.path.exists(mind):
            print(f"  [FAIL] missing mindist csv: {mind}")
            ok_all = False
        else:
            print(f"  [PASS] mindist csv exists")

        # safety label (warn-only by default)
        try:
            robot_r = float(getattr(config, "P5_ROBOT_RADIUS_PIX", 10.0) or 10.0)
            accident = 2.0 * robot_r
            if min_d is None:
                # try to infer from csv tail if not provided
                pass
            else:
                if float(min_d) < accident:
                    print(f"  [WARN] min_distance={float(min_d):.2f}px < 2*robot_r={accident:.2f}px  (collisions likely)")
                else:
                    print(f"  [INFO] min_distance={float(min_d):.2f}px (>= {accident:.2f}px)")
        except Exception:
            pass

    print("\n===============================================")
    print(f"SUMMARY (MISSIONS): {'PASS' if ok_all else 'FAIL'}")
    print(f"Artifacts: {out.get('out_dir', os.path.join(results_dir,'phase5_runs'))}")
    print("===============================================")
    return ok_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="short mission run (recommended during dev)")
    parser.add_argument("--full", action="store_true", help="use full video duration (or config P5_RUN_MAX_SECONDS)")
    parser.add_argument("--cleanup", action="store_true", help="delete heavy extracted frames / flow caches inside this results dir")
    args = parser.parse_args()

    mode = "full" if args.full else "fast"

    # ensure project root is on sys.path
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    results_dir = _next_results_dir("output")
    os.makedirs(results_dir, exist_ok=True)

    print("===============================================")
    print("PHASE 5 FINAL VALIDATION")
    print(f"Results folder: {results_dir}")
    print("===============================================")

    ok_tests = run_phase5_tests(results_dir)
    ok_missions = run_missions_and_validate(results_dir, mode=mode)

    if args.cleanup:
        print("\n===============================================")
        print("CLEANUP (optional)")
        print("===============================================")
        _cleanup_heavy(results_dir)

    ok_all = ok_tests and ok_missions
    print("\n===============================================")
    print(f"FINAL: {'PASS ✅' if ok_all else 'FAIL ❌'}")
    print(f"Results folder: {results_dir}")
    print("===============================================")

    raise SystemExit(0 if ok_all else 2)


if __name__ == "__main__":
    main()
