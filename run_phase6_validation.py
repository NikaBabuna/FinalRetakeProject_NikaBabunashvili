# run_phase6_validation.py
"""
Phase 6 — Validation + test cases (ALL-in-one runner)

Fixes in this version:
  - Subproblem 2 (large N pass) no longer starts robots in collision:
      uses rejection sampling spawn enforcing min separation >= 2*robot_radius
  - Subproblem 3 camera-motion failure demo is robust:
      uses Farnebäck directly + phase correlation translation estimate

Usage:
  python run_phase6_validation.py --fast
  python run_phase6_validation.py --full
  python run_phase6_validation.py --fast --skip-video
  python run_phase6_validation.py --fast --cleanup
"""

from __future__ import annotations

import os
import sys
import glob
import json
import math
import shutil
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import cv2


# ----------------------------
# Small test logger
# ----------------------------

@dataclass
class Ctx:
    out_dir: str
    ok: bool = True
    metrics: Dict[str, Any] = field(default_factory=dict)

    def _p(self, level: str, msg: str) -> None:
        print(f"  [{level}] {msg}")

    def info(self, msg: str) -> None:
        self._p("INFO", msg)

    def pass_(self, msg: str) -> None:
        self._p("PASS", msg)

    def fail(self, msg: str) -> None:
        self._p("FAIL", msg)
        self.ok = False

    def warn(self, msg: str) -> None:
        self._p("WARN", msg)

    def save_json(self, name: str, obj: Dict[str, Any]) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        self.info(f"Saved: {path}")

    def save_image(self, name: str, img: np.ndarray) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, name)
        cv2.imwrite(path, img)
        self.info(f"Saved: {path}")

    def save_txt(self, name: str, text: str) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.info(f"Saved: {path}")


# ----------------------------
# Results dir helper
# ----------------------------

def next_results_dir(base: str = "output") -> str:
    os.makedirs(base, exist_ok=True)
    existing = glob.glob(os.path.join(base, "results_*"))
    nums: List[int] = []
    for p in existing:
        b = os.path.basename(p)
        try:
            nums.append(int(b.split("_")[1]))
        except Exception:
            pass
    nxt = (max(nums) + 1) if nums else 1
    return os.path.join(base, f"results_{nxt:04d}")


# ----------------------------
# Geometry utilities
# ----------------------------

def polyline_lengths(pts: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s


def closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return a.copy(), float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    return q, float(np.linalg.norm(p - q))


def distance_to_polyline(p: np.ndarray, pts: np.ndarray) -> float:
    best = float("inf")
    for i in range(len(pts) - 1):
        _, d = closest_point_on_segment(p, pts[i], pts[i + 1])
        if d < best:
            best = d
    return best


def point_at_arclength(pts: np.ndarray, s: np.ndarray, s_query: float) -> np.ndarray:
    s_query = float(np.clip(s_query, s[0], s[-1]))
    j = int(np.searchsorted(s, s_query))
    if j <= 0:
        return pts[0].copy()
    if j >= len(s):
        return pts[-1].copy()
    s0, s1 = float(s[j - 1]), float(s[j])
    if abs(s1 - s0) < 1e-9:
        return pts[j].copy()
    t = (s_query - s0) / (s1 - s0)
    return (1 - t) * pts[j - 1] + t * pts[j]


# ----------------------------
# Simple dynamics (lightweight validator)
# ----------------------------

def sat_vec(v: np.ndarray, vmax: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= vmax or n < 1e-12:
        return v
    return v * (float(vmax) / n)


def rk2_step(x: np.ndarray, v: np.ndarray, dt: float, accel_fn):
    a1 = accel_fn(x, v)
    x_mid = x + 0.5 * dt * v
    v_mid = v + 0.5 * dt * a1
    a2 = accel_fn(x_mid, v_mid)
    x_new = x + dt * v_mid
    v_new = v + dt * a2
    return x_new, v_new


# ----------------------------
# Subproblem 1: Path-following tests
# ----------------------------

def simulate_path_following(
    centerline: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    corridor_halfwidth: float,
    dt: float,
    steps: int,
    vmax: float,
    lookahead: float,
    kv: float,
    kd: float,
    k_border: float,
) -> Dict[str, Any]:
    pts = centerline.astype(np.float32)
    s = polyline_lengths(pts)

    x = start.astype(np.float32).copy()
    v = np.zeros((2,), dtype=np.float32)

    max_violation = 0.0
    traj = []

    def accel(xv: np.ndarray, vv: np.ndarray) -> np.ndarray:
        best_d = float("inf")
        best_s = 0.0
        best_q = pts[0]
        for i in range(len(pts) - 1):
            q, d = closest_point_on_segment(xv, pts[i], pts[i + 1])
            if d < best_d:
                best_d = d
                best_q = q
                seg_len = float(np.linalg.norm(pts[i + 1] - pts[i]))
                if seg_len < 1e-9:
                    best_s = float(s[i])
                else:
                    t = float(np.linalg.norm(q - pts[i]) / seg_len)
                    best_s = float(s[i] + t * seg_len)

        tgt = point_at_arclength(pts, s, best_s + lookahead)

        to_tgt = tgt - xv
        d_t = float(np.linalg.norm(to_tgt))
        dir_t = (to_tgt / d_t) if d_t > 1e-6 else np.zeros((2,), dtype=np.float32)

        to_goal = goal - xv
        d_g = float(np.linalg.norm(to_goal))
        dir_g = (to_goal / d_g) if d_g > 1e-6 else np.zeros((2,), dtype=np.float32)

        v_des = sat_vec((0.85 * dir_t + 0.15 * dir_g) * vmax, vmax)

        dist_c = distance_to_polyline(xv, pts)
        outside = max(0.0, dist_c - corridor_halfwidth)

        if outside > 0.0 and k_border > 0.0:
            dir_in = best_q - xv
            n = float(np.linalg.norm(dir_in))
            dir_in = (dir_in / n) if n > 1e-6 else np.zeros((2,), dtype=np.float32)
            a_border = (k_border * outside) * dir_in
        else:
            a_border = np.zeros((2,), dtype=np.float32)

        a = kv * (v_des - vv) - kd * vv + a_border
        return a.astype(np.float32)

    for k in range(steps):
        dist_c = distance_to_polyline(x, pts)
        viol = max(0.0, dist_c - corridor_halfwidth)
        max_violation = max(max_violation, float(viol))

        traj.append((float(x[0]), float(x[1])))

        x, v = rk2_step(x, v, dt, accel)
        v = sat_vec(v, vmax)

        if float(np.linalg.norm(goal - x)) < 10.0:
            break

    return {
        "max_border_violation": float(max_violation),
        "steps": int(len(traj)),
        "traj": traj,
    }


def draw_corridor_debug(centerline: np.ndarray, corridor_halfwidth: float, traj: List[Tuple[float, float]], W: int, H: int) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    pts = centerline.astype(np.int32)

    for i in range(len(pts) - 1):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), (120, 120, 120), 2)

    for p in pts[:: max(1, len(pts)//60)]:
        cv2.circle(img, (int(p[0]), int(p[1])), int(corridor_halfwidth), (40, 40, 40), 1)

    for i in range(1, len(traj)):
        a = (int(traj[i - 1][0]), int(traj[i - 1][1]))
        b = (int(traj[i][0]), int(traj[i][1]))
        cv2.line(img, a, b, (255, 255, 255), 2)

    return img


def test_sub1(ctx: Ctx, fast: bool) -> Tuple[bool, Dict[str, Any]]:
    ctx.info("SUBPROBLEM 1 — Path following synthetic tests")

    try:
        from src import config
        lookahead_default = float(getattr(config, "LOOKAHEAD", 25.0))
        dt = float(getattr(config, "DT", 0.05))
        vmax = float(getattr(config, "VMAX", 80.0) or 80.0)
        path_w = float(getattr(config, "PATH_WIDTH_PIX", getattr(config, "PATH_WIDTH", 60.0)))
    except Exception:
        lookahead_default, dt, vmax, path_w = 25.0, 0.05, 80.0, 60.0

    vmax = max(vmax, 60.0)

    corridor_half = 0.5 * path_w
    steps = 250 if fast else 700
    W, H = 900, 600

    results: Dict[str, Any] = {}

    center = np.stack([np.linspace(60, W - 60, 200), np.full(200, H * 0.5)], axis=1).astype(np.float32)
    out = simulate_path_following(center, center[0], center[-1], corridor_half, dt, steps, vmax, lookahead_default, 6.0, 0.6, 80.0)
    results["straight"] = out
    if out["max_border_violation"] <= 1e-6:
        ctx.pass_(f"Straight path: max border violation={out['max_border_violation']:.4f} (PASS)")
    else:
        ctx.fail(f"Straight path: max border violation={out['max_border_violation']:.4f} (should be 0)")
    ctx.save_image("sub1_straight_debug.png", draw_corridor_debug(center, corridor_half, out["traj"], W, H))

    xs = np.linspace(60, W - 60, 300)
    ys = (H * 0.5) + 90.0 * np.sin(2.0 * np.pi * xs / (W - 120))
    curvy = np.stack([xs, ys], axis=1).astype(np.float32)

    out = simulate_path_following(curvy, curvy[0], curvy[-1], corridor_half, dt, steps, vmax, lookahead_default, 6.0, 0.6, 120.0)
    results["curvy"] = out
    if out["max_border_violation"] <= 1e-6:
        ctx.pass_(f"Curvy path: max border violation={out['max_border_violation']:.4f} (PASS)")
    else:
        ctx.fail(f"Curvy path: max border violation={out['max_border_violation']:.4f} (should be 0)")
    ctx.save_image("sub1_curvy_debug.png", draw_corridor_debug(curvy, corridor_half, out["traj"], W, H))

    out = simulate_path_following(
        curvy, curvy[0], curvy[-1], corridor_half, dt, steps, vmax,
        lookahead=max(lookahead_default * 3.0, 120.0),
        kv=7.0, kd=0.3, k_border=0.0
    )
    results["failure_overshoot"] = out
    if out["max_border_violation"] > 0.5:
        ctx.pass_(f"Expected failure (overshoot): max border violation={out['max_border_violation']:.4f} (as expected)")
    else:
        ctx.warn(f"Expected failure did NOT fail strongly (violation={out['max_border_violation']:.4f}).")
    ctx.save_image("sub1_failure_overshoot_debug.png", draw_corridor_debug(curvy, corridor_half, out["traj"], W, H))

    ctx.save_json("sub1_metrics.json", results)
    return ctx.ok, results


# ----------------------------
# Subproblem 2: Swarm tests (FIXED SPAWN)
# ----------------------------

def swarm_step(
    X: np.ndarray,
    V: np.ndarray,
    goal_dir: np.ndarray,
    dt: float,
    vmax: float,
    rsafe: float,
    krep: float,
    kv: float,
    kd: float,
) -> Tuple[np.ndarray, np.ndarray]:
    N = X.shape[0]
    A = np.zeros_like(X, dtype=np.float32)

    v_des = (goal_dir[None, :] * vmax).astype(np.float32)

    for i in range(N):
        ai = np.zeros((2,), dtype=np.float32)
        for j in range(N):
            if i == j:
                continue
            dvec = X[i] - X[j]
            d = float(np.linalg.norm(dvec))
            if d < 1e-6:
                continue
            if d < rsafe:
                ai += (krep * (rsafe - d) / d) * (dvec / d)
        A[i] = ai

    A = kv * (v_des - V) - kd * V + A

    Xn = X + dt * V
    Vn = V + dt * A

    for i in range(N):
        Vn[i] = sat_vec(Vn[i], vmax)

    return Xn, Vn


def swarm_metrics(X: np.ndarray, robot_r: float) -> Tuple[int, float]:
    N = X.shape[0]
    min_d = float("inf")
    collisions = 0
    thresh = 2.0 * float(robot_r)
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(X[i] - X[j]))
            min_d = min(min_d, d)
            if d < thresh:
                collisions += 1
    if min_d == float("inf"):
        min_d = 0.0
    return collisions, min_d


def draw_swarm_debug(X: np.ndarray, W: int, H: int, robot_r: float) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(X.shape[0]):
        cv2.circle(img, (int(X[i, 0]), int(X[i, 1])), int(robot_r), (255, 255, 255), 1)
    return img


def spawn_points_rejection(
    N: int,
    W: int,
    H: int,
    margin: float,
    min_sep: float,
    rng: np.random.Generator,
    max_tries: int = 20000,
) -> np.ndarray:
    pts: List[np.ndarray] = []
    tries = 0
    while len(pts) < N and tries < max_tries:
        tries += 1
        x = rng.uniform(margin, W - margin)
        y = rng.uniform(margin, H - margin)
        p = np.array([x, y], dtype=np.float32)
        ok = True
        for q in pts:
            if float(np.linalg.norm(p - q)) < min_sep:
                ok = False
                break
        if ok:
            pts.append(p)
    if len(pts) < N:
        raise RuntimeError(f"Could not spawn {N} points with min_sep={min_sep} (got {len(pts)} after {tries} tries)")
    return np.stack(pts, axis=0).astype(np.float32)


def test_sub2(ctx: Ctx, fast: bool) -> Tuple[bool, Dict[str, Any]]:
    ctx.info("SUBPROBLEM 2 — Swarm tests")

    try:
        from src import config
        dt = float(getattr(config, "DT", 0.05))
        robot_r = float(getattr(config, "ROBOT_RADIUS", 6.0))
        rsafe = float(getattr(config, "RSAFE", 18.0))
        krep = float(getattr(config, "KREP", 25.0))
        vmax = float(getattr(config, "VMAX", 80.0) or 80.0)
        n_safe = int(getattr(config, "N_SAFE", 40))
    except Exception:
        dt, robot_r, rsafe, krep, vmax, n_safe = 0.05, 6.0, 18.0, 25.0, 80.0, 40

    vmax = max(vmax, 60.0)

    W, H = 900, 500
    steps = 160 if fast else 400
    goal_dir = np.array([1.0, 0.0], dtype=np.float32)

    results: Dict[str, Any] = {}
    rng = np.random.default_rng(0)

    def run_case(name: str, N: int, rsafe_case: float, expect_fail: bool) -> None:
        # IMPORTANT FIX: spawn with min separation so we don't start in collision
        min_sep_spawn = max(2.0 * robot_r + 1.0, 0.55 * rsafe_case)
        margin = max(20.0, robot_r + 5.0)

        try:
            X = spawn_points_rejection(N, W=int(0.55 * W), H=H, margin=margin, min_sep=min_sep_spawn, rng=rng)
            # bias to left side so they move right
            X[:, 0] = np.clip(X[:, 0], margin, 0.55 * W - margin)
        except Exception as e:
            # If we *expect* fail, it's fine to start dense.
            if expect_fail:
                ctx.warn(f"{name}: dense spawn fallback (expected fail). Reason: {e}")
                ys = rng.uniform(margin, H - margin, size=(N,)).astype(np.float32)
                xs = np.full((N,), 90.0, dtype=np.float32)
                X = np.stack([xs, ys], axis=1).astype(np.float32)
            else:
                ctx.fail(f"{name}: could not spawn non-colliding initial state: {e}")
                return

        V = np.zeros_like(X, dtype=np.float32)

        worst_coll = 0
        min_dist_global = float("inf")

        for _ in range(steps):
            X, V = swarm_step(
                X, V, goal_dir,
                dt=dt,
                vmax=vmax,
                rsafe=rsafe_case,
                krep=krep,
                kv=4.0,
                kd=0.5,
            )
            X[:, 0] = np.clip(X[:, 0], robot_r, W - 1 - robot_r)
            X[:, 1] = np.clip(X[:, 1], robot_r, H - 1 - robot_r)

            coll, min_d = swarm_metrics(X, robot_r)
            worst_coll = max(worst_coll, coll)
            min_dist_global = min(min_dist_global, min_d)

        results[name] = {
            "N": int(N),
            "rsafe": float(rsafe_case),
            "collision_count_max": int(worst_coll),
            "min_inter_robot_distance": float(min_dist_global),
            "required_min_distance": float(2.0 * robot_r),
            "spawn_min_sep": float(min_sep_spawn),
        }

        ctx.save_image(f"{name}_debug.png", draw_swarm_debug(X, W, H, robot_r))

        if expect_fail:
            if worst_coll > 0 or min_dist_global < 2.0 * robot_r:
                ctx.pass_(f"{name}: expected failure observed (collisions={worst_coll}, min_d={min_dist_global:.2f})")
            else:
                ctx.warn(f"{name}: expected failure did NOT fail (collisions={worst_coll}, min_d={min_dist_global:.2f})")
        else:
            if worst_coll == 0 and min_dist_global >= 2.0 * robot_r:
                ctx.pass_(f"{name}: collisions=0 and min_d={min_dist_global:.2f} (PASS)")
            else:
                ctx.fail(f"{name}: collisions={worst_coll}, min_d={min_dist_global:.2f} (should be 0 collisions and min_d >= {2.0*robot_r:.2f})")

    # small N pass
    run_case("sub2_smallN_pass", N=10, rsafe_case=max(rsafe, 3.5 * robot_r), expect_fail=False)

    # larger N pass (use tuned N_SAFE but keep spawn feasible)
    N_big = int(max(20, min(n_safe, 60)))
    run_case("sub2_largeN_pass", N=N_big, rsafe_case=max(rsafe, 3.5 * robot_r), expect_fail=False)

    # failure: too large N + too small rsafe
    run_case("sub2_failure_collisions", N=120, rsafe_case=1.2 * robot_r, expect_fail=True)

    ctx.save_json("sub2_metrics.json", results)
    return ctx.ok, results


# ----------------------------
# Subproblem 3: Video tests
# ----------------------------

def video_duration_seconds(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 1e-6 or n <= 0:
        return 0.0
    return float(n / fps)


def farneback_flow(prev_gray: np.ndarray, nxt_gray: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        prev_gray, nxt_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    ).astype(np.float32)


def test_sub3(ctx: Ctx, fast: bool) -> Tuple[bool, Dict[str, Any]]:
    ctx.info("SUBPROBLEM 3 — Video tests")

    results: Dict[str, Any] = {}

    # ----- PASS case: run your Phase 5 mission runner -----
    try:
        from src import config
        from src.phase5 import run_missions

        video_path = os.path.join("data", str(getattr(config, "P5_VIDEO_FILENAME", "pedestrians.mp4")))
        dur = video_duration_seconds(video_path)

        run_max = 8.0 if fast else (dur if dur > 0 else 30.0)
        stride = 2 if fast else 1
        resize_w = int(getattr(config, "P5_RUN_RESIZE_WIDTH", 640) or 640)

        setattr(config, "P5_RUN_MAX_SECONDS", float(run_max))
        setattr(config, "P5_RUN_FRAME_STRIDE", int(stride))
        setattr(config, "P5_RUN_RESIZE_WIDTH", int(resize_w))

        setattr(config, "P5_MAX_SECONDS", float(run_max))
        setattr(config, "P5_FRAME_STRIDE", int(stride))
        setattr(config, "P5_RESIZE_WIDTH", int(resize_w))

        ctx.info(f"Running Phase5 missions as 'Video A works' (max_seconds={run_max}, stride={stride}, resize={resize_w})")
        out = run_missions.run(ctx.out_dir)

        rsafe = float(getattr(config, "P5_RSAFE_PIX", 45.0))
        m1 = out.get("mission1", {})
        m2 = out.get("mission2", {})

        m1_min = float(m1.get("min_distance", float("inf")))
        m2_min = float(m2.get("min_distance", float("inf")))

        results["videoA"] = {
            "rsafe": rsafe,
            "mission1_min_distance": m1_min,
            "mission2_min_distance": m2_min,
            "out_dir": out.get("out_dir", ""),
        }

        if (m1_min >= rsafe) and (m2_min >= rsafe):
            ctx.pass_(f"Video A: min distances OK (m1={m1_min:.2f} >= {rsafe}, m2={m2_min:.2f} >= {rsafe})")
        else:
            ctx.fail(
                f"Video A: min distance below Rsafe "
                f"(m1={m1_min:.2f}, m2={m2_min:.2f}, Rsafe={rsafe}). "
                f"Likely tuning: increase P5_KREP_PED / P5_REP_MAX or reduce P5_GOAL_SPEED / P5_VMAX."
            )

    except Exception as e:
        ctx.fail(f"Video A test could not run Phase5 mission runner: {e}")
        results["videoA"] = {"error": str(e)}

    # ----- Failure case: heavy camera motion dominates optical flow (robust demo) -----
    try:
        from src import config
        video_path = os.path.join("data", str(getattr(config, "P5_VIDEO_FILENAME", "pedestrians.mp4")))
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frames = []
        for _ in range(6):
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if len(frames) < 3:
            raise RuntimeError("Not enough frames for camera-motion failure demo")

        H, W = frames[0].shape[:2]

        # Create deterministic "camera pan": shift each subsequent frame by (dx,dy)
        dx, dy = 14, -10
        shaken = []
        for i, fr in enumerate(frames):
            M = np.array([[1, 0, i * dx], [0, 1, i * dy]], dtype=np.float32)
            shaken_fr = cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            shaken.append(shaken_fr)

        prev = cv2.cvtColor(shaken[0], cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(shaken[1], cv2.COLOR_BGR2GRAY)

        # Flow using Farnebäck directly (robust baseline)
        flow = farneback_flow(prev, nxt)
        mag = np.linalg.norm(flow, axis=2)
        med_mag = float(np.median(mag))
        p95_mag = float(np.percentile(mag, 95))

        # Also estimate global translation by phase correlation
        prev_f = np.float32(prev)
        nxt_f = np.float32(nxt)
        shift, response = cv2.phaseCorrelate(prev_f, nxt_f)
        est_dx, est_dy = float(shift[0]), float(shift[1])

        results["camera_motion_failure_demo"] = {
            "true_shift_px": {"dx": dx, "dy": dy},
            "phasecorr_est_shift_px": {"dx": est_dx, "dy": est_dy, "response": float(response)},
            "median_flow_mag": med_mag,
            "p95_flow_mag": p95_mag,
            "note": "Camera motion creates strong global flow; pedestrian-flow-as-velocity-field assumption breaks unless compensated.",
        }

        # expected failure: global motion should inflate magnitude
        if med_mag > 1.0 or p95_mag > 4.0:
            ctx.pass_(f"Expected failure demo: camera motion inflates flow (median={med_mag:.2f}, p95={p95_mag:.2f}, est_shift=({est_dx:.1f},{est_dy:.1f}))")
        else:
            ctx.warn(f"Expected failure demo not dramatic (median={med_mag:.2f}, p95={p95_mag:.2f})")

        vis = np.hstack([frames[1], shaken[1]])
        cv2.putText(vis, "original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(vis, "camera-motion", (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
        ctx.save_image("sub3_camera_motion_failure_demo.png", vis)

        ctx.save_txt(
            "sub3_camera_motion_failure_explanation.txt",
            "Failure case: heavy camera motion breaks optical flow assumptions.\n"
            "Dense optical flow returns apparent motion for *everything*, including background.\n"
            "If the camera pans/shakes, the flow is dominated by global motion.\n"
            "Fixes would include stabilization, global motion estimation (e.g., homography),\n"
            "or subtracting dominant global translation/affine motion before using flow as V(x,t).\n"
        )

    except Exception as e:
        ctx.warn(f"Camera-motion failure demo skipped: {e}")
        results["camera_motion_failure_demo"] = {"skipped": True, "reason": str(e)}

    ctx.save_json("sub3_metrics.json", results)
    return ctx.ok, results


# ----------------------------
# Cleanup helper (optional)
# ----------------------------

def cleanup_heavy(results_dir: str) -> None:
    candidates = []
    candidates += glob.glob(os.path.join(results_dir, "phase5_runs", "**", "flows_cache"), recursive=True)
    candidates += glob.glob(os.path.join(results_dir, "**", "frames"), recursive=True)

    for p in sorted(set(candidates)):
        if os.path.isdir(p):
            try:
                shutil.rmtree(p, ignore_errors=True)
                print(f"  [INFO] cleaned: {p}")
            except Exception:
                pass


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args()

    fast = True if args.fast or not args.full else False

    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    results_dir = next_results_dir("output")
    os.makedirs(results_dir, exist_ok=True)

    out_dir = os.path.join(results_dir, "phase6_validation")
    os.makedirs(out_dir, exist_ok=True)

    print("===============================================")
    print("PHASE 6 — VALIDATION + TEST CASES")
    print(f"Results folder: {results_dir}")
    print(f"Mode: {'FAST' if fast else 'FULL'}")
    print("===============================================")

    all_ok = True
    summary: Dict[str, Any] = {}

    print("\n-----------------------------------------------")
    print("SUBPROBLEM 1")
    print("-----------------------------------------------")
    ctx1 = Ctx(out_dir=os.path.join(out_dir, "subproblem1"))
    ok1, res1 = test_sub1(ctx1, fast=fast)
    all_ok = all_ok and ok1
    summary["subproblem1"] = {"ok": ok1, "metrics": res1}

    print("\n-----------------------------------------------")
    print("SUBPROBLEM 2")
    print("-----------------------------------------------")
    ctx2 = Ctx(out_dir=os.path.join(out_dir, "subproblem2"))
    ok2, res2 = test_sub2(ctx2, fast=fast)
    all_ok = all_ok and ok2
    summary["subproblem2"] = {"ok": ok2, "metrics": res2}

    if not args.skip_video:
        print("\n-----------------------------------------------")
        print("SUBPROBLEM 3")
        print("-----------------------------------------------")
        ctx3 = Ctx(out_dir=os.path.join(out_dir, "subproblem3"))
        ok3, res3 = test_sub3(ctx3, fast=fast)
        all_ok = all_ok and ok3
        summary["subproblem3"] = {"ok": ok3, "metrics": res3}
    else:
        summary["subproblem3"] = {"ok": None, "skipped": True}

    with open(os.path.join(out_dir, "phase6_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===============================================")
    print(f"PHASE 6 FINAL: {'PASS ✅' if all_ok else 'FAIL ❌'}")
    print(f"Artifacts: {out_dir}")
    print("===============================================")

    if args.cleanup:
        print("\n===============================================")
        print("CLEANUP")
        print("===============================================")
        cleanup_heavy(results_dir)

    raise SystemExit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
