# src/phase5/run_missions.py
from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src import config
from src.phase5.video_frames import extract_frames
from src.phase5.optical_flow import compute_dense_flow
from src.phase5.ped_detect import PedestrianDetectorMOG2, boxes_to_centroids
from src.phase5.repulsion import repulsion_force, min_distance


@dataclass
class Mission:
    name: str
    start: Tuple[float, float]
    goal: Tuple[float, float]


class SimpleNNTracker:
    def __init__(self, dist_gate: float = 35.0, max_missed: int = 8):
        self.dist_gate = float(dist_gate)
        self.max_missed = int(max_missed)
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}

    def update(self, detections_xy: np.ndarray) -> List[Dict]:
        dets = detections_xy.astype(np.float32, copy=False)
        used = np.zeros((dets.shape[0],), dtype=bool)

        for tid in list(self.tracks.keys()):
            self.tracks[tid]["missed"] += 1

        for tid, tr in list(self.tracks.items()):
            if dets.shape[0] == 0:
                continue
            d = np.linalg.norm(dets - tr["xy"][None, :], axis=1)
            j = int(np.argmin(d))
            if (not used[j]) and float(d[j]) <= self.dist_gate:
                tr["xy"] = dets[j].copy()
                tr["missed"] = 0
                tr["age"] += 1
                used[j] = True

        for j in range(dets.shape[0]):
            if used[j]:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"id": tid, "xy": dets[j].copy(), "age": 1, "missed": 0}

        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["missed"] > self.max_missed:
                del self.tracks[tid]

        out = []
        for tid, tr in self.tracks.items():
            out.append({"id": tid, "x": float(tr["xy"][0]), "y": float(tr["xy"][1])})
        return out


def _get_cfg(name: str, default):
    return getattr(config, name, default)


def _as_float_or_none(v):
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() == "none":
        return None
    return float(v)


def _as_int_or_none(v):
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() == "none":
        return None
    return int(v)


def _sat_vec(v: np.ndarray, vmax: float, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= vmax:
        return v
    return v * (vmax / max(n, eps))


def _sample_flow_bilinear(flow: np.ndarray, xy: Tuple[float, float]) -> np.ndarray:
    """
    flow: (H,W,2) in px/frame
    xy: (x,y) in pixel coords
    returns 2-vector float32 in px/frame
    """
    H, W = flow.shape[:2]
    x, y = float(xy[0]), float(xy[1])
    x = np.clip(x, 0.0, W - 1.001)
    y = np.clip(y, 0.0, H - 1.001)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)

    dx = x - x0
    dy = y - y0

    f00 = flow[y0, x0].astype(np.float32)
    f10 = flow[y0, x1].astype(np.float32)
    f01 = flow[y1, x0].astype(np.float32)
    f11 = flow[y1, x1].astype(np.float32)

    f0 = f00 * (1 - dx) + f10 * dx
    f1 = f01 * (1 - dx) + f11 * dx
    return f0 * (1 - dy) + f1 * dy


def make_two_missions(W: int, H: int) -> Tuple[Mission, Mission]:
    margin = int(_get_cfg("P5_MISSION_MARGIN", 30))
    y = H * 0.5
    m1 = Mission("mission1_left_to_right", (margin, y), (W - margin, y))
    m2 = Mission("mission2_right_to_left", (W - margin, y), (margin, y))
    return m1, m2


def run_mission_on_frames(
    results_dir: str,
    mission: Mission,
    frame_paths: List[str],
    fps: float,
    out_dir: str,
    stride_used: int,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    flow_method = str(_get_cfg("P5_FLOW_METHOD", "farneback"))

    # IMPORTANT: From here onward, VT dynamics runs in px/sec.
    vmax_sec = float(_get_cfg("P5_VMAX", 120.0))  # px/sec
    rsafe = float(_get_cfg("P5_RSAFE_PIX", 45.0))
    robot_r = float(_get_cfg("P5_ROBOT_RADIUS_PIX", 10.0))

    kv = float(_get_cfg("P5_KV", 6.0))
    kd = float(_get_cfg("P5_KD", 0.3))
    w_goal = float(_get_cfg("P5_W_GOAL", 1.5))
    goal_speed_sec = float(_get_cfg("P5_GOAL_SPEED", 80.0))  # px/sec

    krep = float(_get_cfg("P5_KREP_PED", 600.0))
    rep_gamma = float(_get_cfg("P5_REP_GAMMA", 2.5))
    rep_max = float(_get_cfg("P5_REP_MAX", 5000.0))

    # ✅ NEW: goal stopping tolerance
    goal_tol = float(_get_cfg("P5_GOAL_TOL_PIX", 20.0))

    dt = float(_get_cfg("P5_DT", stride_used / max(fps, 1e-6)))  # seconds

    # flow conversion factor: (px/frame) * (frames/sec) = px/sec
    frames_per_second_used = max(fps / max(stride_used, 1), 1e-6)
    flow_to_sec = frames_per_second_used

    det = PedestrianDetectorMOG2(
        history=int(_get_cfg("P5_DET_HISTORY", 300)),
        var_threshold=float(_get_cfg("P5_DET_VAR_THR", 25.0)),
        detect_shadows=bool(_get_cfg("P5_DET_SHADOWS", False)),
        min_area=int(_get_cfg("P5_DET_MIN_AREA", 80)),
        morph_k=int(_get_cfg("P5_DET_MORPH_K", 5)),
    )
    tracker = SimpleNNTracker(
        dist_gate=float(_get_cfg("P5_TRACK_DIST_GATE", 35.0)),
        max_missed=int(_get_cfg("P5_TRACK_MAX_MISSED", 8)),
    )

    first = cv2.imread(frame_paths[0])
    H, W = first.shape[:2]
    out_mp4 = os.path.join(out_dir, f"{mission.name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, frames_per_second_used, (W, H))

    flows_dir = os.path.join(out_dir, "flows_cache")
    os.makedirs(flows_dir, exist_ok=True)

    x = np.array(mission.start, dtype=np.float32)
    v = np.zeros((2,), dtype=np.float32)  # px/sec

    traj_csv = os.path.join(out_dir, f"{mission.name}_traj.csv")
    mind_csv = os.path.join(out_dir, f"{mission.name}_mindist.csv")

    global_min_d = float("inf")
    global_min_t = 0.0
    global_min_step = 0

    # ✅ NEW: track whether we actually reached the goal
    reached_goal = False
    reached_step = None
    reached_time = None
    final_goal_dist = None

    def deriv(xv: np.ndarray, vv: np.ndarray, flow_field_pf: np.ndarray, ped_points) -> Tuple[np.ndarray, np.ndarray]:
        # sample flow in px/frame then convert to px/sec
        vf_pf = _sample_flow_bilinear(flow_field_pf, (float(xv[0]), float(xv[1])))
        vf_sec = vf_pf * float(flow_to_sec)

        # cap sampled flow speed in px/sec
        vf_sec = _sat_vec(vf_sec.astype(np.float32), vmax_sec).astype(np.float32)

        # goal bias in px/sec
        g = np.array(mission.goal, dtype=np.float32)
        to_goal = g - xv
        dist = float(np.linalg.norm(to_goal))
        goal_dir = (to_goal / dist).astype(np.float32) if dist > 1e-6 else np.zeros((2,), dtype=np.float32)
        v_goal = goal_dir * float(goal_speed_sec)

        v_des = vf_sec + float(w_goal) * v_goal

        # VERY IMPORTANT: cap desired velocity too, otherwise goal term can overpower everything
        v_des = _sat_vec(v_des.astype(np.float32), vmax_sec).astype(np.float32)

        a_rep = repulsion_force(
            robot_xy=(float(xv[0]), float(xv[1])),
            points=ped_points,
            rsafe=rsafe,
            krep=krep,
            gamma=rep_gamma,
            max_force=rep_max,
        ).astype(np.float32)

        # VT dynamics in px/sec
        a = kv * (v_des - vv) - kd * vv + a_rep
        dx = vv
        dv = a
        return dx, dv

    with open(traj_csv, "w", newline="") as ftraj, open(mind_csv, "w", newline="") as fmd:
        wtraj = csv.writer(ftraj)
        wmd = csv.writer(fmd)
        wtraj.writerow(["step", "t", "x", "y", "vx", "vy"])
        wmd.writerow(["step", "t", "min_dist"])

        for i in range(len(frame_paths) - 1):
            frame = cv2.imread(frame_paths[i])
            nxt = cv2.imread(frame_paths[i + 1])
            if frame is None or nxt is None:
                continue

            gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)

            flow_path = os.path.join(flows_dir, f"flow_{i:06d}.npy")
            if os.path.exists(flow_path):
                flow_pf = np.load(flow_path).astype(np.float32, copy=False)
            else:
                flow_pf = compute_dense_flow(gray0, gray1, method=flow_method).astype(np.float32)

                # Saturate in px/frame using vmax_sec converted to px/frame
                vmax_pf = float(vmax_sec) / float(flow_to_sec)
                mag = np.linalg.norm(flow_pf, axis=2)
                scale = np.ones_like(mag, dtype=np.float32)
                over = mag > vmax_pf
                scale[over] = (vmax_pf / np.maximum(mag[over], 1e-6)).astype(np.float32)
                flow_pf[..., 0] *= scale
                flow_pf[..., 1] *= scale

                np.save(flow_path, flow_pf)

            boxes = det.detect(frame)
            cents = boxes_to_centroids(boxes)
            tracks = tracker.update(cents)
            ped_points = tracks

            md = min_distance((float(x[0]), float(x[1])), ped_points)
            if md < global_min_d:
                global_min_d = md
                global_min_t = i * dt
                global_min_step = i
            wmd.writerow([i, i * dt, md])

            dx1, dv1 = deriv(x, v, flow_pf, ped_points)
            x_mid = x + 0.5 * dt * dx1
            v_mid = v + 0.5 * dt * dv1
            dx2, dv2 = deriv(x_mid, v_mid, flow_pf, ped_points)

            x = x + dt * dx2
            v = v + dt * dv2

            # Hard speed cap for stability/safety (px/sec)
            v = _sat_vec(v.astype(np.float32), vmax_sec).astype(np.float32)

            x[0] = float(np.clip(x[0], robot_r, W - 1 - robot_r))
            x[1] = float(np.clip(x[1], robot_r, H - 1 - robot_r))

            wtraj.writerow([i, i * dt, float(x[0]), float(x[1]), float(v[0]), float(v[1])])

            # ✅ NEW: goal reached check (after state update + clamp, before writing)
            gxy = np.array(mission.goal, dtype=np.float32)
            d_goal = float(np.linalg.norm(gxy - x))
            final_goal_dist = d_goal
            if d_goal <= goal_tol:
                reached_goal = True
                reached_step = i
                reached_time = i * dt

            vis = frame.copy()

            for (bx, by, bw, bh) in boxes:
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            for tr in tracks:
                cx, cy = int(tr["x"]), int(tr["y"])
                cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(vis, f"id={tr['id']}", (cx + 4, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            sx, sy = int(mission.start[0]), int(mission.start[1])
            gx, gy = int(mission.goal[0]), int(mission.goal[1])
            cv2.circle(vis, (sx, sy), 6, (255, 255, 0), 2)
            cv2.circle(vis, (gx, gy), 6, (0, 255, 255), 2)

            cv2.circle(vis, (int(x[0]), int(x[1])), int(robot_r), (255, 255, 255), 2)
            cv2.circle(vis, (int(x[0]), int(x[1])), 2, (255, 255, 255), -1)

            # flow arrows (still in px/frame for visualization)
            rx, ry = int(x[0]), int(x[1])
            step = 18
            for yy in range(max(0, ry - 36), min(H, ry + 37), step):
                for xx in range(max(0, rx - 36), min(W, rx + 37), step):
                    fxy = flow_pf[yy, xx].astype(np.float32)
                    ex = int(xx + 2.0 * fxy[0])
                    ey = int(yy + 2.0 * fxy[1])
                    cv2.arrowedLine(vis, (xx, yy), (ex, ey), (200, 200, 200), 1, tipLength=0.3)

            # ✅ include goal dist in overlay
            cv2.putText(
                vis,
                f"{mission.name}  minD={global_min_d:.1f}px  goalD={d_goal:.1f}px  vmax={vmax_sec:.0f}px/s",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            vw.write(vis)

            # ✅ NEW: stop after writing the frame where goal is reached
            if reached_goal:
                break

    vw.release()

    summary_txt = os.path.join(out_dir, f"{mission.name}_summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"mission={mission.name}\n")
        f.write(f"start={mission.start}\n")
        f.write(f"goal={mission.goal}\n")
        f.write(f"dt={dt}\n")
        f.write(f"stride_used={stride_used}\n")
        f.write(f"fps={fps}\n")
        f.write(f"frames_per_second_used={frames_per_second_used}\n")
        f.write(f"flow_to_sec={flow_to_sec}\n")
        f.write(f"rsafe={rsafe}\n")
        f.write(f"robot_radius={robot_r}\n")
        f.write(f"vmax_sec={vmax_sec}\n")
        f.write(f"goal_tol_pix={goal_tol}\n")
        f.write(f"reached_goal={reached_goal}\n")
        f.write(f"reached_step={reached_step}\n")
        f.write(f"reached_time={reached_time}\n")
        f.write(f"final_goal_dist={final_goal_dist}\n")
        f.write(f"min_distance={global_min_d:.4f}\n")
        f.write(f"min_distance_time={global_min_t:.4f}\n")
        f.write(f"min_distance_step={global_min_step}\n")
        f.write(f"video={out_mp4}\n")
        f.write(f"traj_csv={traj_csv}\n")
        f.write(f"mindist_csv={mind_csv}\n")

    return {
        "video": out_mp4,
        "traj_csv": traj_csv,
        "mindist_csv": mind_csv,
        "min_distance": global_min_d,
        "summary": summary_txt,
        "reached_goal": reached_goal,
        "reached_step": reached_step,
        "reached_time": reached_time,
        "final_goal_dist": final_goal_dist,
        "goal_tol_pix": goal_tol,
    }


def run(results_dir: str) -> Dict:
    run_max_seconds = _as_float_or_none(_get_cfg("P5_RUN_MAX_SECONDS", None))
    run_stride = int(_get_cfg("P5_RUN_FRAME_STRIDE", 1))
    run_resize_width = _as_int_or_none(_get_cfg("P5_RUN_RESIZE_WIDTH", 640))
    keep_aspect = bool(_get_cfg("P5_KEEP_ASPECT", True))

    meta = extract_frames(
        results_dir=results_dir,
        max_seconds=run_max_seconds,
        stride=run_stride,
        resize_width=run_resize_width,
        keep_aspect=keep_aspect,
        overwrite=True,
    )

    frame_paths = meta["frame_paths"]
    fps = float(meta.get("fps", 30.0))
    out_hw = meta["out_hw"]
    H, W = int(out_hw[0]), int(out_hw[1])

    m1, m2 = make_two_missions(W, H)

    out_dir = os.path.join(results_dir, "phase5_runs")
    os.makedirs(out_dir, exist_ok=True)

    r1 = run_mission_on_frames(results_dir, m1, frame_paths, fps, out_dir, stride_used=run_stride)
    r2 = run_mission_on_frames(results_dir, m2, frame_paths, fps, out_dir, stride_used=run_stride)

    return {"mission1": r1, "mission2": r2, "out_dir": out_dir}


__all__ = ["run", "run_mission_on_frames", "make_two_missions", "Mission"]
