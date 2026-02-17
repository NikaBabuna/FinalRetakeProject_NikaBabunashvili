"""
src/config.py

Single source of truth for all tunable parameters.

Conventions
-----------
- Distances are in PIXELS.
- Time is in seconds.
- Velocity is pixels/second.

Notes
-----
Phase 5 optical flow from OpenCV is naturally in px/frame, but the mission
runner should convert it to px/sec using dt = stride/fps. This config assumes
Phase 5 "P5_VMAX" and "P5_GOAL_SPEED" are px/sec.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)


# ============================================================
# INPUTS (MAP)
# ============================================================

MAP_FILENAME: str = "map.png"


# ============================================================
# PHASE 5 (VIDEO / FRAMES / FLOW)
# ============================================================

P5_VIDEO_FILENAME: str = "pedestrians.mp4"

# Resize controls
P5_RESIZE_WIDTH: int | None = 640
P5_KEEP_ASPECT: bool = True
P5_FRAME_EXT: str = "png"

# FAST TEST SETTINGS (used by run_phase5_tester.py)
P5_TEST_MAX_SECONDS: float | None = 2.0
P5_TEST_FRAME_STRIDE: int = 3

# REAL RUN SETTINGS (used by mission runner)
P5_RUN_MAX_SECONDS: float | None = None   # None = full video
P5_RUN_FRAME_STRIDE: int = 1
P5_RUN_RESIZE_WIDTH: int | None = 640

# Optical flow method
P5_FLOW_METHOD: str = "farneback"  # "farneback" or "dis" (if implemented)

# Saturation cap for the velocity field (px/sec)
P5_VMAX: float = 110.0   # slightly calmer than 120; try 80..160


# ============================================================
# PHASE 5 (DETECTION / TRACKING)
# ============================================================

# Background subtractor (MOG2) tuning
P5_MOG2_HISTORY: int = 300
P5_MOG2_VAR_THRESHOLD: int = 25
P5_MOG2_DETECT_SHADOWS: bool = False

# Detector post-processing
P5_DET_WARMUP_FRAMES: int = 45
P5_DET_MIN_AREA: int = 80
P5_DET_MAX_AREA: int = 20000
P5_DET_THRESH: int = 200
P5_DET_CLOSE_ITERS: int = 2
P5_DET_OPEN_ITERS: int = 1
P5_DET_MORPH_K: int = 5

# Tracking (names match your run_missions.py)
P5_TRACK_DIST_GATE: float = 60.0
P5_TRACK_MAX_MISSED: int = 10


# ============================================================
# PHASE 5 (REPULSION + VT DYNAMICS)
# ============================================================

P5_ROBOT_RADIUS_PIX = 10.0

# Rsafe should be > 2*radius, but 60 is huge for centroid-distance in a dense crowd.
P5_RSAFE_PIX = 35.0

# Make repulsion ramp HARD only when close:
P5_KREP_PED = 1100.0
P5_REP_GAMMA = 4.0      # steeper => less “carrying”, more emergency shove
P5_REP_MAX = 15000.0    # raise the cap so it can actually prevent collisions

# Make “get to goal” still happen, but don’t bulldoze:
P5_GOAL_SPEED = 60.0
P5_W_GOAL = 1.2

# Calm the field-following a bit:
P5_VMAX = 95.0

# Add damping so it doesn’t get yanked around:
P5_KV = 6.0
P5_KD = 1.0



# ============================================================
# BACKWARDS-COMPAT ALIASES (older tests / modules)
# ============================================================
# Some earlier tests referenced these names. Keep them wired to the new knobs.

P5_MAX_SECONDS: float | None = P5_TEST_MAX_SECONDS
P5_FRAME_STRIDE: int = P5_TEST_FRAME_STRIDE

P5_DET_HISTORY: int = P5_MOG2_HISTORY
P5_DET_VAR_THR: float = float(P5_MOG2_VAR_THRESHOLD)
P5_DET_SHADOWS: bool = P5_MOG2_DETECT_SHADOWS


# ============================================================
# MAP CLICK A/B (PHASE 1 pipeline dependency)
# ============================================================

A_PIX: Optional[Tuple[int, int]] = (402, 548)
B_PIX: Optional[Tuple[int, int]] = (930, 231)

# Set to False once you’ve selected A/B and saved them.
FORCE_AB_RESELECT: bool = True


# ============================================================
# CORE DYNAMICS (PHASE 2/3/4 ENGINE)
# ============================================================

DT: float = 0.05

# NOTE: Phase 2 engine uses VMAX; Phase 5 uses P5_VMAX.
VMAX: float = 2.0

KP: float = 1.2
KD: float = 0.6

KREP: float = 25.0
RSAFE: float = 18.0

ROBOT_RADIUS: float = 6.0


# ============================================================
# BORDER REPULSION (PHASE 4)
# ============================================================

K_BORDER: float = 40.0
BORDER_MARGIN: float = 6.0
BORDER_FORCE_MAX: float = 200.0


# ============================================================
# PATH / CONTROLLER (PHASE 3)
# ============================================================

PATH_WIDTH_PIX: float = 60.0
PATH_WIDTH: float = PATH_WIDTH_PIX

LOOKAHEAD: float = 25.0

KP_PATH: float = 1.2
KD_PATH: float = 0.6

CLOSEST_S_SAMPLES: int = 1200


# ============================================================
# SWARM (PHASE 4) — dynamic spawn sizing
# ============================================================

# "Requested" robots
N_ROBOTS: int = 10

# Keep robot size physically meaningful (do NOT shrink too aggressively)
ROBOT_RADIUS: float = 6.0

# src/config.py
#
# Paste these additions into your existing config.py (keep your current values).
# Best place: inside the "SWARM (PHASE 4)" section, right after your SPAWN_* constants.

# ============================================================
# SWARM (PHASE 4) — dynamic spawn sizing (so big N still fits)
# ============================================================

# "Expected to work reliably" size. Above this, we automatically widen spawn bands.
N_SAFE: int = 40  # (you already have this, keep it)

def swarm_spawn_band_px(spline_length_px: float) -> float:
    """
    How far along the spline we allow spawning near EACH endpoint.
    As N grows, widen the band so robots don't overlap at t=0.

    - For N <= N_SAFE: about SPAWN_S_BAND_PIX (or 12% of length fallback)
    - For N >  N_SAFE: grows gently, capped at 25% of spline length
    """
    L = float(spline_length_px)
    n = float(max(1, int(getattr(__import__(__name__), "N_ROBOTS", 10))))
    n_safe = float(max(1, int(getattr(__import__(__name__), "N_SAFE", 40))))

    base = float(getattr(__import__(__name__), "SPAWN_S_BAND_PIX", 60.0))
    # gentle growth above N_SAFE
    scale = max(1.0, n / n_safe)

    band = base * scale
    # also ensure it's not *too* small on long paths
    band = max(band, 0.12 * L)
    # cap so we don't spawn across half the path
    band = min(band, 0.25 * L)
    return float(band)

def swarm_spawn_min_sep() -> float:
    """
    Minimum initial separation used by spawner.
    Keep tied to robot radius (stable + avoids instant collisions).
    """
    r = float(getattr(__import__(__name__), "ROBOT_RADIUS", 6.0))
    return float(getattr(__import__(__name__), "SPAWN_MIN_SEP", 2.0 * r + 0.5))


# Spawn settings (base)
SPAWN_S_BAND_PIX: float = 60.0       # base along-spline band near each endpoint
SPAWN_SPACING: float = 30.0          # base desired spacing
SPAWN_LANE_OFFSET: float = 12.0
SPAWN_JITTER: float = 3.0

# --- Derived knobs based on N ---
def swarm_spawn_band_px(spline_length_px: float) -> float:
    """
    Increase spawn band length as N grows so robots have enough space to spawn.
    """
    n = float(max(1, N_ROBOTS))
    # scale factor: 1.0 at N<=N_SAFE, grows gently above that
    scale = max(1.0, n / float(N_SAFE))
    band = SPAWN_S_BAND_PIX * scale
    # also cap by a fraction of total path length so it doesn't get silly
    return float(min(band, 0.25 * spline_length_px))

def swarm_spawn_min_sep() -> float:
    """
    Minimum allowed separation for spawning (prevents overlap).
    Tied to robot size.
    """
    return float(2.0 * ROBOT_RADIUS + 1.0)

def swarm_spawn_spacing() -> float:
    """
    Preferred spacing for spawner attempts.
    Keep it at least min_sep and allow slight shrink if needed.
    """
    base = float(SPAWN_SPACING)
    return float(max(base, swarm_spawn_min_sep()))



# ============================================================
# SAFETY (PHASE 4+)
# ============================================================

ACCIDENT_DIST: float = 2.0 * ROBOT_RADIUS


# ============================================================
# SANITY CHECKS
# ============================================================

def _sanity_checks() -> None:
    assert DT > 0, "DT must be > 0"
    assert VMAX > 0, "VMAX must be > 0"
    assert PATH_WIDTH_PIX > 0, "PATH_WIDTH_PIX must be > 0"
    assert ROBOT_RADIUS >= 0, "ROBOT_RADIUS must be >= 0"
    assert RSAFE >= 0, "RSAFE must be >= 0"
    assert KREP >= 0, "KREP must be >= 0"

    assert P5_VMAX > 0, "P5_VMAX must be > 0"
    assert P5_RSAFE_PIX >= 0, "P5_RSAFE_PIX must be >= 0"
    assert P5_ROBOT_RADIUS_PIX >= 0, "P5_ROBOT_RADIUS_PIX must be >= 0"
    assert P5_KREP_PED >= 0, "P5_KREP_PED must be >= 0"
    assert P5_REP_GAMMA > 0, "P5_REP_GAMMA must be > 0"
    assert P5_REP_MAX > 0, "P5_REP_MAX must be > 0"

_sanity_checks()
