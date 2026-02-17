"""
src/config.py

Single source of truth for all tunable parameters.

Conventions
-----------
- Distances are in PIXELS (because your map + spline coordinates are pixels).
- Time is in seconds.
- Velocity is pixels/second.
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

# Map image filename inside /data/
MAP_FILENAME: str = "map.png"

# Start/End points (x, y) in pixel coordinates.
# If left as None, your click tool will ask once and then persist to data/ab_points.json.
A_PIX: Optional[Tuple[int, int]] = (402, 548)
B_PIX: Optional[Tuple[int, int]] = (930, 231)

# If True: force the map click UI to reselect A/B on next run.
# After selecting, set it back to False.
FORCE_AB_RESELECT: bool = True


# ============================================================
# CORE DYNAMICS (PHASE 2 ENGINE)
# ============================================================

# Integrator step size
DT: float = 0.05

# Speed cap (pixels/second)
VMAX: float = 2.0

# Target attraction (spring-like pull)
KP: float = 1.2

# Damping (velocity drag)
KD: float = 0.6

# Inter-robot repulsion strength and safety distance
KREP: float = 25.0
RSAFE: float = 18.0

# Robot geometry (pixels)
ROBOT_RADIUS: float = 6.0

# ============================================================
# BORDER REPULSION (PHASE 4)
# ============================================================

# Border “guard rail” push strength
K_BORDER: float = 40.0

# Start pushing this many pixels BEFORE the border (helps avoid oscillation)
BORDER_MARGIN: float = 6.0

# Clamp border force magnitude (prevents crazy spikes)
BORDER_FORCE_MAX: float = 200.0


# ============================================================
# PATH / CONTROLLER (PHASE 3)
# ============================================================

# Corridor width around centerline (pixels)
# Keep both names because some modules/tests reference one or the other.
PATH_WIDTH_PIX: float = 60.0
PATH_WIDTH: float = PATH_WIDTH_PIX

# Lookahead distance along spline (arc-length units = pixels here)
LOOKAHEAD: float = 25.0

# If you ever want separate controller gains (optional)
KP_PATH: float = 1.2
KD_PATH: float = 0.6

# closest_s brute force sampling (higher = more accurate but slower)
CLOSEST_S_SAMPLES: int = 1200


# ============================================================
# SWARM (PHASE 4)
# ============================================================

# Default number of robots in swarm tests / demos
N_ROBOTS: int = 10

# After tuning, set this to the max safe N you found
N_SAFE: int = 40


# Spawn behavior near endpoints (in pixels along spline arc length)
# "Near" means within these ranges from A-side or B-side.
SPAWN_S_BAND_PIX: float = 60.0     # along the spline
SPAWN_T_JITTER_PIX: float = 12.0   # perpendicular jitter (keeps them from stacking)

# Phase 4 spawn tuning
SPAWN_SPACING = 30.0         # try ~ (2*ROBOT_RADIUS)*1.5
SPAWN_LANE_OFFSET = 12.0     # try ~ ROBOT_RADIUS*1.2
SPAWN_JITTER = 3.0
SPAWN_MAX_TRIES = 2000

SPAWN_MIN_SEP = 2.0 * ROBOT_RADIUS + 0.5   # default



# If you want to bias spawn away from borders, keep jitter smaller than PATH_WIDTH/2.


# ============================================================
# SAFETY / DETECTION (PHASE 4+)
# ============================================================

# Accident definition (distance threshold between robots)
# Common: 2 * ROBOT_RADIUS, but configurable.
ACCIDENT_DIST: float = 2.0 * ROBOT_RADIUS


# ============================================================
# SMALL SANITY CHECKS (OPTIONAL)
# ============================================================

def _sanity_checks() -> None:
    assert DT > 0, "DT must be > 0"
    assert VMAX > 0, "VMAX must be > 0"
    assert PATH_WIDTH_PIX > 0, "PATH_WIDTH_PIX must be > 0"
    assert ROBOT_RADIUS >= 0, "ROBOT_RADIUS must be >= 0"
    assert RSAFE >= 0, "RSAFE must be >= 0"
    assert KREP >= 0, "KREP must be >= 0"

_sanity_checks()
