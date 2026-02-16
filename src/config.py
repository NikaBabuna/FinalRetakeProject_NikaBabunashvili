import numpy as np
import random

# ==================================================
# MAP SETTINGS (PHASE 3)
# ==================================================

MAP_FILENAME = "map.png"

# Will be filled after clicking
A_PIX = None
B_PIX = None

# ----------------------------
# PATH / SPLINE SETTINGS
# ----------------------------
PATH_DOWNSAMPLE_STEP = 6        # pixels between kept points
PATH_SMOOTH_WINDOW = 9          # moving-average window (odd number)
CLOSEST_S_SAMPLES = 1200        # samples used for closest_s()


DT = 0.05
VMAX = 2.0

KP = 1.0
KD = 0.5
KREP = 2.0

RSAFE = 1.0
ROBOT_RADIUS = 10

# ==================================================
# PATH CONSTRAINT SETTINGS (STEP 3.5)
# ==================================================


PATH_WIDTH_PIX = 27.5

PATH_WIDTH = PATH_WIDTH_PIX   # pixels or world units (tune later)
  # corridor width in pixels (adjust later if needed)

LOOKAHEAD = 25.0        # pixels ahead on path
PROGRESS_GAIN = 0.5     # how fast progress moves



N_ROBOTS = 10

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

KP_PATH = 2.0
KD_PATH = 1.0
