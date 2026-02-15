import numpy as np
import random

DT = 0.05
VMAX = 2.0

KP = 1.0
KD = 0.5
KREP = 2.0

RSAFE = 1.0
ROBOT_RADIUS = 0.2

N_ROBOTS = 10

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
