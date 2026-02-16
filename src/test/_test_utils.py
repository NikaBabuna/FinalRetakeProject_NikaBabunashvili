from dataclasses import dataclass, field
from typing import Dict, Any, List
import os
import numpy as np


# ============================================================
# TEST CONTEXT (logging + paths)
# ============================================================

@dataclass
class TestContext:
    results_dir: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)

    def path(self, filename: str) -> str:
        os.makedirs(self.results_dir, exist_ok=True)
        return os.path.join(self.results_dir, filename)

    def info(self, msg: str):
        print(f"  [INFO] {msg}")
        self.messages.append(f"INFO: {msg}")

    def pass_(self, msg: str):
        print(f"  [PASS] {msg}")
        self.messages.append(f"PASS: {msg}")

    def fail(self, msg: str):
        print(f"  [FAIL] {msg}")
        self.messages.append(f"FAIL: {msg}")


# ============================================================
# SIMPLE ASSERT HELPERS
# ============================================================

def log(code: str, msg: str):
    print(f"  [INFO] {code} - {msg}")


def assert_true(condition: bool, msg: str):
    if condition:
        print(f"  [PASS] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        raise AssertionError(msg)


# ============================================================
# LOAD CENTERLINE FROM PIPELINE
# ============================================================

def load_centerline_from_previous_step():
    """
    Rebuilds centerline using the actual pipeline:
    map -> mask -> centerline
    """

    from src.path.path_extraction import extract_path_mask
    from src.path.centerline import extract_centerline_points
    from src.map_tools.map_click_ab import ensure_AB_points

    # build mask
    mask = extract_path_mask()

    # get A/B from config or UI
    A, B = ensure_AB_points()

    # build centerline
    centerline = extract_centerline_points(mask, A, B)

    if centerline is None or len(centerline) < 10:
        raise RuntimeError("Centerline extraction failed")

    centerline = np.asarray(centerline, dtype=float)
    return centerline
