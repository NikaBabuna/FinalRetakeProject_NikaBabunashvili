# run_phase5_missions.py
import os
import re

from src.phase5.run_missions import run as run_phase5_missions


def _next_results_dir(base: str = "output", prefix: str = "results_") -> str:
    """
    Finds next available output/results_XXXX directory (zero-padded to 4).
    Example: output/results_0058 -> next is output/results_0059
    """
    os.makedirs(base, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    max_n = -1
    for name in os.listdir(base):
        m = pat.match(name)
        if not m:
            continue
        try:
            n = int(m.group(1))
            max_n = max(max_n, n)
        except ValueError:
            pass

    next_n = max_n + 1
    results_dir = os.path.join(base, f"{prefix}{next_n:04d}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def main():
    results_dir = _next_results_dir()

    print("===============================================")
    print("PHASE 5 MISSIONS RUNNER")
    print(f"Results folder: {os.path.abspath(results_dir)}")
    print("===============================================")

    out = run_phase5_missions(results_dir)

    print("\nSaved outputs in:", out["out_dir"])
    print("Mission 1 video:", out["mission1"]["video"])
    print("Mission 2 video:", out["mission2"]["video"])
    print("Mission 1 min dist:", out["mission1"]["min_distance"])
    print("Mission 2 min dist:", out["mission2"]["min_distance"])
    print("\nSummaries:")
    print(" -", out["mission1"]["summary"])
    print(" -", out["mission2"]["summary"])


if __name__ == "__main__":
    main()
