import os
import sys
import argparse
import importlib.util
from typing import List, Tuple

from src.utils.io_utils import ensure_output_root, create_new_results_dir
from src import config


def discover_phase5_tests(test_root: str) -> List[str]:
    """
    Find Phase 5 test files.
    Default: test_phase5*.py (recommended naming convention)
    """
    found = []
    for root, _, files in os.walk(test_root):
        for f in files:
            if f.startswith("test_phase5") and f.endswith(".py"):
                found.append(os.path.join(root, f))
    found.sort()
    return found


def import_module_from_path(file_path: str):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_one_test(module, results_dir: str) -> Tuple[str, bool, str]:
    name = getattr(module, "TEST_NAME", module.__name__)

    if hasattr(module, "run") and callable(module.run):
        try:
            passed = bool(module.run(results_dir))
            return name, passed, "Used run(results_dir)"
        except Exception as e:
            import traceback
            print("\n--- FULL TRACEBACK ---")
            traceback.print_exc()
            print("--- END TRACEBACK ---\n")
            return name, False, f"Exception: {e}"

    return name, True, "No run() found; passed by import execution"


def apply_fast_overrides():
    """
    Keep Phase 5 unit tests quick even if config defaults are 'demo sized'.
    These are temporary monkeypatches for this process only.
    """
    setattr(config, "P5_MAX_SECONDS", 2.0)
    setattr(config, "P5_FRAME_STRIDE", 3)
    setattr(config, "P5_RESIZE_WIDTH", 640)
    setattr(config, "P5_KEEP_ASPECT", True)
    # Optional: choose method
    if not hasattr(config, "P5_FLOW_METHOD"):
        setattr(config, "P5_FLOW_METHOD", "farneback")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5 tests only.")
    parser.add_argument("--list", action="store_true", help="List discovered Phase 5 tests and exit.")
    parser.add_argument("--pattern", type=str, default="", help="Run only tests whose filename contains this substring.")
    parser.add_argument("--fast", action="store_true", help="Force fast config for tests (2s, stride 3).")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if args.fast:
        apply_fast_overrides()
        print("[INFO] FAST mode ON (P5_MAX_SECONDS=2.0, P5_FRAME_STRIDE=3, width=640)")

    test_root = os.path.join(project_root, "src", "test")
    if not os.path.isdir(test_root):
        print(f"ERROR: test folder not found at {test_root}")
        sys.exit(1)

    tests = discover_phase5_tests(test_root)

    if args.pattern:
        pat = args.pattern.lower().strip()
        tests = [p for p in tests if pat in os.path.basename(p).lower()]

    if not tests:
        print("No Phase 5 tests found matching your filters.")
        sys.exit(1)

    if args.list:
        print("Discovered Phase 5 tests:")
        for p in tests:
            print(" -", os.path.relpath(p, project_root))
        sys.exit(0)

    out_root = ensure_output_root("output")
    session_dir = create_new_results_dir(out_root, prefix="results_")
    tests_dir = os.path.join(session_dir, "tests_phase5")
    os.makedirs(tests_dir, exist_ok=True)

    print("===============================================")
    print("PHASE 5 TESTER")
    print(f"Discovered: {len(tests)} tests")
    print(f"Results folder: {session_dir}")
    print("===============================================\n")

    results = []
    for path in tests:
        rel = os.path.relpath(path, project_root)
        print("------------------------------------------------")
        print(f"RUNNING: {rel}")
        print("------------------------------------------------")
        try:
            mod = import_module_from_path(path)
            name, passed, msg = run_one_test(mod, tests_dir)
        except Exception as e:
            name, passed, msg = rel, False, f"Import/Run error: {e}"

        status = "PASS" if passed else "FAIL"
        print(f"Test: {name} -> {status} ({msg})\n")
        results.append((name, passed))

    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    failed_count = total - passed_count

    print("===============================================")
    print("SUMMARY (PHASE 5)")
    print(f"Total: {total} | Passed: {passed_count} | Failed: {failed_count}")

    failed = [name for name, ok in results if not ok]
    if failed:
        print("Failed tests:")
        for name in failed:
            print(f" - {name}")

    print(f"Saved test artifacts in: {tests_dir}")
    print("===============================================")

    sys.exit(0 if failed_count == 0 else 2)


if __name__ == "__main__":
    main()
