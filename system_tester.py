import os
import sys
import importlib.util
from typing import List, Tuple

from src.utils.io_utils import ensure_output_root, create_new_results_dir


def discover_test_files(test_root: str) -> List[str]:
    """
    Finds any python file under test_root starting with 'test' and ending with '.py'
    """
    found = []
    for root, _, files in os.walk(test_root):
        for f in files:
            if f.startswith("test") and f.endswith(".py"):
                found.append(os.path.join(root, f))
    found.sort()
    return found


def import_module_from_path(file_path: str):
    """
    Dynamically import a .py file as a module.
    """
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_one_test(module, results_dir: str) -> Tuple[str, bool, str]:
    """
    Runs a test module.
    Expected:
      - TEST_NAME (optional)
      - run(results_dir) -> bool (recommended)
    Returns: (name, passed, message)
    """
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

    # fallback: if no run() exists, attempt to execute module-level code
    # (works for your current tests that run immediately)
    try:
        # importing already executed top-level code;
        # if it didn't crash, consider it pass
        return name, True, "No run() found; passed by import execution"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return name, False, f"Exception during import execution: {e}"


def main():
    # Ensure project root is on sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    test_root = os.path.join(project_root, "src", "test")
    if not os.path.isdir(test_root):
        print(f"ERROR: test folder not found at {test_root}")
        sys.exit(1)

    output_root = ensure_output_root("output")
    session_dir = create_new_results_dir(output_root, prefix="results_")
    tests_output_dir = os.path.join(session_dir, "tests")
    os.makedirs(tests_output_dir, exist_ok=True)

    print("===============================================")
    print("SYSTEM TESTER")
    print(f"Discovered tests under: {test_root}")
    print(f"Results folder: {session_dir}")
    print("===============================================\n")

    test_files = discover_test_files(test_root)
    if not test_files:
        print("No test files found (expected files starting with 'test' under src/test/).")
        sys.exit(1)

    results = []

    for path in test_files:
        rel = os.path.relpath(path, project_root)
        print("------------------------------------------------")
        print(f"RUNNING: {rel}")
        print("------------------------------------------------")

        try:
            module = import_module_from_path(path)
            name, passed, msg = run_one_test(module, tests_output_dir)
        except Exception as e:
            name, passed, msg = rel, False, f"Import/Run error: {e}"

        status = "PASS" if passed else "FAIL"
        print(f"Test: {name} -> {status} ({msg})\n")
        results.append((name, passed))

    # summary
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    failed_count = total - passed_count

    print("===============================================")
    print("SUMMARY")
    print(f"Total: {total} | Passed: {passed_count} | Failed: {failed_count}")
    failed = [name for name, passed in results if not passed]
    if failed:
        print("Failed tests:")
        for name in failed:
            print(f" - {name}")

    print(f"Saved test artifacts in: {tests_output_dir}")
    print("===============================================")

    # exit code for CI-style usage
    sys.exit(0 if failed_count == 0 else 2)


if __name__ == "__main__":
    main()
