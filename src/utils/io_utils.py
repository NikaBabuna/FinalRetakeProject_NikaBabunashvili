from pathlib import Path
import os
import re

def get_project_root() -> str:
    """
    Returns the repository root folder (the one containing both 'src' and 'data').
    Works no matter what your working directory is.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return str(parent)
    # fallback (shouldn't happen, but prevents crashes)
    return str(here.parents[2])


def ensure_output_root(folder_name: str = "output") -> str:
    project_root = get_project_root()
    out_root = os.path.join(project_root, folder_name)
    os.makedirs(out_root, exist_ok=True)
    return out_root


def create_new_results_dir(output_root: str, prefix: str = "results_") -> str:
    """
    Creates a new folder output_root/results_XXXX where XXXX increments automatically.
    Returns the absolute path to the new folder.
    """
    existing = os.listdir(output_root)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    max_id = 0
    for name in existing:
        m = pattern.match(name)
        if m:
            max_id = max(max_id, int(m.group(1)))

    new_id = max_id + 1
    folder_name = f"{prefix}{new_id:04d}"

    results_dir = os.path.join(output_root, folder_name)
    os.makedirs(results_dir, exist_ok=False)
    return results_dir
