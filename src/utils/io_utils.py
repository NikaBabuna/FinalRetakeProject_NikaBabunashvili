import os
import re


def get_project_root() -> str:
    """
    Assumes this file is in src/.
    """
    current_file = os.path.abspath(__file__)
    src_folder = os.path.dirname(current_file)
    project_root = os.path.dirname(src_folder)
    return project_root


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
