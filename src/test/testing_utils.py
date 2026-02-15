from dataclasses import dataclass, field
from typing import Dict, Any, List
import os


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
