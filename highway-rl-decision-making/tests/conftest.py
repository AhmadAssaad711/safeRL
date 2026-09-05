"""Shared pytest bootstrap for the organized safeRL source tree."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = PROJECT_ROOT / "laneless highway env"

for _path in (PROJECT_ROOT, ENV_ROOT):
    _path_string = str(_path)
    if _path_string not in sys.path:
        sys.path.insert(0, _path_string)
