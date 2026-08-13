from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "scripts" / "data"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
