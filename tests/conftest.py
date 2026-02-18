from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
