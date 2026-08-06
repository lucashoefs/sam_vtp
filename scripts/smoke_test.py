#!/usr/bin/env python3
"""Run the smoke test without installing the package (adds src/ to sys.path)."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam_vtp_core.smoke import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
