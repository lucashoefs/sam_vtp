"""Continuous traversability supervision from simulator depth and semantics.

`taxonomy` maps a semantic class name to an absolute, platform-grounded score;
`generate` combines that prior with depth-derived slope, roughness and height
into the dense target `y in [0, 1]` the model is trained on.

    python -m sam_vtp_core.labels.generate --root /data/tartanground

These modules need `opencv-python`, which the base install does not pull in:

    pip install "sam-vtp[labels]"
"""

from . import taxonomy
from .taxonomy import build_lut, lookup

__all__ = ["taxonomy", "build_lut", "lookup"]
