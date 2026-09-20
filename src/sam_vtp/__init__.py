"""SAM-VTP: temporally consistent, continuous traversability from a video stream.

    from sam_vtp import build_model
    model = build_model("sam_vtp.pt", device="cuda")
    out = model.forward_frame(rgb, is_init_frame=True, depth=depth)
    traversability = out["logits"].sigmoid()

The network is assembled from the pinned SAM 2 submodule (`third_party/sam2`)
plus the SAM-VTP parts in `sam_vtp.modeling`. The label pipeline lives in
`sam_vtp.labels` and is imported separately, since it needs OpenCV and is only
used to build training data.
"""

from .build import build_config, build_model, describe, tune_for_inference
from .checkpoint import load_checkpoint_state, read_checkpoint, resolve_path
from .fullgraph import SAMVTPFullGraphReplayRunner
from .inference import (
    compatible_input_hw,
    depth_from_uint16_mm,
    infer_sequence,
    list_depths,
    list_images,
    read_depth_metres,
)
from .modeling.geometry import TARTANGROUND, CameraIntrinsics, geometry_channels
from .modeling.sam_vtp import SAMVTP
from .stream import SAMVTPStreamProcessor, StreamStepResult, run_stream_loop

__version__ = "0.2.0"

__all__ = [
    "SAMVTP",
    "CameraIntrinsics",
    "TARTANGROUND",
    "geometry_channels",
    "depth_from_uint16_mm",
    "build_config",
    "build_model",
    "describe",
    "tune_for_inference",
    "read_checkpoint",
    "load_checkpoint_state",
    "resolve_path",
    "infer_sequence",
    "list_images",
    "list_depths",
    "read_depth_metres",
    "compatible_input_hw",
    "SAMVTPStreamProcessor",
    "StreamStepResult",
    "run_stream_loop",
    "SAMVTPFullGraphReplayRunner",
    "__version__",
]
