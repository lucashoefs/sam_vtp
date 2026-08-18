"""SAM-VTP: temporally consistent, continuous traversability from a video stream.

    from sam_vtp_core import build_model
    model = build_model("sam_vtp.pt", device="cuda")
    out = model.forward_frame(rgb, is_init_frame=True, depth=depth)
    traversability = out["logits"].sigmoid()

The network is defined end to end in `architecture.py` and depends only on
PyTorch. The label pipeline lives in `sam_vtp_core.labels` and is imported
separately, since it needs OpenCV and is only used to build training data.
"""

from .architecture import (
    CameraIntrinsics,
    SAMVTP,
    SAMVTPConfig,
    TARTANGROUND,
    geometry_channels,
)
from .factory import (
    build_model,
    describe,
    load_checkpoint_state,
    read_checkpoint,
    resolve_path,
    tune_for_inference,
)
from .fullgraph import SAMVTPFullGraphReplayRunner
from .inference import (
    compatible_input_hw,
    depth_from_uint16_mm,
    infer_sequence,
    list_depths,
    list_images,
    read_depth_metres,
)
from .stream import SAMVTPStreamProcessor, StreamStepResult, run_stream_loop

__version__ = "0.2.0"

__all__ = [
    "SAMVTP",
    "SAMVTPConfig",
    "CameraIntrinsics",
    "TARTANGROUND",
    "geometry_channels",
    "depth_from_uint16_mm",
    "build_model",
    "describe",
    "read_checkpoint",
    "load_checkpoint_state",
    "resolve_path",
    "tune_for_inference",
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
