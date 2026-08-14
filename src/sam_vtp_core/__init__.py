"""SAM-VTP: temporally consistent, continuous traversability from a video stream.

    from sam_vtp_core import build_model
    model = build_model("sam_vtp.pt", device="cuda")
    out = model.forward_frame(rgb, is_init_frame=True, depth=depth, pose=pose)
    traversability = out["logits"].sigmoid()

The network is defined end to end in `architecture.py` and depends only on
PyTorch. The label pipeline lives in `sam_vtp_core.labels` and is imported
separately, since it needs OpenCV and is only used to build training data.
"""

from .architecture import (
    CameraIntrinsics,
    Q_FLU_TO_NED_BODY,
    Q_OPTICAL_TO_NED_BODY,
    SAMVTP,
    SAMVTPConfig,
    TARTANGROUND,
    geometry_channels,
    pose_from_flu,
    pose_from_optical,
    quat_mul,
    warp_features,
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
    load_poses,
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
    "warp_features",
    "pose_from_optical",
    "pose_from_flu",
    "quat_mul",
    "Q_OPTICAL_TO_NED_BODY",
    "Q_FLU_TO_NED_BODY",
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
    "load_poses",
    "read_depth_metres",
    "compatible_input_hw",
    "SAMVTPStreamProcessor",
    "StreamStepResult",
    "run_stream_loop",
    "SAMVTPFullGraphReplayRunner",
    "__version__",
]
