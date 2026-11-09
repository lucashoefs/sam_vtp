# Integration

Conventions for running SAM-VTP on a physical camera. See the [README](../README.md) for the API.

## Intrinsics

Geometry is computed from depth and intrinsics alone. Supply intrinsics at the resolution they were calibrated at; the default is the TartanGround camera (640x640, 90° FOV).

```python
from sam_vtp import CameraIntrinsics, build_model

cam = CameraIntrinsics.from_K(camera_info.k, width=848, height=480)
model = build_model("sam_vtp.pt", device=dev, camera=cam)
model.warmup()  # run lazy CUDA initialisation before going live
```

## Depth

Align depth to the colour stream. Non-positive values are treated as missing.

```python
from sam_vtp import depth_from_uint16_mm

depth_m, valid = depth_from_uint16_mm(aligned_depth_16uc1)  # 0 = no return
result = stream.step(rgb, depth=depth_m, depth_valid=valid)
```

Use `depth_valid` for anything a threshold on depth cannot express, such as sensor confidence, a tighter range limit, or holes filled by post-processing.

## Aspect ratio

Training used a square 90x90° image, so `fx/W == fy/H == 0.5`. `input_fit` handles cameras that differ and adjusts the intrinsics to match. Measured on a D455 at 848x480 (fx 424, fy 377), running at 512x512:

| `input_fit` | Fitted size | FOV (°) | `fx/W`, `fy/H` | Trade-off |
|---|---|---|---|---|
| `"stretch"` (default) | 848x480 | 90x65 | 0.500, 0.785 | Full sensor; magnified about 1.6x vertically relative to training |
| `"crop"` | 540x480 | 65x65 | 0.785, 0.785 | Isotropic; loses 36% of horizontal FOV |
| `"pad"` | 848x753 | 90x90 | 0.500, 0.500 | Matches training; adds synthetic border content |

Crop and pad target the aspect `fx:fy` rather than the output aspect, so the final resize is deliberately anisotropic. The best choice depends on the camera; frame-to-frame prediction change ranks the options on your own footage without labels.

## Image range

Set `normalization_mode` to the range of your images (default `"0_255"`). `"auto"` infers it per batch and can be misled by dark frames, so avoid it in deployment.

## Memory cadence

The bank holds a fixed number of frames, so its time span depends on the write rate. Training wrote about every 0.2 s, giving 7 slots a window of about 1.2 s. Writing every frame at 35 FPS shrinks that to about 0.2 s.

```python
SAMVTPStreamProcessor(model, memory_write_interval_s=0.2)  # wall clock, for live cameras
SAMVTPStreamProcessor(model, memory_write_every=3)         # every k-th frame, for fixed-rate sources
```

Every frame still gets a prediction; only the memory write is gated. With CUDA graph replay, each outcome has its own captured graph. This uses more workspace, and capture waits for a full bank, which takes k times as many frames at `memory_write_every=k`.

## Streaming

`stream.reset()` clears temporal state and rebuilds the graph runner, so subsequent frames run eagerly until the bank refills. Avoid triggering it from a sensitive staleness check.

Hold planning while `result.memory_fill_level < 1.0`, since the prediction has less temporal context than in training.
