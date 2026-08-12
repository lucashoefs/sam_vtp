# Integration guide

Everything you need to run SAM-VTP on a real camera. The [README](../README.md)
covers the API; this covers the conventions that produce a plausible but wrong
map when violated.

## Pose

Only the *relative* transform between two frames is ever used, and that is
invariant to the frame the poses are expressed in, so ROS `odom` is fine and
there is no need to convert ENU to NED. What matters is the camera axis
convention, and the helpers handle it:

```python
from sam_vtp_core import CameraIntrinsics, build_model, pose_from_optical

cam = CameraIntrinsics.from_K(camera_info.k, width=848, height=480)   # NATIVE resolution
model = build_model("sam_vtp.pt", device=dev, camera=cam)
model.warmup()                       # pay lazy CUDA init before the robot is live

# TF: odom -> <camera>_color_optical_frame, looked up AT THE IMAGE TIMESTAMP
tf = buffer.lookup_transform("odom", "d455_color_optical_frame", img.header.stamp)
t, q = tf.transform.translation, tf.transform.rotation
pose = pose_from_optical([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
```

`pose_from_flu` exists for a REP-103 body frame, but prefer the optical frame:
the translation must belong to the same frame as the rotation, and a base-link
position with a camera rotation carries a lever-arm error that does not cancel
when the robot turns.

Sample the pose at the **image timestamp**, not at processing time. At 0.5 m/s,
60 ms of skew is 3 cm, which shows up as smearing in the warped memory.

## Depth

```python
from sam_vtp_core import depth_from_uint16_mm

depth_m, valid = depth_from_uint16_mm(aligned_depth_16uc1)   # 0 means no return
result = stream.step(rgb, depth=depth_m, pose=pose, depth_valid=valid)
```

Non-positive depths are already treated as missing, so `depth_valid` is for
evidence a threshold cannot express: the camera's own confidence, a range limit
tighter than the model's 100 m, or a depth map whose holes have been filled with
plausible-looking values by a post-processing filter.

Align depth to the colour stream. On a D455 the two have different intrinsics and
a baseline offset, so unaligned depth computes geometry on a grid that does not
correspond to the image.

## Aspect ratio

Training used a square 90x90 degree image, so `fx/W == fy/H == 0.5`: a pixel
subtends the same angle in both axes. A 16:9 camera with `fx != fy` does not, and
the encoder has never seen anything else. `input_fit` decides what to do about
it, and adjusts the intrinsics to match, the part that is easy to get wrong by
hand.

Measured on a D455 at 848x480 (fx 424, fy 377), running at 512x512:

| `input_fit` | fitted size | FOV | `fx/W`, `fy/H` | trade |
|---|---|---|---|---|
| `"stretch"` (default) | 848x480 | 90 x 65 | 0.500, 0.785 | whole sensor, invents nothing, but vertically magnified ~1.6x versus training |
| `"crop"` | 540x480 | 65 x 65 | 0.785, 0.785 | isotropic, invents nothing, loses 36% of horizontal FOV |
| `"pad"` | 848x753 | 90 x 90 | 0.500, 0.500 | isotropic **and** an exact match to training, at the cost of fabricated border content |

Crop and pad target the aspect `fx:fy` rather than the output aspect; the final
resize is then anisotropic on purpose, which is what equalises the two normalised
focal lengths.

None of the three is free, and which one wins is empirical on your camera. The
frame-to-frame stability metric (warp *t-1* into *t*, compare with *t*) ranks them
on your own footage without needing any labels.

## Image range

Set `normalization_mode` to the range your images arrive in (`"0_255"` is the
default). `"auto"` infers it per batch and a dark frame can fool it, so pin it in
deployment.

## Memory cadence

The bank holds a fixed number of *frames*, so how much time it spans depends on
your frame rate. Training used about 0.2 s between memory writes, giving the
7-slot bank roughly a 1.2 s window. At 35 FPS with a write every frame that
collapses to about 0.2 s and the bank fills with near-duplicate views.

```python
SAMVTPStreamProcessor(model, memory_write_interval_s=0.2)  # wall-clock, best for live cameras
SAMVTPStreamProcessor(model, memory_write_every=3)         # every k-th frame, for fixed-rate sources
```

Every frame still gets a prediction, only the write is gated. Cadence needs eager
control flow, so it turns off CUDA-graph replay.

## Streaming

`stream.reset()` drops temporal state and rebuilds the graph runner, so the next
~17 frames run eagerly while the bank refills. Don't wire it to a trigger-happy
staleness check.

Gate planning on `result.memory_fill_level < 1.0`: before the bank is full the
prediction is made from less temporal context than the model was trained with.
