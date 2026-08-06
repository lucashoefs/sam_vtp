# SAM-VTP

Traversability prediction from a video stream. The output is continuous rather
than binary: each pixel gets a value in `[0,1]` meaning expected cost-free
progress for a given platform, so fresh asphalt and a 25 degree gravel slope
don't collapse to the same "drivable" label.

The model keeps a short memory across frames so the prediction stays stable
while the camera moves. That memory is resampled into the current camera frame
using depth and relative pose, so a stored token and the pixel it informs
describe the same piece of world rather than just the same image coordinate.

The repo has two parts:

1. **The model** (`src/sam_vtp_core/`), defined end to end in one file that
   depends only on PyTorch.
2. **The label pipeline** (`src/sam_vtp_core/labels/`), which builds the
   continuous training target from simulator depth and semantics.

---

## Install

```bash
# PyTorch first, for your CUDA build: https://pytorch.org/get-started/locally/
pip install -e .              # model + inference
pip install -e ".[labels]"    # also the label pipeline (adds OpenCV)
```

No SAM 2, Hydra or OmegaConf dependency; the architecture is defined in-tree.
Needs Python 3.8+ and PyTorch 2.0+.

---

## Usage

Frames have to arrive in capture order, since memory carries across them. A
shuffled folder won't error, it'll just give you an incoherent result.

### Python

```python
import torch
from sam_vtp_core import build_model

model = build_model("sam_vtp.pt", device=torch.device("cuda"))

model.reset_memory()                     # between independent sequences
for t, (rgb, depth, pose) in enumerate(stream):
    out = model.forward_frame(rgb,       # (1,3,H,W) RGB in [0,255]
                              is_init_frame=(t == 0),
                              depth=depth,   # (1,1,H,W) planar z-depth, METRES
                              pose=pose)     # (1,7) [x y z qx qy qz qw], NED
    traversability = out["logits"].sigmoid()   # (1,1,H,W) in [0,1]
```

Or a whole clip at once:

```python
out = model(images,                # (B,T,3,H,W)
            depth_sequence=depths, # (B,T,1,H,W)
            pose_sequence=poses)   # (B,T,7)
```

### Command line

```bash
sam-vtp-infer \
  --images-dir  frames/ \
  --depth-dir   depth/ \
  --poses       pose_lcam_front.txt \
  --checkpoint  sam_vtp.pt \
  --output-dir  predictions/ \
  --height 512 --width 512
```

You get `predictions/probability/*_prob.png` (the continuous map as 0-255) and
`predictions/binary/*_bin.png` (thresholded, default 0.5).

`sam-vtp-smoke --checkpoint sam_vtp.pt` runs a few synthetic frames and prints
what the checkpoint actually built, which is the quickest way to check a file
is what you think it is.

### Live camera

`SAMVTPStreamProcessor` does no camera I/O, just the loop. Frames come from
wherever you get them, as numpy arrays or tensors:

```python
from sam_vtp_core import SAMVTPStreamProcessor

stream = SAMVTPStreamProcessor(model, resize_hw=(512, 512), threshold=0.5)
while (frame := grab()) is not None:
    result = stream.step(frame.rgb, depth=frame.depth, pose=frame.pose)
    plan(result.probabilities[0, 0])
```

---

## Inputs that have to be right

Three conventions will give you a plausible but wrong map if they're violated,
and nothing downstream can detect it. Worth checking once at integration time.

| Input | Requirement | If it's wrong |
|---|---|---|
| **Depth** | Planar z-depth along the optical axis, in **metres** | The geometry channels and the warp are metric. Millimetres are off by 1000x, normalised `[0,1]` depth is meaningless. |
| **Pose** | Camera-to-world `[x y z qx qy qz qw]`, scalar-**last**, NED world frame | A scalar-first file loads fine and rotates the memory wrongly every frame. |
| **Intrinsics** | Yours, via `CameraIntrinsics` | The default is the TartanGround simulator camera (640x640, 90 degree FOV). Wrong intrinsics tilt every slope and shift every correspondence. |

```python
from sam_vtp_core import CameraIntrinsics, build_model

cam = CameraIntrinsics(fx=..., fy=..., cx=..., cy=..., width=1280, height=720)
model = build_model("sam_vtp.pt", device=dev, camera=cam)   # from factory calibration
```

Set `normalization_mode` to the range your images arrive in (`"0_255"` is the
default). `"auto"` infers it per batch and a dark frame can fool it, so pin it
in deployment.

If a geometry checkpoint is run without depth or pose the model raises instead
of quietly falling back. To run the RGB-only ablation on purpose, set
`model.warp_memory = model.use_geometry = False`.

---

## How it works

<p align="center"><img src="docs/architecture.png" width="100%" alt="SAM-VTP architecture"></p>

RGB and depth are encoded and summed, memory is warped into the current frame
before it's fused, and the decoder's output gets a residual correction from
full-resolution RGB before being written back into the memory bank. Starred
blocks are what SAM-VTP adds on top of SAM 2. The prompt block is a fixed
learned prompt rather than user clicks (see below).

`architecture.py` is the whole network. Per frame:

1. **Image encoder** (Hiera-T + FPN) gives features at strides 4, 8 and 16.
2. **Geometry** (optional) derives log-depth, slope, roughness, height and
   validity from the depth map and adds them to the stride-16 level through a
   zero-initialised gate, so a geometry model starts out identical to one
   without it.
3. **Memory attention** conditions those features on a 7-slot bank. Each slot
   is first resampled into the current camera frame using the current depth and
   the relative pose.
4. **Mask decoder** produces logits under a fixed learned prompt.
5. **Refinement head** adds a residual conditioned on full-resolution RGB,
   recovering detail below the decoder's stride-4 output.
6. **Memory encoder** compresses (features, prediction) back into the bank.

| Component | Parameters |
|---|---:|
| Image encoder (Hiera-T + FPN) | 27.2 M |
| Memory attention | 5.9 M |
| Mask decoder | 4.2 M |
| Learned prompt | 2.1 M |
| Memory encoder | 1.4 M |
| Geometry encoder | 127.6 k |
| Refinement head | 905 |
| **Total** | **41.2 M** |

Two things differ from SAM 2 and are worth knowing about before you read the
code.

**The prompt is learned, not clicked.** SAM 2 is interactive: you click, and the
prompt encoder embeds the clicks. Here there's no user and the region of
interest is always "ground this platform can drive on", so the prompt is fixed
and learned during training. `LearnedPrompt` holds it in three tensors. Point
coordinates passed to the decoder are accepted and ignored. A checkpoint
carrying SAM 2's `point_embeddings` / `mask_downscaling` weights is a different
model and won't load.

**Memory is ego-motion compensated.** SAM 2 tags each memory slot with a
position encoding that's a pure function of grid index, so memory attention
gets the prior "token *k* in memory is token *k* now", i.e. the same pixel. For
a near-static camera tracking an object that's close enough to true. On a
forward-driving robot it's false every frame: the field flows outward and
rescales, so the same pixel is a different piece of world each time. Supplying
the correspondence from depth and pose means the network doesn't have to
recover it from appearance. Cells with no counterpart in a past frame (out of
frustum, or behind the camera) get a learned `no_corr_embed` rather than zeros.
Occlusion isn't resolved geometrically; that's left to the network.

Checkpoints load **strictly**, and the architecture is read back from the
weights rather than from flags. The geometry encoder, warp embedding and
refinement head are all top-level modules, so a lenient load would drop every
one of their tensors without saying anything and run an ablated model under the
full model's name.

### Memory cadence

The bank holds a fixed number of *frames*, so how much time it spans depends on
your frame rate. Training used about 0.2 s between memory writes, giving the
7-slot bank roughly a 1.2 s window. At 35 FPS with a write every frame that
collapses to about 0.2 s and the bank fills with near-duplicate views.

```python
SAMVTPStreamProcessor(model, memory_write_interval_s=0.2)  # wall-clock, best for live cameras
SAMVTPStreamProcessor(model, memory_write_every=3)         # every k-th frame, for fixed-rate sources
```

Every frame still gets a prediction, only the write is gated. Cadence needs
eager control flow, so it turns off CUDA-graph replay.

---

## Label generation

<p align="center"><img src="docs/label-pipeline.png" width="100%" alt="Traversability supervision pipeline"></p>

RGB, depth and semantics on the left are the inputs; height, slope and the final
target `y` on the right are what the pipeline derives. The target can't be read
off either the semantics or the geometry alone, which is the reason for training
a model on it rather than computing it at runtime.

The training target isn't hand-annotated. It's built per frame as

```
y = semantic_prior(class) x slope_factor x roughness_factor x height_gate
```

so semantics say what the surface is and geometry says what state it's in. You
need both: a concrete floor is 1.00 flat and 0.30 at 30 degrees, and grass,
bushes, trunks and canopy all share one semantic class in the forest
environments.

The semantic prior is absolute rather than per-environment, so a threshold
chosen in one scene means the same thing in another.

```bash
sam-vtp-labels --root /data/tartanground --workers 24
```

This writes `<traj>/trav_lcam_front/{frame:06d}_lcam_front_trav.png`, 8-bit,
255 being fully traversable.

`labels/taxonomy.py` holds the class-to-score mapping and the reference platform
it's grounded in (about 0.15 m wheel radius, 0.20 m clearance, 30 degree stable
slope, 0.15 m step). `labels/generate.py` holds the geometry. Both docstrings
record why each constant is what it is, so read them before changing one.

Four choices shape the output more than the rest:

- **Camera height is estimated once per trajectory**, not per frame. It's a
  rigid-body constant, and a per-frame ground-pixel median swings by half a
  metre on vegetated terrain, which moves the height gate around.
- **Slope is measured at terrain scale** with an 11 px stencil. Depth resolves
  individual leaves, so a narrow stencil measures the tilt of whichever leaf a
  pixel landed on instead of the grade of the ground.
- **Modulation is smoothed at sigma 12 and normalised by the traversable mask.**
  The decoder predicts at stride 4, so finer detail can't be represented
  anyway. The mask normalisation stops a wall, which is legitimately
  near-vertical, from bleeding a dark halo onto the ground next to it. Class
  boundaries stay sharp either way, since only the modulation is smoothed and
  never the prior.
- **Height and slope are gravity-aligned** using per-frame pose. If you treat
  the optical axis as horizontal the height error grows linearly with range,
  around 0.87 m at 50 m per degree of pitch, so distant flat ground drifts up
  until it crosses the gate and gets labelled an obstacle. With no pose file the
  generator falls back to the level-camera assumption and tells you.

---

## Layout

```
src/sam_vtp_core/
  architecture.py   the entire network: SAM 2 blocks, geometry, warp, SAM-VTP
  factory.py        checkpoint-driven construction (strict)
  inference.py      sequence inference, depth/pose loading
  stream.py         camera-agnostic streaming loop
  fullgraph.py      optional CUDA-graph replay (RGB-only models)
  cli.py, smoke.py  entry points
  labels/
    taxonomy.py     class name -> absolute traversability
    generate.py     the supervision generator
scripts/
  infer_sequence.py           run without installing
  smoke_test.py
  verify_fullgraph_parity.py  eager vs graph-replay agreement
```

`fullgraph.py` is a second implementation of the forward pass, written over
tensorised state so it can be captured as a CUDA graph. It only covers RGB-only
models: the warp reads a variable set of stored poses, which isn't constant
control flow, so a geometry checkpoint turns graph mode off rather than
silently running unwarped. Since it duplicates the forward pass, run
`verify_fullgraph_parity.py` after touching `architecture.py`.

---

## Citation

```bibtex
[TBD]
```

## License

MIT, see `LICENSE`. Parts of `architecture.py` derive from SAM 2 (Apache 2.0),
see `NOTICE`.
