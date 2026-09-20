# SAM-VTP

Traversability prediction from a video stream. Output is continuous rather than
binary: each pixel gets a value in `[0,1]` meaning expected cost-free progress
for a given platform, so fresh asphalt and a 25 degree gravel slope don't
collapse to the same "drivable" label.

The model keeps a short memory across frames so the prediction stays stable while
the camera moves, and reads it with attention over content rather than over image
coordinates. It takes RGB and metric depth; it needs no pose, odometry or
localisation of any kind.

This repository builds on **SAM 2 as a pinned git submodule**.

The ROS 2 stack that runs this on a robot (camera, planner, controller, safety
gate) is a separate repository:
[sam_vtp_jetacker](https://github.com/lucashoefs/sam_vtp_jetacker).

## Install

```bash
git clone --recurse-submodules <repository-url>
cd sam_vtp

# PyTorch first, for your CUDA build: https://pytorch.org/get-started/locally/
pip install -e third_party/sam2    # the pinned SAM 2; not on PyPI
pip install -e .                   # model + inference
pip install -e ".[labels]"         # also the label pipeline (adds OpenCV)
```

Python 3.10+ (SAM 2's floor), PyTorch 2.0+. If you already cloned without
`--recurse-submodules`, run `git submodule update --init --recursive` first.

The pin is asserted at build time: `build_config()` raises a named error if
SAM 2 has moved underneath it.

## Quick start

Frames must arrive in capture order; memory carries across them. A shuffled
folder won't error, it will just give you an incoherent result.

```python
import torch
from sam_vtp import build_model

model = build_model("sam_vtp.pt", device=torch.device("cuda"))

model.reset_memory()                     # between independent sequences
for t, (rgb, depth) in enumerate(stream):
    out = model.forward_frame(rgb,       # (1,3,H,W) RGB in [0,255]
                              is_init_frame=(t == 0),
                              depth=depth)   # (1,1,H,W) planar z-depth, METRES
    traversability = out["logits"].sigmoid()   # (1,1,H,W) in [0,1]
```

A whole clip at once:

```python
out = model(images, depth_sequence=depths)   # (B,T,...)
```

Command line:

```bash
sam-vtp-infer --images-dir frames/ --depth-dir depth/ \
              --checkpoint sam_vtp.pt --output-dir predictions/ \
              --height 512 --width 512
```

Writes `predictions/probability/*_prob.png` (continuous, 0-255) and
`predictions/binary/*_bin.png` (thresholded, default 0.5). CUDA-graph replay is
on by default when CUDA is available; `--no-fullgraph` turns it off.
`sam-vtp-smoke --checkpoint sam_vtp.pt` runs a few synthetic frames and prints
what the checkpoint actually built.

For a live camera, `SAMVTPStreamProcessor` does no camera I/O, just the loop:

```python
from sam_vtp import SAMVTPStreamProcessor

stream = SAMVTPStreamProcessor(model, resize_hw=(512, 512), threshold=0.5,
                               input_fit="pad", return_numpy=True)
while (frame := grab()) is not None:
    result = stream.step(frame.rgb, depth=frame.depth)
    if result.memory_fill_level < 1.0:
        continue                       # bank still filling
    plan(result.probabilities)         # (H, W) numpy array
```

See [`docs/integration.md`](docs/integration.md) for camera intrinsics,
aspect-ratio fitting and memory cadence.

## Component switches

Three parts can be switched off at runtime:

| Switch | Off means |
|---|---|
| `memory` | every frame predicted from scratch, bank never read or written |
| `geometry` | RGB only, no depth-derived feature |
| `refine` | no residual refinement head |

```python
model = build_model("sam_vtp.pt", device=dev, components={"memory": False})
model.set_components(geometry=False)          # chainable, returns the model

with model.components(geometry=False):        # temporary, restored on exit
    rgb_only = model.forward_frame(rgb, is_init_frame=True)
```

`--no-memory`, `--no-geometry` and `--no-refine` do the same on the CLI.

A switch picks which of the checkpoint's weights are used. It cannot add a
component the checkpoint lacks, and `set_components` raises instead of running an
untrained module. Any deviation from the trained configuration shows up as
`[ABLATED: ...]` in `describe(model)`.

Switching after a CUDA graph is captured raises. The graph has the old
configuration baked in.

## Input conventions

| Input | Requirement | If it's wrong |
|---|---|---|
| **Depth** | Planar z-depth along the optical axis, in **metres** | The geometry channels are metric. Millimetres are off by 1000x. |
| **Intrinsics** | Yours, at the resolution they were calibrated at | The default is the TartanGround camera (640x640, 90 degree FOV). Wrong values tilt every slope. |
| **Frame order** | Capture order, one sequence per `reset_memory()` | Memory carries across frames, so a shuffled folder gives an incoherent result rather than an error. |

## How it works

<p align="center"><img src="docs/architecture.png" width="100%" alt="SAM-VTP architecture"></p>

RGB and depth are encoded and summed, fused with the memory bank by memory
attention, then given a residual correction from full-resolution RGB before
being written back. Starred blocks are what SAM-VTP adds on top of SAM 2.

Per frame, in `modeling/sam_vtp.py`:

1. **Image encoder** (Hiera-T + FPN, from SAM 2) gives features at strides 4, 8
   and 16.
2. **Geometry** (optional) turns depth into log-depth, slope, roughness, height
   and validity, added to the stride-16 level through a zero-initialised gate.
3. **Memory attention** conditions those features on a 7-slot bank.
4. **Mask decoder** produces logits under a fixed learned prompt.
5. **Refinement head** adds a residual from full-resolution RGB.
6. **Memory encoder** writes (features, prediction) back into the bank.

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

### Where this diverges from SAM 2

Six differences, each marked `DIVERGENCE n` in the source. Each one changes the
numbers, which is why `SAMVTPBase` overrides SAM 2's frame loop instead of
calling it.

| # | Divergence | Where |
|---|---|---|
| 1 | `no_mem_embed` is written back into `vision_feats`, so the conditioning frame's *stored* memory includes it | `sam_vtp.py` |
| 2 | `object_score_logits` comes from the prediction itself, not from `pred_obj_score_head` | `sam_vtp.py` |
| 3 | The object-pointer horizon is clamped by frames actually *seen*, not by clip length, which is unknown when streaming | `sam_vtp_base.py` |
| 4 | Memory slot selection at unit stride, and the learned prompt is resampled to the running feature grid | `sam_vtp_base.py`, `sam_vtp.py` |
| 5 | The mask is upsampled to the **input** size, not `image_size`, switching to bicubic at scale >= 3 | `sam_vtp.py` |
| 6 | `_prepare_backbone_features` returns four values in SAM 2; the first is discarded | `sam_vtp.py` |

**The prompt is learned, not clicked.** SAM 2 is interactive. Here the target is
always "ground this platform can drive on", so the prompt is fixed and learned
during training. Point coordinates are accepted and ignored. A checkpoint
carrying SAM 2's `point_embeddings` or `mask_downscaling` weights will not load.

**Checkpoints load strictly.** The architecture is read back from the weights
rather than from flags, and `build_model` raises on any missing or unexpected
key.

## Label generation

The training target is generated, not hand-annotated:

```
y = semantic_prior(class) x slope_factor x roughness_factor x height_gate
```

```bash
sam-vtp-labels --root /data/tartanground --workers 24
```

See [`docs/labels.md`](docs/labels.md) for the taxonomy and the geometry.

## Layout

```
third_party/sam2/       pinned SAM 2 submodule (Apache 2.0, not vendored)
src/sam_vtp/
  build.py              config assembly, asserts the SAM 2 pin
  checkpoint.py         checkpoint reading, architecture inferred from weights
  modeling/
    sam_vtp.py          the frame loop; divergences 1, 2, 5, 6
    sam_vtp_base.py     SAM2Base subclass; divergences 3, 4
    patches.py          cached Hiera, position encoding, RoPE attention
    geometry.py         intrinsics, depth-derived channels, geometry encoder
    prompt.py           the fixed learned prompt
    refine.py           full-resolution RGB residual head
  configs/              delta YAML layered over SAM 2's config
  inference.py          sequence inference, depth loading
  stream.py             camera-agnostic streaming loop
  fullgraph.py          optional CUDA-graph replay of the forward pass
  cli.py, smoke.py      entry points
  labels/               taxonomy.py, generate.py
tests/
  check_fullgraph.py    CUDA-graph replay vs the eager path
  check_cadence.py      memory-write cadence gate (CPU only, no checkpoint)
scripts/
  infer_sequence.py, smoke_test.py   run without installing the package
```

Run `tests/check_fullgraph.py` after changing `modeling/sam_vtp.py`.
`fullgraph.py` reimplements the forward pass and is what runs by default on
CUDA, so the two drift apart silently.

## Citation

```bibtex
[TBD]
```

## License

MIT, see `LICENSE`. Depends on SAM 2 (Apache 2.0) as a submodule and subclasses
several of its modules. See `NOTICE`.
