# SAM-VTP

Traversability prediction from a video stream. Output is continuous rather than
binary: each pixel gets a value in `[0,1]` meaning expected cost-free progress
for a given platform, so fresh asphalt and a 25 degree gravel slope don't
collapse to the same "drivable" label.

The model keeps a short memory across frames so the prediction stays stable while
the camera moves, and reads it with attention over content rather than over image
coordinates. It takes RGB and metric depth; it needs no pose, odometry or
localisation of any kind.

This repository builds on **SAM 2 as a pinned git submodule** rather than a
vendored copy of its layers. SAM 2's modules are imported and subclassed where
SAM-VTP has to diverge; everything SAM-VTP adds — geometry, the learned prompt,
the refinement head, the streaming frame loop — lives in `src/sam_vtp/`.

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

The submodule is pinned to a specific SAM 2 commit, and the build asserts that
pin: `build_config()` checks every upstream `_target_` it depends on and raises a
named error if SAM 2 has moved underneath it, rather than silently assembling a
different network.

## Quick start

Frames must arrive in capture order, memory carries across them. A shuffled
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
what the checkpoint actually built — the quickest way to check a file is what you
think it is.

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
| `memory` | frame-wise; every frame predicted from scratch, bank never read or written |
| `geometry` | RGB-only; no depth-derived feature |
| `refine` | no residual refinement head |

```python
model = build_model("sam_vtp.pt", device=dev, components={"memory": False})
model.set_components(geometry=False)          # chainable, returns the model

with model.components(geometry=False):        # temporary, restored on exit
    rgb_only = model.forward_frame(rgb, is_init_frame=True)
```

`--no-memory`, `--no-geometry` and `--no-refine` do the same on the CLI.

A switch selects which of the checkpoint's weights are *used*; it cannot add a
component the checkpoint lacks, and `set_components` raises rather than running a
randomly-initialised module. Turning one off is a deliberate ablation, not a
neutral setting: the remaining weights were fitted alongside it and now run
outside their training distribution. `describe(model)` appends an explicit
`[ABLATED: ...]` clause whenever the active configuration differs from the
as-trained one, so a log line distinguishes the two.

Switching components after a CUDA graph has been captured raises, the graph has
the old configuration baked into its control flow.

## Inputs that have to be right

These conventions give you a plausible but wrong map if violated, and nothing
downstream can detect it. Worth checking once at integration time.

| Input | Requirement | If it's wrong |
|---|---|---|
| **Depth** | Planar z-depth along the optical axis, in **metres** | The geometry channels are metric. Millimetres are off by 1000x, normalised `[0,1]` depth is meaningless. |
| **Intrinsics** | Yours, at the resolution they were calibrated at | The default is the TartanGround simulator camera (640x640, 90 degree FOV). Wrong intrinsics tilt every slope and shift every correspondence. |
| **Frame order** | Capture order, one sequence per `reset_memory()` | Memory carries across frames, so a shuffled folder gives an incoherent result rather than an error. |

**No pose or odometry is required anywhere.** The geometry channels are a
function of depth and intrinsics alone, so the perception stack carries no
localisation dependency. A geometry checkpoint run without depth raises rather
than quietly falling back.

## How it works

<p align="center"><img src="docs/architecture.png" width="100%" alt="SAM-VTP architecture"></p>

RGB and depth are encoded and summed, the result is fused with the memory bank
by memory attention, and the decoder's output gets a residual correction from
full-resolution RGB before being written back into the bank. Starred blocks are
what SAM-VTP adds on top of SAM 2.

Per frame, in `modeling/sam_vtp.py`:

1. **Image encoder** (Hiera-T + FPN, from SAM 2) gives features at strides 4, 8
   and 16.
2. **Geometry** (optional) derives log-depth, slope, roughness, height and
   validity from depth and adds them to the stride-16 level through a
   zero-initialised gate, so a geometry model starts out identical to one without.
3. **Memory attention** conditions those features on a 7-slot bank.
4. **Mask decoder** produces logits under a fixed learned prompt.
5. **Refinement head** adds a residual conditioned on full-resolution RGB.
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

### Where this diverges from SAM 2

Six behavioural differences, each marked `DIVERGENCE n` at its site in the
source. They are not stylistic — each one moves the numbers, and together they
are why `SAMVTPBase` overrides rather than calls upstream's frame loop.

| # | Divergence | Where |
|---|---|---|
| 1 | `no_mem_embed` is written back into `vision_feats`, so the conditioning frame's *stored* memory includes it | `sam_vtp.py` |
| 2 | `object_score_logits` comes from the prediction itself, not from `pred_obj_score_head` | `sam_vtp.py` |
| 3 | The object-pointer horizon is clamped by frames actually *seen*, not by total clip length — which is unknowable when streaming | `sam_vtp_base.py` |
| 4 | Memory slot selection at unit stride, and the learned prompt is resampled to the running feature grid | `sam_vtp_base.py`, `sam_vtp.py` |
| 5 | The mask is upsampled to the **input** size, not `image_size`, switching to bicubic at scale >= 3 | `sam_vtp.py` |
| 6 | `_prepare_backbone_features` returns four values upstream; the first is discarded | `sam_vtp.py` |

**The prompt is learned, not clicked.** SAM 2 is interactive; here there is no
user and the region of interest is always "ground this platform can drive on", so
the prompt is fixed and learned during training. Point coordinates passed to the
decoder are accepted and ignored. A checkpoint carrying SAM 2's
`point_embeddings` / `mask_downscaling` weights is a different model and won't
load.

**There is no ego-motion warp.** An earlier version resampled each memory slot
into the current camera frame using depth and relative pose. It was removed: it
cost a localisation dependency and did not measurably improve accuracy.
Checkpoints from that era carry an inert all-zero `no_corr_embed`, which
`strip_inert_keys` drops; one with a *non-zero* embed raises rather than scoring
a model nobody trained.

**Checkpoints load strictly.** The architecture is read back from the weights
rather than from flags (`overrides_from_state`), and `build_model` raises on any
missing or unexpected key — a lenient load would drop the geometry encoder and
refinement head silently and run an ablated model under the full model's name.
The target key count is 482, not 483, for the `no_corr_embed` reason above.

## Verifying a change

Two checks, both self-contained — no stored reference data, nothing to download.

```bash
python tests/check_fullgraph.py --checkpoint sam_vtp.pt          # needs a GPU
python tests/check_fullgraph.py --checkpoint sam_vtp.pt --variant no_memory
python tests/check_cadence.py                                    # CPU, no checkpoint
```

`fullgraph.py` is a second implementation of the forward pass, written over
tensorised state so it can be captured as a CUDA graph — and it is the path that
runs by default on CUDA, so it is what a robot executes. `check_fullgraph.py`
runs a 40-frame clip twice through the same model, once eager and once replayed,
and reports the first replayed frame that moves. **Run it after any change to
`modeling/sam_vtp.py`**, or the two implementations drift apart silently.

`check_cadence.py` covers the memory-write gate: with `memory_write_every > 1`,
the graph runner's decision about *which* frames write has to stay in phase with
the eager path across the eager-to-replay handover. It needs no checkpoint and
no GPU, so it runs in seconds anywhere.

Structural drift in the submodule is caught at build time instead:
`build_config()` asserts every upstream `_target_` and every override landing
point, so a SAM 2 update that renames or re-nests a module raises a named error
rather than assembling a different network. Note that this catches renames, not
a changed numeric default inside a SAM 2 module — pin the submodule
deliberately and read SAM 2's changelog when you move it.

## Performance

RTX 5070 Ti, 512x512, steady state, median over 60 frames:

| Path | fp32 | channels_last + fp16 |
|---|---:|---:|
| Eager | 20.0 ms (50 fps) | 25.9 ms |
| CUDA-graph replay | 16.7 ms (60 fps) | **7.2 ms (140 fps)** |

The model is launch-bound at inference — many small kernels — which is why graph
replay is worth this much, and more so on an embedded part where the CPU issuing
the launches is weaker. `tune_for_inference(model)` applies the NHWC layout;
neither it nor fp16 is bit-exact, so don't enable them for a run whose numbers
have to reproduce an earlier one exactly.

## Label generation

The training target is generated, not hand-annotated:

```
y = semantic_prior(class) x slope_factor x roughness_factor x height_gate
```

```bash
sam-vtp-labels --root /data/tartanground --workers 24
```

See [`docs/labels.md`](docs/labels.md) for the taxonomy, the geometry, and the
four choices that shape the output most.

## Layout

```
third_party/sam2/       pinned SAM 2 submodule (Apache 2.0, not vendored)
src/sam_vtp/
  build.py              hydra/OmegaConf assembly + upstream-drift assertions
  checkpoint.py         checkpoint reading; architecture inferred from weights
  modeling/
    sam_vtp.py          the frame loop; divergences 1, 2, 5, 6
    sam_vtp_base.py     SAM2Base subclass; divergences 3, 4
    patches.py          cached Hiera / position encoding / RoPE attention
    geometry.py         intrinsics, depth-derived channels, geometry encoder
    prompt.py           the fixed learned prompt replacing SAM 2's click encoder
    refine.py           full-resolution RGB-guided residual head
  configs/              the delta YAML layered over SAM 2's config
  inference.py          sequence inference, depth loading
  stream.py             camera-agnostic streaming loop
  fullgraph.py          optional CUDA-graph replay of the full forward pass
  cli.py, smoke.py      entry points
  labels/               taxonomy.py, generate.py
tests/
  check_fullgraph.py    CUDA-graph replay vs the eager path
  check_cadence.py      memory-write cadence gate (CPU only, no checkpoint)
scripts/
  infer_sequence.py, smoke_test.py   run without installing the package
```

## Citation

```bibtex
[TBD]
```

## License

MIT, see `LICENSE`. This repository depends on SAM 2 (Apache 2.0) as a submodule
and subclasses several of its modules; see `NOTICE`.
