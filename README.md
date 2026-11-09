# SAM-VTP

Continuous traversability segmentation from RGB-D video. SAM-VTP builds on [SAM 2](https://github.com/facebookresearch/sam2), adapting its video memory for prompt-free prediction and fusing geometry measured from depth, to produce a per-pixel score in `[0,1]` at each frame. It requires no pose or odometry.

The ROS 2 deployment stack is in [sam_vtp_jetacker](https://github.com/lucashoefs/sam_vtp_jetacker).

Measured at 512x512 with a full memory bank and CUDA graph replay: 21.4 FPS on a Jetson Orin Nano and 30.1 FPS on a Jetson AGX Orin.

## Install

Requires Python 3.10+ and [PyTorch](https://pytorch.org/get-started/locally/) 2.0+. SAM 2 is a pinned submodule, and `build_config()` raises an error if it has moved.

```bash
git clone --recurse-submodules https://github.com/lucashoefs/sam_vtp.git
cd sam_vtp
pip install -e third_party/sam2
pip install -e .            # model and inference
pip install -e ".[labels]"  # adds the label pipeline (OpenCV)
```

Trained weights are not yet published.

## Usage

Frames must be in capture order. Call `reset_memory()` between independent sequences.

```python
import torch
from sam_vtp import CameraIntrinsics, build_model

cam = CameraIntrinsics.from_K(K, width=848, height=480)  # native resolution
model = build_model("sam_vtp.pt", device=torch.device("cuda"), camera=cam)

model.reset_memory()
for t, (rgb, depth) in enumerate(stream):
    out = model.forward_frame(
        rgb,                     # (1,3,H,W), RGB in [0,255]
        is_init_frame=(t == 0),
        depth=depth,             # (1,1,H,W), planar z-depth in metres
    )
    traversability = out["logits"].sigmoid()  # (1,1,H,W)
```

For a clip, call `model(images, depth_sequence=depths)` with `(B,T,...)` tensors. `SAMVTPStreamProcessor` wraps the per-frame loop for live cameras.

From the command line:

```bash
sam-vtp-infer --images-dir frames/ --depth-dir depth/ --checkpoint sam_vtp.pt \
              --output-dir predictions/ --height 512 --width 512
sam-vtp-smoke --checkpoint sam_vtp.pt  # synthetic frames; prints the built configuration
```

`sam-vtp-infer` uses CUDA graph replay by default when CUDA is available; `--no-fullgraph` disables it.

### Input conventions

Violating either of these produces a plausible but incorrect map, with no error.

| Input | Requirement |
|---|---|
| Depth | Planar z-depth along the optical axis, in metres |
| Intrinsics | Your camera's, at the calibrated resolution. The default is the TartanGround camera (640x640, 90° FOV) |

See [`docs/integration.md`](docs/integration.md) for aspect-ratio fitting, image range and memory cadence.

## Architecture

<p align="center"><img src="docs/architecture.png" width="100%" alt="SAM-VTP architecture"></p>

For each frame (41.2 M parameters in total):

1. **Image encoder:** Hiera-T with an FPN, from SAM 2, giving features at strides 4, 8 and 16.
2. **Geometry encoder (128 k):** log-depth, slope, roughness, height and validity, added to the stride-16 features through a zero-initialised gate.
3. **Memory attention:** conditions those features on a 7-frame bank.
4. **Mask decoder:** predicts logits under a fixed learned prompt, which replaces SAM 2's prompt encoder.
5. **Refinement head (905):** adds a residual from full-resolution RGB.
6. **Memory encoder:** writes the features and prediction back to the bank.

Checkpoints load strictly, and the architecture is read from the weights. SAM 2 checkpoints are not compatible.

### Differences from SAM 2

`SAMVTPBase` overrides SAM 2's frame loop. Each difference is marked `DIVERGENCE n` in the source:

| # | Change | File |
|---|---|---|
| 1 | `no_mem_embed` is written back into `vision_feats`, so the conditioning frame's stored memory includes it | `sam_vtp.py` |
| 2 | `object_score_logits` comes from the prediction, not `pred_obj_score_head` | `sam_vtp.py` |
| 3 | The object-pointer horizon is clamped by frames seen, not clip length | `sam_vtp_base.py` |
| 4 | Memory slots are selected at unit stride, and the learned prompt is resampled to the feature grid | `sam_vtp_base.py`, `sam_vtp.py` |
| 5 | Masks are upsampled to the input size, not `image_size`, using bicubic at scale >= 3 | `sam_vtp.py` |
| 6 | The first return value of `_prepare_backbone_features` is discarded | `sam_vtp.py` |

### Ablation

Memory, geometry and refinement can be disabled at runtime. This selects among the checkpoint's existing weights without retraining, and `describe(model)` reports the result as ablated.

```python
model = build_model("sam_vtp.pt", device=dev, components={"memory": False})

with model.components(geometry=False):
    out = model.forward_frame(rgb, is_init_frame=True)
```

The CLI equivalents are `--no-memory`, `--no-geometry` and `--no-refine`.

## Label generation

Training targets are generated from simulator depth and semantics:

```
y = semantic_prior(class) * slope_factor * roughness_factor * height_gate
```

```bash
sam-vtp-labels --root /data/tartanground --workers 24
```

See [`docs/labels.md`](docs/labels.md).

## Repository layout

```
third_party/sam2/       pinned SAM 2 submodule
src/sam_vtp/
  build.py              config assembly; asserts the SAM 2 pin
  checkpoint.py         checkpoint reading
  modeling/             frame loop, geometry, prompt, refinement, SAM 2 patches
  configs/              delta YAML over SAM 2's config
  inference.py          sequence inference and depth loading
  stream.py             streaming loop
  fullgraph.py          CUDA graph replay of the forward pass
  labels/               label pipeline
tests/                  graph-replay and memory-cadence checks
scripts/                inference and smoke test without installing
```

`fullgraph.py` reimplements the forward pass and is the default on CUDA. Run `tests/check_fullgraph.py` after changing `modeling/sam_vtp.py`.

## Citation

```bibtex
@misc{hoefs2026samvtp,
  title  = {{SAM-VTP}: Video Memory and Measured Geometry for Continuous Traversability Segmentation},
  author = {Hoefs, Lucas and Xie, Hui and An, Senjian and Zhang, Fangyi},
  year   = {2026}
}
```

## License

MIT. SAM 2 is used as a submodule under Apache 2.0; see `NOTICE`.
