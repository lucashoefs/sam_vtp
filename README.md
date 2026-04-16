# SAM-VTP Core

SAM-VTP Core is an inference-first release of **SAM-VTP** (SAM Video Traversability Predictor), built for real-time traversability segmentation in robotic perception pipelines.

SAM-VTP leverages SAM2's native video memory design to produce temporally consistent traversability maps from RGB streams. Rather than segmenting each frame independently, it carries compact memory across time so predictions stay stable under ego-motion, appearance shifts, and partial occlusions.

At a high level, SAM-VTP runs the following loop on each frame:
1. SAM2 image encoder extracts multi-scale visual features.
2. Memory attention conditions the current frame on recent temporal memory.
3. SAM decoder predicts traversability logits/mask for the frame.
4. Memory encoder writes updated state for the next step.

This repository is intentionally deployment-oriented:
- Core SAM-VTP model implementation.
- Optional full-forward CUDA Graph replay for higher throughput.
- Simple CLI and Python APIs for sequence and real-time stream inference.
- No dataset loaders, training loops, benchmarking suites, or unrelated baselines.

## Repository Layout

- `src/sam_vtp_core/model.py`: SAM-VTP core model
- `src/sam_vtp_core/fullgraph.py`: optional full-forward CUDA graph replay runner
- `src/sam_vtp_core/inference.py`: image-sequence utilities
- `src/sam_vtp_core/stream.py`: real-time stream processor API (camera-agnostic)
- `scripts/infer_sequence.py`: ready-to-run CLI for sequence prediction
- `scripts/smoke_test.py`: quick model wiring sanity check
- `configs/sam2.1_hiera_t.yaml`: default SAM2 config used by SAM-VTP

## Install

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2) Install PyTorch (pick your CUDA build)

Follow the official selector for your machine:
https://pytorch.org/get-started/locally/

### 3) Install SAM-VTP Core package (includes SAM2 dependency)

```bash
pip install -e .
```

Installed commands:
- `sam-vtp-infer`
- `sam-vtp-smoke`

## Inputs You Need

1. A SAM-VTP checkpoint
2. A folder of RGB frames (sorted lexicographically)

## Quick Start

```bash
sam-vtp-infer \
  --images-dir /path/to/frames \
  --output-dir /path/to/predictions \
  --sam2-cfg configs/sam2.1_hiera_t.yaml \
  --sam-vtp-checkpoint /path/to/sam_vtp.pt \
  --height 512 --width 512
```

On CUDA devices, full-forward CUDA Graph replay is enabled by default.
Use `--no-fullgraph` to force eager mode.

## Smoke Test

```bash
sam-vtp-smoke \
  --sam2-cfg configs/sam2.1_hiera_t.yaml \
  --sam-vtp-checkpoint /path/to/sam_vtp.pt \
  --height 512 --width 512
```

Outputs are written to:
- `output-dir/probability/*_prob.png`
- `output-dir/binary/*_bin.png`

## Fast Inference Mode (CUDA Graph)

```bash
sam-vtp-infer \
  --images-dir /path/to/frames \
  --output-dir /path/to/predictions \
  --sam2-cfg configs/sam2.1_hiera_t.yaml \
  --sam-vtp-checkpoint /path/to/sam_vtp.pt \
  --height 512 --width 512 \
  --capture-warmup 1 --fullgraph-min-history 16
```

If capture fails on your stack, execution safely falls back to eager mode.

## Python API

```python
import torch
from sam_vtp_core import build_model, infer_sequence, list_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(
    sam2_cfg_path="configs/sam2.1_hiera_t.yaml",
    sam_vtp_checkpoint_path="/path/to/sam_vtp.pt",
    device=device,
)

frames = list_images("/path/to/frames")
stats = infer_sequence(model, frames, output_dir="/path/to/predictions", resize_hw=(512, 512))
print(stats)
```

## Real-Time Streaming API (Camera-Agnostic)

```python
import torch
from sam_vtp_core import SAMVTPStreamProcessor, build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(
  sam2_cfg_path="configs/sam2.1_hiera_t.yaml",
  sam_vtp_checkpoint_path="/path/to/sam_vtp.pt",
  device=device,
)

stream = SAMVTPStreamProcessor(
  model=model,
  resize_hw=(512, 512),   # Optional. Use for fixed deployment resolution.
  use_fullgraph=None,     # None -> auto-enable on CUDA.
  threshold=0.5,
)

def get_next_rgb_frame():
  # Replace with your camera/frame source.
  # Expected frame format: HxWx3 or 3xHxW (numpy or torch), RGB, [0,255].
  return None

while True:
  frame = get_next_rgb_frame()
  if frame is None:
    break

  result = stream.step(frame)
  traversability_prob = result.probabilities[0, 0]   # torch.Tensor [H, W]
  traversability_bin = result.binary_mask[0, 0]      # torch.Tensor [H, W]
  # Send traversability_prob / traversability_bin to planning stack or visualization.
```

Callback-style loop API:

```python
from sam_vtp_core import run_stream_loop

def frame_provider():
  # Return one frame or None to stop.
  return None

def on_result(result):
  print(result.frame_index, result.elapsed_ms, result.graph_status)

num_processed = run_stream_loop(stream, frame_provider=frame_provider, on_result=on_result)
print("processed", num_processed)
```

## Notes

- Input tensors are expected in RGB with values in the `[0, 255]` range.
- The default model config uses `num_maskmem=7`.
- Non-aligned sizes (for example 550x688 from RUGD) are padded internally to a SAM2-compatible processing shape and logits are cropped back to the requested output size.
- If frame size changes during a live stream, the stream processor resets temporal memory by default (`reset_on_size_change=True`).
- This package is designed for inference-first public release.
