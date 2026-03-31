## DST (Agent-based Territorial Style Substitution)

This repository contains `DST.py`, a command-line tool that:

- Subdivides an input image with a quadtree (splitting high-error regions)
- Encodes each tile with a neural encoder (from a `.keras` model)
- Substitutes each tile with its nearest-neighbor “style” patch from a style-image folder
- Optionally writes an MP4 animation of the substitution process

---

## Installation

### Requirements

- **Python**: 3.10+ recommended
- **OS**:
  - **Windows 10/11**: supported (CPU or NVIDIA GPU)
  - **Linux**: supported (CPU or NVIDIA GPU)
  - **macOS**: may work on CPU, but TensorFlow support can be limited depending on your Python/TF build

### Recommended hardware

- **Minimum (CPU)**: 4–8 cores, 16 GB RAM
- **Recommended (GPU)**: NVIDIA GPU with **8+ GB VRAM** (CUDA-capable), 32 GB RAM
  - TensorFlow GPU requires a compatible NVIDIA driver/CUDA stack for your TensorFlow build.

### Install steps

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Model + data layout

You need:

- **Model**: a Keras `.keras` file (zip-format) passed via `--model-path`
- **Input image**: passed via `--input-path`
- **Style images folder**: passed via `--style-folder` (all images inside are candidates)

The repo currently includes example assets:

- Input image: `inputs/Gemini_Generated_Image_f9sfhjf9sfhjf9sf-edit.png`
- Example output image: `outputs/substitution.jpg`

---

## Quick start

From the repo root:

```bash
python DST.py ^
  --model-path ".\checkpoints\saved_model.keras" ^
  --style-folder ".\inputs\Gemini_Generated_Image_am1379am1379am13\quadtree" ^
  --input-path ".\inputs\Gemini_Generated_Image_f9sfhjf9sfhjf9sf-edit.png" ^
  --threshold 2 --min-cell 6 ^
  --use-random-split --w-randomness 0.5 --h-randomness 0.1 ^
  --agent-population 48 ^
  --verbose
```

Notes:

- Output is written under a folder derived from the input filename (see `ensure_output_dir()` in `DST.py`).
- If you want to disable video writing: add `--no-save-substitution-video`.

---

## Inputs / CLI arguments

All “settings” at the top of `DST.py` are configurable via CLI flags (defaults match the file).

### Paths / I/O

- **`--model-path`**: Path to `.keras` model. Default: `../checkpoints/saved_model.keras`
- **`--style-folder`**: Folder of style images. Default: `inputs/.../quadtree` (see `DST.py`)
- **`--input-path`**: Input image path. Default: `inputs/...png` (see `DST.py`)
- **`--output-name`**: Output image filename (saved under input-derived folder). Default: `../outputs/substitution.jpg`

### Image + batching

- **`--img-width`**: Tile resize width for encoder. Default: 256
- **`--img-height`**: Tile resize height for encoder. Default: 256
- **`--style-batch-size`**: Batch size when encoding style images. Default: 64

### Quadtree subdivision

- **`--threshold`**: Subdivision error threshold. Default: 2
- **`--min-cell`**: Minimum leaf size in pixels. Default: 6
- **`--use-random-split`**: Enable randomized split positions (flag). Default: off
- **`--w-randomness`**: Width split randomness in \([0, 0.5]\). Default: 0.5
- **`--h-randomness`**: Height split randomness in \([0, 0.5]\). Default: 0.1

### Style-code options

- **`--use-global-avg-pool-for-codes` / `--no-use-global-avg-pool-for-codes`**: Default: on
- **`--use-memmap-for-style-codes` / `--no-use-memmap-for-style-codes`**: Default: off
- **`--style-codes-memmap-path`**: Default: `/tmp/style_codes.dat`

### Caching + nearest-neighbor config

- **`--style-cache-size`**: LRU cache size for loaded style images. Default: 128
- **`--patch-cache-size`**: LRU cache size for resized style patches. Default: 1024
- **`--nn-algorithm`**: scikit-learn NN algorithm. Default: `auto`
- **`--nn-metric`**: NN metric. Default: `euclidean`

### Style file-size filter

- **`--filter-small-style-files` / `--no-filter-small-style-files`**: Default: on
- **`--min-style-file-size-bytes`**: Default: 1200

### Video export

- **`--save-substitution-video` / `--no-save-substitution-video`**: Default: on
- **`--video-name`**: Default: `substitution_animation.mp4`
- **`--video-fps`**: Default: 24
- **`--video-codec-fourcc`**: Default: `mp4v`
- **`--video-queue-maxsize`**: Default: 128
- **`--save-every-n-substitutions`**: Default: 4
- **`--write-final-frame-at-end` / `--no-write-final-frame-at-end`**: Default: on

### Agent system

- **`--agent-population`**: Number of agents. Default: 24
- **`--agent-neighbor-k`**: Neighbor count for each tile in agent movement. Default: 24
- **`--agent-shuffle-each-round` / `--no-agent-shuffle-each-round`**: Default: off

### Reproducibility + logging

- **`--seed`**: RNG seed. Default: 42
- **`--verbose` / `--no-verbose`**: Default: off
- **`--style-log-every`**: Default: 20
- **`--subdivide-log-every`**: Default: 5000
- **`--agent-log-every`**: Default: 250
- **`--render-batch-detail` / `--no-render-batch-detail`**: Default: off

---

## Dependency versions

The pinned requirements depend on your platform (especially for TensorFlow). This repo provides version *ranges* in `requirements.txt` that are commonly compatible:

- `tensorflow>=2.16`
- `keras>=3.0`
- `numpy>=1.26,<3`
- `opencv-python>=4.8`
- `scikit-learn>=1.3`

If you need strict reproducibility, freeze your environment after install:

```bash
pip freeze > requirements-lock.txt
```

---

## Sample input / output

Sample input (from `inputs/`):

![Sample input](inputs/Gemini_Generated_Image_f9sfhjf9sfhjf9sf-edit.png)

Sample output (written to `outputs/` in this repo snapshot):

![Sample output](outputs/substitution.jpg)

