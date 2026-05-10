# mini-nerf

Single-file implementation of [Neural Radiance Fields (NeRF)](https://arxiv.org/abs/2003.08934), inspired by [Andrej Karpathy's microgpt](https://karpathy.github.io/2026/02/12/microgpt/).

Supports the NeRF Blender synthetic dataset (lego, chair, drums, ficus, hotdog, materials, mic, ship).

## Blog

[Spatial Intelligence - Part 1: NeRF](https://halannhile.github.io/posts/nerf/)

https://github.com/user-attachments/assets/1d7cc00d-999f-49b7-99e6-8ba2e1bfa76e

## Setup

```bash
python3 -m venv nerf-venv
source nerf-venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Download the NeRF Blender synthetic dataset (~770 MB) from [Kaggle](https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset):

1. Download and unzip the dataset
2. Rename the unzipped folder to `data` and place it in the repo root in this format:

```
data/
└── lego/
    ├── transforms_train.json
    ├── transforms_val.json
    ├── transforms_test.json
    ├── train/
    │   ├── r_0.png
    │   ├── r_1.png
    │   └── ...
    ├── val/
    └── test/
```

To train on a different scene (e.g. `chair`), update `cfg.scene_dir` at the top of `nerf_single.py`:

```python
cfg.scene_dir      = './data/chair'
cfg.checkpoint_dir = f'./checkpoints/{os.path.basename(cfg.scene_dir)}'
```

## Usage

### Train from scratch
```bash
python nerf_single.py train
```

### Resume training from a checkpoint
```bash
python nerf_single.py train --resume checkpoints/lego/latest.pt
```

### Render an MP4 video from a trained model
Generates a 360°-sweep MP4 video of the trained scene:
```bash
python nerf_single.py render --checkpoint checkpoints/lego/latest.pt
python nerf_single.py render --checkpoint checkpoints/lego/latest.pt --n_frames 120 --fps 30
```

- `--n_frames`: number of frames in the MP4. Default: `120`
- `--fps`: frames per second. Default: `30`

## Configuration

Edit the `Config` dataclass at the top of `nerf_single.py` to change settings. Key options:

| Field | Default | Description |
|---|---|---|
| `scene_dir` | `./data/lego` | Path to scene directory |
| `white_bkgd` | `True` | White background compositing (use `True` for Blender scenes) |
| `half_res` | `True` | Downsample images 2× (400×400 instead of 800×800) |
| `near` | `2.0` | Near plane for ray sampling |
| `far` | `6.0` | Far plane for ray sampling |
| `N_coarse` | `64` | Coarse samples per ray |
| `N_fine` | `128` | Fine samples per ray (hierarchical sampling) |
| `L_pos` | `10` | Positional encoding frequencies for 3D position |
| `L_dir` | `4` | Positional encoding frequencies for view direction |
| `W` | `256` | MLP hidden layer width |
| `D` | `8` | MLP depth (number of hidden layers) |
| `skip` | `4` | Layer index where input is concatenated (skip connection) |
| `n_iters` | `200000` | Training iterations |
| `batch_rays` | `4096` | Rays per training batch |
| `lr` | `5e-4` | Learning rate (with exponential decay) |
| `warmup_iters` | `1000` | Linear LR warmup steps before decay |
| `grad_clip` | `0.0` | Gradient clipping (0 = disabled) |
| `ema_decay` | `0.9999` | EMA decay for shadow weights used during video rendering |
| `use_amp` | `True` | bfloat16 autocast on MPS (no GradScaler needed) |
| `use_compile` | `False` | torch.compile (disabled: causes silent wrong gradients on MPS) |
| `chunk` | `32768` | Rays per chunk during full-image rendering (reduce if OOM) |
| `i_print` | `100` | Log loss/PSNR every N iterations |
| `i_save` | `10000` | Save checkpoint every N iterations |
| `i_video` | `100000` | Render and save MP4 video every N iterations |
