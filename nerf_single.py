"""
Neural Radiance Fields (NeRF), single-file implementation.
Edit Config to change settings.

Dataset: NeRF Blender synthetic scenes (lego, chair, drums, ficus, hotdog, materials, mic, ship).
See README.md for download instructions.

MPS training optimizations:
    1. Pre-cached rays     - compute all training rays once at startup, store on device
    2. bfloat16 AMP        - torch.autocast with bfloat16 on MPS (no GradScaler needed)
    3. EMA                 - exponential moving average shadow weights for video rendering
    4. torch.compile       - graph compilation for faster iteration (PyTorch 2.x)
    5. LR warmup + grad clip - linear warmup before exponential decay; clip_grad_norm_

Modes:
    python nerf_single.py train
    python nerf_single.py train --resume checkpoints/lego/latest.pt
    python nerf_single.py render --checkpoint checkpoints/lego/latest.pt
    python nerf_single.py render --checkpoint checkpoints/lego/latest.pt --n_frames 120 --fps 30
"""

import os, json, argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import imageio.v2 as imageio


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # data
    scene_dir:  str   = './data/lego'
    white_bkgd: bool  = True
    half_res:   bool  = True

    # rendering bounds
    near:       float = 2.0
    far:        float = 6.0

    # sampling
    N_coarse:   int   = 64
    N_fine:     int   = 128

    # positional encoding
    L_pos:      int   = 10
    L_dir:      int   = 4

    # network
    W:          int   = 256
    D:          int   = 8
    skip:       int   = 4

    # training
    n_iters:      int   = 200_000
    batch_rays:   int   = 4096
    lr:           float = 5e-4
    lr_decay:     int   = 250
    warmup_iters: int   = 1_000       # [opt 5] linear LR warmup
    grad_clip:    float = 0.0         # [opt 5] gradient clipping - 0 = disabled (original NeRF has no clipping)
    ema_decay:    float = 0.9999      # [opt 3] EMA decay

    # MPS optimizations
    use_amp:     bool  = True         # [opt 2] bfloat16 autocast
    use_compile: bool  = False        # [opt 4] torch.compile - disabled: causes silent wrong gradients on MPS

    # chunked rendering (avoids OOM on full images)
    chunk:      int   = 32_768

    # logging
    i_print:    int   = 100
    i_save:     int   = 10_000
    i_video:    int   = 100_000

    checkpoint_dir: str = ''


cfg = Config()
cfg.checkpoint_dir = f'./checkpoints/{os.path.basename(cfg.scene_dir)}'

# # For a different scene, uncomment and edit:
# cfg.scene_dir      = './data/chair'
# cfg.checkpoint_dir = f'./checkpoints/{os.path.basename(cfg.scene_dir)}'


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
device = (
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cpu')
)

# [opt 2] pick AMP dtype per device; MPS uses bfloat16, CUDA uses float16
amp_dtype = (
    torch.float16  if device.type == 'cuda' and cfg.use_amp else
    torch.bfloat16 if device.type == 'mps'  and cfg.use_amp else
    None
)
# float16 on CUDA requires GradScaler to prevent gradient underflow; bfloat16/MPS does not
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and amp_dtype == torch.float16))


# ─────────────────────────────────────────────────────────────────────────────
# [opt 3] EMA
# ─────────────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model: nn.Module):
        self.decay  = cfg.ema_decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.lerp_(m.data.float(), 1.0 - self.decay)


# ─────────────────────────────────────────────────────────────────────────────
# [opt 5] LR schedule: linear warmup then exponential decay
# ─────────────────────────────────────────────────────────────────────────────
def get_lr(i: int) -> float:
    if i < cfg.warmup_iters:
        return cfg.lr * (i + 1) / cfg.warmup_iters
    return cfg.lr * (0.1 ** (i / (cfg.lr_decay * 1000)))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (NeRF Blender format)
# ─────────────────────────────────────────────────────────────────────────────
def load_blender(split: str) -> Tuple[Tensor, Tensor, int, int, float]:
    with open(os.path.join(cfg.scene_dir, f'transforms_{split}.json')) as f:
        meta = json.load(f)

    imgs, poses = [], []
    for frame in meta['frames']:
        path = os.path.join(cfg.scene_dir, frame['file_path'] + '.png')
        img  = imageio.imread(path).astype(np.float32) / 255.0
        if img.shape[-1] == 4:
            img = img[..., :3] * img[..., 3:] + (1.0 - img[..., 3:])
        imgs.append(img)
        poses.append(np.array(frame['transform_matrix'], np.float32))

    imgs  = np.stack(imgs)
    poses = np.stack(poses)
    H, W  = imgs.shape[1:3]
    focal = 0.5 * W / np.tan(0.5 * float(meta['camera_angle_x']))

    if cfg.half_res:
        H2, W2 = H // 2, W // 2
        imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2)   # NHWC -> NCHW
        imgs_t = F.interpolate(imgs_t, (H2, W2), mode='area')
        imgs   = imgs_t.permute(0, 2, 3, 1).numpy()           # NCHW -> NHWC
        focal /= 2.0
        H, W   = H2, W2

    return (
        torch.tensor(imgs,  dtype=torch.float32),
        torch.tensor(poses, dtype=torch.float32),
        H, W, focal,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ray utilities
# ─────────────────────────────────────────────────────────────────────────────
def get_rays(H: int, W: int, focal: float, c2w: Tensor) -> Tuple[Tensor, Tensor]:
    j, i = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=c2w.device),
        torch.arange(W, dtype=torch.float32, device=c2w.device),
        indexing='ij',
    )
    dx   = (i - W * 0.5) / focal
    dy   = -(j - H * 0.5) / focal
    dz   = -torch.ones_like(i)
    dirs = torch.stack([dx, dy, dz], dim=-1)
    rays_d = (dirs[..., None, :] * c2w[:3, :3]).sum(-1)
    rays_o = c2w[:3, 3].expand_as(rays_d)
    return rays_o, rays_d


def pose_spherical(theta: float, phi: float, radius: float) -> np.ndarray:
    def trans_t(t):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ], np.float32)

    def rot_phi(p):
        return np.array([
            [1,          0,           0, 0],
            [0, np.cos(p), -np.sin(p), 0],
            [0, np.sin(p),  np.cos(p), 0],
            [0,          0,           0, 1],
        ], np.float32)

    def rot_theta(t):
        return np.array([
            [np.cos(t), 0, -np.sin(t), 0],
            [        0, 1,          0, 0],
            [np.sin(t), 0,  np.cos(t), 0],
            [        0, 0,          0, 1],
        ], np.float32)

    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    blender_to_nerf = np.array([
        [-1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1],
    ], np.float32)
    c2w = blender_to_nerf @ c2w
    return c2w


# ─────────────────────────────────────────────────────────────────────────────
# [opt 1] Pre-cache all training rays on device
# ─────────────────────────────────────────────────────────────────────────────
def cache_rays(imgs: Tensor, poses: Tensor, H: int, W: int, focal: float
               ) -> Tuple[Tensor, Tensor, Tensor]:
    print("Pre-caching all training rays on device...")
    all_rays_o, all_rays_d, all_target = [], [], []
    for idx in range(len(imgs)):
        rays_o, rays_d = get_rays(H, W, focal, poses[idx])
        all_rays_o.append(rays_o.reshape(-1, 3))
        all_rays_d.append(rays_d.reshape(-1, 3))
        all_target.append(imgs[idx].reshape(-1, 3))
    rays_o  = torch.cat(all_rays_o)
    rays_d  = torch.cat(all_rays_d)
    targets = torch.cat(all_target)
    print(f"  → cached {rays_o.shape[0]:,} rays on {device}")
    return rays_o, rays_d, targets


# ─────────────────────────────────────────────────────────────────────────────
# Render a full image in chunks (avoids OOM)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def render_image(coarse: nn.Module, fine: nn.Module,
                 H: int, W: int, focal: float, c2w: Tensor) -> np.ndarray:
    coarse.eval(); fine.eval()
    rays_o, rays_d = get_rays(H, W, focal, c2w)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    rgbs = []
    for i in range(0, rays_o.shape[0], cfg.chunk):
        o_chunk = rays_o[i : i + cfg.chunk]
        d_chunk = rays_d[i : i + cfg.chunk]
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, rgb_f = render_rays(coarse, fine, o_chunk, d_chunk)
        rgbs.append(rgb_f.float().cpu())

    rgb = torch.cat(rgbs).reshape(H, W, 3).numpy()
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Video rendering
# ─────────────────────────────────────────────────────────────────────────────
def _save_video(coarse: nn.Module, fine: nn.Module, H: int, W: int, focal: float,
                tag: str = 'latest', n_frames: int = 120, fps: int = 30):
    print(f"Rendering {n_frames} frames...")
    frames = []
    for theta in np.linspace(0, 360, n_frames, endpoint=False):
        c2w   = torch.tensor(pose_spherical(theta, -30.0, 4.0), device=device)
        frame = render_image(coarse, fine, H, W, focal, c2w)
        frames.append(frame)
        if len(frames) % 10 == 0:
            print(f"  {len(frames)}/{n_frames}")

    path = os.path.join(cfg.checkpoint_dir, f'render_{tag}.mp4')
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    print(f"  → saved {path}")


# =============================================================================
# MAIN CODE
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────
def positional_encoding(x: Tensor, L: int) -> Tensor:
    freqs = 2.0 ** torch.arange(L, dtype=torch.float32, device=x.device)
    xf    = x.unsqueeze(-1) * freqs   # [..., 3, L]
    sins  = torch.sin(xf).flatten(-2) # [..., 3*L]
    coss  = torch.cos(xf).flatten(-2) # [..., 3*L]
    return torch.cat([x, sins, coss], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# NeRF MLP
# ─────────────────────────────────────────────────────────────────────────────
class NeRF(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        pos_ch   = 3 * (1 + 2 * cfg.L_pos)   # 63
        dir_ch   = 3 * (1 + 2 * cfg.L_dir)   # 27
        W        = cfg.W
        self.skip = cfg.skip

        self.pts_layers = nn.ModuleList()
        for i in range(cfg.D):
            in_ch = pos_ch if i == 0 else (W + pos_ch if i == cfg.skip + 1 else W)
            self.pts_layers.append(nn.Linear(in_ch, W))

        self.sigma_layer   = nn.Linear(W, 1)
        self.feature_layer = nn.Linear(W, W)
        self.dir_layer     = nn.Linear(W + dir_ch, W // 2)
        self.rgb_layer     = nn.Linear(W // 2, 3)

    def forward(self, pts_enc: Tensor, dir_enc: Tensor) -> Tensor:
        h = pts_enc
        for i, layer in enumerate(self.pts_layers):
            h = F.relu(layer(h))
            if i == self.skip:
                h = torch.cat([h, pts_enc], dim=-1)

        sigma   = self.sigma_layer(h)
        feature = self.feature_layer(h)
        h       = F.relu(self.dir_layer(torch.cat([feature, dir_enc], dim=-1)))
        rgb     = self.rgb_layer(h)

        return torch.cat([rgb, sigma], dim=-1)   # [..., 4]


# ─────────────────────────────────────────────────────────────────────────────
# Volume rendering
# ─────────────────────────────────────────────────────────────────────────────
def volume_render(raw: Tensor, z_vals: Tensor, rays_d: Tensor,
                  white_bkgd: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
    dists = dists * rays_d.norm(dim=-1, keepdim=True)

    rgb   = torch.sigmoid(raw[..., :3])
    sigma = F.relu(raw[..., 3])
    alpha    = 1.0 - torch.exp(-sigma * dists)
    # exclusive cumprod: T_i = prob the ray survived all segments before i
    survival = torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1)
    T        = torch.cumprod(survival, dim=-1)[..., :-1]

    weights = T * alpha
    rgb_map = (weights[..., None] * rgb).sum(-2)
    depth   = (weights * z_vals).sum(-1)
    acc     = weights.sum(-1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc[..., None])

    return rgb_map, depth, weights


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical sampling
# ─────────────────────────────────────────────────────────────────────────────
def sample_pdf(bins: Tensor, weights: Tensor, N: int) -> Tensor:
    weights = weights + 1e-5
    pdf     = weights / weights.sum(-1, keepdim=True)
    cdf     = torch.cumsum(pdf, -1)
    cdf     = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    u     = torch.rand(*cdf.shape[:-1], N, device=weights.device).contiguous()
    idx   = torch.searchsorted(cdf.contiguous(), u, right=True)
    below = (idx - 1).clamp(min=0)
    above = idx.clamp(max=cdf.shape[-1] - 1)
    idx2  = torch.stack([below, above], -1)

    shape  = (*idx2.shape[:-2], -1)
    cdf_g  = torch.gather(cdf,  -1, idx2.reshape(shape)).reshape(*idx2.shape)
    bins_g = torch.gather(bins, -1, idx2.reshape(shape)).reshape(*idx2.shape)

    denom = (cdf_g[..., 1] - cdf_g[..., 0]).clamp(min=1e-5)
    t     = (u - cdf_g[..., 0]) / denom
    return bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])


# ─────────────────────────────────────────────────────────────────────────────
# Render rays (coarse + fine)
# ─────────────────────────────────────────────────────────────────────────────
def render_rays(coarse: nn.Module, fine: nn.Module,
                rays_o: Tensor, rays_d: Tensor) -> Tuple[Tensor, Tensor]:
    N_rays = rays_o.shape[0]

    # coarse
    t = torch.linspace(0., 1., cfg.N_coarse, device=rays_o.device)
    z = cfg.near * (1 - t) + cfg.far * t
    z = z.expand(N_rays, cfg.N_coarse)

    # stratified sampling: perturb each sample within its bin
    mids  = 0.5 * (z[..., 1:] + z[..., :-1])
    lower = torch.cat([z[..., :1], mids], dim=-1)
    upper = torch.cat([mids, z[..., -1:]], dim=-1)
    z     = lower + (upper - lower) * torch.rand_like(z)

    pts   = rays_o[:, None] + rays_d[:, None] * z[..., None]
    dirs  = F.normalize(rays_d, dim=-1)[:, None].expand_as(pts)

    pts_enc  = positional_encoding(pts,  cfg.L_pos)
    dirs_enc = positional_encoding(dirs, cfg.L_dir)

    raw_c = coarse(pts_enc.flatten(0, 1), dirs_enc.flatten(0, 1))
    raw_c = raw_c.reshape(N_rays, cfg.N_coarse, 4)
    rgb_c, _, weights = volume_render(raw_c, z, rays_d, cfg.white_bkgd)

    # fine
    z_mid  = 0.5 * (z[..., 1:] + z[..., :-1])
    z_fine = sample_pdf(z_mid, weights[..., 1:-1].detach(), cfg.N_fine)
    z2     = torch.sort(torch.cat([z, z_fine], -1), -1).values

    N_total = cfg.N_coarse + cfg.N_fine
    pts2    = rays_o[:, None] + rays_d[:, None] * z2[..., None]
    dirs2   = F.normalize(rays_d, dim=-1)[:, None].expand_as(pts2)

    pts_enc2  = positional_encoding(pts2,  cfg.L_pos)
    dirs_enc2 = positional_encoding(dirs2, cfg.L_dir)

    raw_f = fine(pts_enc2.flatten(0, 1), dirs_enc2.flatten(0, 1))
    raw_f = raw_f.reshape(N_rays, N_total, 4)
    rgb_f, _, _ = volume_render(raw_f, z2, rays_d, cfg.white_bkgd)

    return rgb_c, rgb_f


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(resume: Optional[str] = None):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    imgs, poses, H, W, focal = load_blender('train')
    imgs  = imgs.to(device)
    poses = poses.to(device)

    # [opt 1] pre-cache rays
    all_rays_o, all_rays_d, all_targets = cache_rays(imgs, poses, H, W, focal)
    n_rays = all_rays_o.shape[0]

    coarse = NeRF(cfg).to(device)
    fine   = NeRF(cfg).to(device)

    # [opt 3] EMA must be created before torch.compile to keep clean key names
    ema_coarse = EMA(coarse)
    ema_fine   = EMA(fine)

    # [opt 4] torch.compile
    if cfg.use_compile:
        coarse = torch.compile(coarse)
        fine   = torch.compile(fine)

    opt = torch.optim.Adam(
        list(coarse.parameters()) + list(fine.parameters()),
        lr=cfg.lr, betas=(0.9, 0.999),
    )

    start = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        coarse.load_state_dict(ckpt['coarse'])
        fine.load_state_dict(ckpt['fine'])
        ema_coarse.shadow.load_state_dict(ckpt['ema_coarse'])
        ema_fine.shadow.load_state_dict(ckpt['ema_fine'])
        opt.load_state_dict(ckpt['opt'])
        start = ckpt['iter'] + 1
        print(f"Resumed from iter {start}")

    for i in range(start, cfg.n_iters):
        coarse.train(); fine.train()

        # [opt 5] LR warmup + exponential decay
        lr = get_lr(i)
        for pg in opt.param_groups:
            pg['lr'] = lr

        # [opt 1] sample from pre-cached rays
        sel      = torch.randint(n_rays, (cfg.batch_rays,), device=device)
        rays_o_b = all_rays_o[sel]
        rays_d_b = all_rays_d[sel]
        target_b = all_targets[sel]

        # [opt 2] bfloat16 autocast
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            rgb_c, rgb_f = render_rays(coarse, fine, rays_o_b, rays_d_b)
            loss = F.mse_loss(rgb_c.float(), target_b) + F.mse_loss(rgb_f.float(), target_b)

        psnr = -10.0 * torch.log10(F.mse_loss(rgb_f.detach().float(), target_b))

        opt.zero_grad()
        scaler.scale(loss).backward()

        # [opt 5] gradient clipping (disabled when grad_clip=0)
        if cfg.grad_clip > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(
                list(coarse.parameters()) + list(fine.parameters()), cfg.grad_clip
            )
        scaler.step(opt)
        scaler.update()

        # [opt 3] update EMA
        ema_coarse.update(coarse)
        ema_fine.update(fine)

        if i % cfg.i_print == 0:
            print(f"iter {i:06d} | loss {loss.item():.4f} | psnr {psnr.item():.2f} dB | lr {lr:.2e}")

        if i % cfg.i_save == 0 and i > 0:
            ckpt = {
                'iter':       i,
                'coarse':     coarse.state_dict(),
                'fine':       fine.state_dict(),
                'ema_coarse': ema_coarse.shadow.state_dict(),
                'ema_fine':   ema_fine.shadow.state_dict(),
                'opt':        opt.state_dict(),
            }
            path = os.path.join(cfg.checkpoint_dir, f'ckpt_{i:06d}.pt')
            torch.save(ckpt, path)
            torch.save(ckpt, os.path.join(cfg.checkpoint_dir, 'latest.pt'))
            print(f"  → saved {path}")

        if i % cfg.i_video == 0 and i > 0:
            _save_video(ema_coarse.shadow, ema_fine.shadow, H, W, focal, tag=f'{i:06d}', n_frames=24, fps=8)

    # save final checkpoint + video at end of training
    ckpt = {
        'iter':       cfg.n_iters - 1,
        'coarse':     coarse.state_dict(),
        'fine':       fine.state_dict(),
        'ema_coarse': ema_coarse.shadow.state_dict(),
        'ema_fine':   ema_fine.shadow.state_dict(),
        'opt':        opt.state_dict(),
    }
    torch.save(ckpt, os.path.join(cfg.checkpoint_dir, f'ckpt_{cfg.n_iters - 1:06d}.pt'))
    torch.save(ckpt, os.path.join(cfg.checkpoint_dir, 'latest.pt'))
    print(f"  → saved final checkpoint at iter {cfg.n_iters - 1}")
    _save_video(ema_coarse.shadow, ema_fine.shadow, H, W, focal, tag=f'{cfg.n_iters - 1:06d}', n_frames=24, fps=8)


# ─────────────────────────────────────────────────────────────────────────────
# Render MP4 from checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def render(checkpoint: str, n_frames: int = 120, fps: int = 30):
    _, _, H, W, focal = load_blender('test')

    coarse = NeRF(cfg).to(device)
    fine   = NeRF(cfg).to(device)
    ckpt   = torch.load(checkpoint, map_location=device)

    # load EMA weights for best render quality
    coarse.load_state_dict(ckpt['ema_coarse'])
    fine.load_state_dict(ckpt['ema_fine'])

    _save_video(coarse, fine, H, W, focal, tag='render', n_frames=n_frames, fps=fps)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mini-NeRF')
    sub    = parser.add_subparsers(dest='mode')

    p = sub.add_parser('train')
    p.add_argument('--resume', type=str, default=None)

    p = sub.add_parser('render')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--n_frames', type=int,   default=120)
    p.add_argument('--fps',      type=int,   default=30)

    args = parser.parse_args()
    print(f"device: {device}")

    if   args.mode == 'train':  train(resume=args.resume)
    elif args.mode == 'render': render(args.checkpoint, args.n_frames, args.fps)
    else:                       parser.print_help()