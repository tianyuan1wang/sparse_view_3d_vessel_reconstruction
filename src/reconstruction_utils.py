"""Shared projection-layout and reconstruction-evaluation utilities."""

import numpy as np
import torch

from helpers import dice_iou, remove_small_components


def ts_proj_to_vhw(raw_proj_t, n_views):
    if not torch.is_tensor(raw_proj_t):
        raw_proj_t = torch.as_tensor(raw_proj_t)

    if raw_proj_t.ndim == 2:
        if int(n_views) != 1:
            raise ValueError(f'2D projection received but n_views={n_views}')
        return raw_proj_t.unsqueeze(0)

    if raw_proj_t.ndim != 3:
        raise ValueError(f'Unexpected projection ndim={raw_proj_t.ndim}, shape={tuple(raw_proj_t.shape)}')

    s = tuple(int(x) for x in raw_proj_t.shape)

    if s[0] == int(n_views):
        return raw_proj_t.contiguous()
    if s[1] == int(n_views):
        return raw_proj_t.permute(1, 0, 2).contiguous()
    if s[2] == int(n_views):
        return raw_proj_t.permute(2, 0, 1).contiguous()

    if int(n_views) == 1:
        return raw_proj_t.reshape(1, s[0], s[-1]).contiguous()

    raise ValueError(f'Could not interpret projection shape {s} for n_views={n_views}')


def posterior_to_binary_map(prob_xyz, thr=0.5, min_vox_component=50):
    m = np.asarray(prob_xyz) >= float(thr)
    m = remove_small_components(m, min_vox=min_vox_component)
    return m.astype(np.uint8)


def find_best_threshold(prob_xyz, gt_xyz, thresholds=None, min_vox_component=50):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19, dtype=np.float32)

    best = {'thr': None, 'dice': -1.0, 'iou': -1.0, 'mask': None}

    for thr in thresholds:
        m = posterior_to_binary_map(prob_xyz, thr=float(thr), min_vox_component=min_vox_component)
        d, j = dice_iou(m > 0, gt_xyz > 0)
        if d > best['dice']:
            best['thr'] = float(thr)
            best['dice'] = float(d)
            best['iou'] = float(j)
            best['mask'] = m

    return best


def get_angle_dependent_threshold(n_views, default_thr=0.50):
    n_views = int(n_views)
    if n_views <= 1:
        return 0.25
    elif n_views <= 2:
        return 0.35
    elif n_views <= 4:
        return 0.45
    else:
        return float(default_thr)
