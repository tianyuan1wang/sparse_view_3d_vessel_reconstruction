import os

import numpy as np
import scipy.ndimage as ndi
import imageio.v2 as imageio
import matplotlib.pyplot as plt


# ============================================================
# Basic helpers
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def strip_label_suffix(path):
    base = os.path.basename(path)
    if base.endswith('.label.nii.gz'):
        return base[:-13]
    if base.endswith('.label.nii'):
        return base[:-10]
    return os.path.splitext(base)[0]


def save_mask_png(path, mask_hw):
    img = (np.asarray(mask_hw) > 0).astype(np.uint8) * 255
    imageio.imwrite(path, img)


def dice_iou(pred, gt):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    ps = pred.sum()
    gs = gt.sum()

    dice = 2.0 * inter / max(ps + gs, 1)
    iou = inter / max(union, 1)
    return float(dice), float(iou)


# ============================================================
# Connected component filtering
# ============================================================

def remove_small_components(mask_bool, min_vox=50):
    mask_bool = np.asarray(mask_bool, dtype=bool)
    if mask_bool.sum() == 0:
        return mask_bool

    lbl, n = ndi.label(mask_bool)
    if n == 0:
        return mask_bool

    counts = np.bincount(lbl.ravel())
    counts[0] = 0

    keep_ids = np.where(counts >= int(min_vox))[0]
    if len(keep_ids) == 0:
        keep_ids = np.array([int(np.argmax(counts))], dtype=np.int32)

    return np.isin(lbl, keep_ids)


# ============================================================
# Crop + optional downsample
# ============================================================

def crop_and_resample_mask(mask_bool, spacing_xyz, pad_vox=10, max_dim=192):
    """
    Returns
    -------
    mask_small      : cropped / optionally downsampled binary mask
    spacing_eff_xyz : effective voxel size after downsampling
    bbox_min_xyz    : crop start in original voxel coordinates
    bbox_max_xyz    : crop stop (exclusive) in original voxel coordinates
    scale           : isotropic zoom factor applied to cropped volume
    """
    mask_bool = np.asarray(mask_bool, dtype=bool)
    spacing_xyz = np.asarray(spacing_xyz, dtype=np.float32)

    if mask_bool.sum() == 0:
        raise ValueError('Empty mask.')

    coords = np.argwhere(mask_bool)
    mn = coords.min(axis=0)
    mx = coords.max(axis=0) + 1

    mn = np.maximum(mn - int(pad_vox), 0)
    mx = np.minimum(mx + int(pad_vox), np.array(mask_bool.shape))

    crop = mask_bool[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]

    max_current = int(max(crop.shape))
    if max_current <= int(max_dim):
        scale = 1.0
        crop_small = crop.copy()
        spacing_eff = spacing_xyz.copy()
    else:
        scale = float(max_dim) / float(max_current)
        crop_small = ndi.zoom(
            crop.astype(np.float32),
            zoom=(scale, scale, scale),
            order=0
        ) > 0.5
        spacing_eff = spacing_xyz / float(scale)

    return crop_small.astype(bool), spacing_eff.astype(np.float32), mn, mx, float(scale)


# ============================================================
# Slice helpers for visualization
# ============================================================

def _mid_slices_xyz(vol_xyz):
    """
    vol_xyz shape: (X, Y, Z)
    Returns central XY / XZ / YZ slices for visualization.
    """
    vol_xyz = np.asarray(vol_xyz)
    ix = vol_xyz.shape[0] // 2
    iy = vol_xyz.shape[1] // 2
    iz = vol_xyz.shape[2] // 2

    return {
        'XY': vol_xyz[:, :, iz].T,
        'XZ': vol_xyz[:, iy, :].T,
        'YZ': vol_xyz[ix, :, :].T,
    }


def _draw_prob_panel(
    ax,
    img2d,
    title='',
    add_contours=True,
    contour_levels=(0.25, 0.50, 0.75, 0.90),
    cmap='viridis',
    vmin=0.0,
    vmax=1.0,
):
    arr = np.asarray(img2d, dtype=np.float32)
    im = ax.imshow(arr, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    shown_levels = []
    if add_contours:
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        valid_levels = [lv for lv in contour_levels if lo < lv < hi]

        if len(valid_levels) > 0:
            cs = ax.contour(
                arr,
                levels=valid_levels,
                colors='white',
                linewidths=1.0,
            )
            ax.clabel(
                cs,
                inline=True,
                fontsize=8,
                fmt={lv: f'q={lv:.2f}' for lv in valid_levels},
            )
            shown_levels = valid_levels

    ax.set_title(title)
    ax.axis('off')

    if len(shown_levels) > 0:
        txt = 'Contours:\n' + '\n'.join([f'white = q={lv:.2f}' for lv in shown_levels])
        ax.text(
            0.02, 0.98, txt,
            transform=ax.transAxes,
            va='top', ha='left',
            fontsize=7,
            color='white',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.45, edgecolor='none'),
        )

    return im


def _draw_binary_panel(ax, img2d, title=''):
    ax.imshow(img2d, origin='lower', cmap='gray', vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis('off')


# ============================================================
# Angle selection helpers for visualization
# ============================================================

def choose_view_indices(n_views, max_views=8):
    """
    Choose evenly spaced view indices for visualization.
    """
    n_views = int(n_views)
    max_views = int(max_views)

    if n_views <= max_views:
        return list(range(n_views))

    idx = np.linspace(0, n_views - 1, max_views)
    idx = np.unique(np.round(idx).astype(np.int32))
    return idx.tolist()


def wrap_angle_0_2pi(angle_rad):
    angle_rad = np.asarray(angle_rad, dtype=np.float32)
    return np.mod(angle_rad, 2.0 * np.pi)


def select_evenly_spaced_values(vals, n_pick):
    vals = np.asarray(vals, dtype=np.float32).ravel()
    n_pick = int(n_pick)

    if n_pick <= 0 or vals.size == 0:
        return np.zeros((0,), dtype=np.float32)

    if vals.size <= n_pick:
        return vals.copy()

    pos = np.linspace(0, vals.size - 1, n_pick)
    idx = np.round(pos).astype(np.int32)
    idx = np.clip(idx, 0, vals.size - 1)
    idx = np.unique(idx)

    if idx.size < n_pick:
        used = set(idx.tolist())
        extra = [i for i in range(vals.size) if i not in used]
        need = n_pick - idx.size
        idx = np.sort(np.concatenate([idx, np.asarray(extra[:need], dtype=np.int32)]))

    return vals[idx]


def choose_used_unused_angles_for_plot(used_angles_rad, n_used=2, n_unused=2):
    """
    Pick:
      - n_used angles from the actually used acquisition angles
      - n_unused midpoint angles between consecutive used angles

    The midpoint angles are guaranteed to be 'never used'.
    """
    used = wrap_angle_0_2pi(np.asarray(used_angles_rad, dtype=np.float32).ravel())
    used = np.sort(used)

    if used.size == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    used_pick = select_evenly_spaced_values(used, n_used)

    if used.size >= 2:
        mids = []
        for i in range(used.size):
            a0 = float(used[i])
            a1 = float(used[(i + 1) % used.size])
            if i == used.size - 1:
                a1 += 2.0 * np.pi
            mids.append(0.5 * (a0 + a1))
        mids = wrap_angle_0_2pi(np.asarray(mids, dtype=np.float32))
        mids = np.sort(mids)
    else:
        mids = wrap_angle_0_2pi(np.asarray([
            used[0] + 0.5 * np.pi,
            used[0] + 1.0 * np.pi,
        ], dtype=np.float32))
        mids = np.sort(mids)

    unused_pick = select_evenly_spaced_values(mids, n_unused)
    return used_pick.astype(np.float32), unused_pick.astype(np.float32)


# ============================================================
# Entropy helper
# ============================================================

def bernoulli_entropy_map(prob, eps=1e-6, normalize=True):
    prob = np.asarray(prob, dtype=np.float32)
    p = np.clip(prob, eps, 1.0 - eps)
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    if normalize:
        h = h / np.log(2.0)
    return h.astype(np.float32)
