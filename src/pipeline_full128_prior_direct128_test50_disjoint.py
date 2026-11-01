"""
Visual-hull-initialized conditional diffusion for vessel reconstruction.

The CT and vessel target are resized to an exact 128 x 128 x 128 grid before
any X-ray or 2-D mask projection is generated. The same grid is used for the
reconstruction and the full-volume diffusion prior.
"""

from __future__ import annotations

import os
import re
import glob
import json
import csv
import math
import hashlib
import random
import traceback

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import tomosipo as ts
from scipy import ndimage as ndi

import imageio.v2 as imageio

from diffusion_model import DiffusionConfig, GaussianDiffusion, UNet3DDiffusion

from helpers import (
    ensure_dir,
    strip_label_suffix,
    save_mask_png,
    dice_iou,
    remove_small_components,
)
from geometry import (
    build_circular_trajectory_mm,
    build_cone_vec_from_world_sources,
)
from projection import (
    project_label_from_mask,
    generate_multiview_masks_from_label,
)
from reconstruction_utils import (
    ts_proj_to_vhw,
    get_angle_dependent_threshold,
    posterior_to_binary_map,
    find_best_threshold,
)
from visual_hull import reconstruct_visual_hull_from_masks_tomosipo
from mask_preprocessing import (resize_training_mask, validate_training_preparation,
                                observation_target_metadata, validate_observation_target)
from saved_diffusion_split import select_saved_test_paths
from shared_observations import (
    load_shared_observations,
    load_shared_observation_auxiliaries,
    save_shared_observations,
    normalize_shared_obs_mode,
    shared_obs_is_load,
    shared_obs_should_save,
)


from mesh_export import (
    write_ply_mesh,
    write_ply_points,
    mesh_from_binary,
    sample_points_from_mesh,
)
from visualization import (
    save_volume_views_xyz_with_contours,
    plot_case_dice_curve,
    plot_case_probability_evolution_3d,
    plot_diffusion_stage_evolution_3d,
    plot_gt_bayes_and_final_reconstruction_3d,
)

from segmentation_model import (
    DoubleConv2D,
    Down2D,
    Up2D,
    UNet2DSoftSeg,
    load_soft_seg_model,
    normalize_image_01,
    predict_soft_masks_from_xrays,
)
from projection import project_label_from_mask

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Case preparation
# ============================================================

def split_mask_stem(path):
    base = os.path.basename(path)
    if base.endswith(".label.nii.gz"):
        return base[:-13]
    if base.endswith(".label.nii"):
        return base[:-10]
    raise ValueError(f"Unexpected mask filename: {base}")


def split_mask_case_number(path):
    stem = split_mask_stem(path)
    m = re.match(r"^(\d+)(?:_(left|right|single))?$", stem)
    if m is None:
        raise ValueError(f"Unexpected split-mask stem: {stem}")
    return int(m.group(1))


def split_mask_sort_key(path):
    stem = split_mask_stem(path)
    m = re.match(r"^(\d+)(?:_(left|right|single))?$", stem)
    if m is None:
        return (10**12, "zzz")
    num = int(m.group(1))
    side = m.group(2) or ""
    return (num, side)

def crop_and_resample_image(img_xyz, bbox_min_xyz, bbox_max_xyz, scale):
    import scipy.ndimage as ndi

    x0, y0, z0 = [int(v) for v in bbox_min_xyz]
    x1, y1, z1 = [int(v) for v in bbox_max_xyz]

    crop = np.asarray(img_xyz[x0:x1, y0:y1, z0:z1], dtype=np.float32)

    if abs(float(scale) - 1.0) > 1e-8:
        crop = ndi.zoom(crop, zoom=(scale, scale, scale), order=1)

    return crop.astype(np.float32)


def _resize_volume_to_exact_shape(
    arr_xyz,
    out_shape_xyz,
    mode="trilinear",
):
    """
    Resize a 3-D volume to an exact shape before any projection is generated.
    """
    arr = np.asarray(arr_xyz, dtype=np.float32)
    out_shape = tuple(int(v) for v in out_shape_xyz)

    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape={arr.shape}")
    if len(out_shape) != 3 or any(v <= 0 for v in out_shape):
        raise ValueError(f"Invalid exact output shape: {out_shape}")
    if tuple(arr.shape) == out_shape:
        return arr.copy()

    x = torch.from_numpy(arr)[None, None]
    mode_name = str(mode).lower()

    if mode_name == "nearest":
        y = F.interpolate(x, size=out_shape, mode="nearest")
    elif mode_name in {"linear", "trilinear"}:
        y = F.interpolate(
            x,
            size=out_shape,
            mode="trilinear",
            align_corners=False,
        )
    else:
        raise ValueError(
            f"Unknown resize mode={mode}. Use nearest or trilinear."
        )

    return y[0, 0].cpu().numpy().astype(np.float32)


def _resampled_affine_exact_grid(
    original_affine,
    bbox_min_xyz,
    source_crop_shape_xyz,
    target_shape_xyz,
):
    """
    Construct an affine consistent with align_corners=False interpolation.
    """
    aff = np.asarray(original_affine, dtype=np.float64)
    bbox_min = np.asarray(bbox_min_xyz, dtype=np.float64)
    source_shape = np.asarray(source_crop_shape_xyz, dtype=np.float64)
    target_shape = np.asarray(target_shape_xyz, dtype=np.float64)

    zoom = target_shape / source_shape
    source_step_per_target = 1.0 / zoom
    source_origin = bbox_min + (0.5 / zoom) - 0.5

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = aff[:3, :3] @ np.diag(source_step_per_target)
    out[:3, 3] = aff[:3, :3] @ source_origin + aff[:3, 3]
    return out.astype(np.float32)


def prepare_case(
    lbl_path,
    img_path=None,
    min_vox_component=50,
    bbox_pad=0,
    max_dim=9999,
    crop_bbox_xyz=None,
    exact_grid_shape_xyz=None,
    exact_mask_resample_mode="nearest",
    exact_mask_eval_threshold=0.5,
    mask_preprocessing="legacy",
):
    """
    Prepare one case.

    If exact_grid_shape_xyz is supplied, the cropped CT and vessel target are
    resized to that exact shape BEFORE any 2-D projection is generated.
    """
    import scipy.ndimage as ndi

    case_id = strip_label_suffix(lbl_path)

    if mask_preprocessing not in {"legacy", "training_matched"}:
        raise ValueError(f"Unknown mask_preprocessing={mask_preprocessing!r}")
    training_matched = mask_preprocessing == "training_matched"
    if training_matched:
        validate_training_preparation(
            crop_bbox_xyz=crop_bbox_xyz, bbox_pad=bbox_pad,
            exact_grid_shape_xyz=exact_grid_shape_xyz,
            exact_mask_resample_mode=exact_mask_resample_mode,
            exact_mask_eval_threshold=exact_mask_eval_threshold,
            min_vox_component=min_vox_component,
        )
    nii_lbl = nib.load(lbl_path)
    mask_full_xyz = (nii_lbl.get_fdata().astype(np.float32) > 0.5) if training_matched else (nii_lbl.get_fdata() > 0)

    sx, sy, sz = [float(x) for x in nii_lbl.header.get_zooms()[:3]]
    spacing_xyz = np.array([sx, sy, sz], dtype=np.float32)

    if not training_matched:
        mask_full_xyz = remove_small_components(mask_full_xyz, min_vox=min_vox_component)
    if mask_full_xyz.sum() == 0:
        raise ValueError("Empty mask after native target preparation.")

    mask_full_xyz = mask_full_xyz.astype(np.uint8)

    img_full_xyz = None
    original_affine = nii_lbl.affine.astype(np.float32)

    if img_path is not None:
        nii_img = nib.load(img_path)
        img_full_xyz = nii_img.get_fdata().astype(np.float32)
        original_affine = nii_img.affine.astype(np.float32)

        if img_full_xyz.shape != mask_full_xyz.shape:
            raise ValueError(
                f"Image/label shape mismatch: img={img_full_xyz.shape}, "
                f"mask={mask_full_xyz.shape}"
            )

    full_shape_xyz = np.array(mask_full_xyz.shape, dtype=np.int32)

    if crop_bbox_xyz is None:
        bbox_min_xyz = np.array([0, 0, 0], dtype=np.int32)
        bbox_max_xyz = full_shape_xyz - 1
    else:
        bbox_min_xyz = np.array(crop_bbox_xyz[0], dtype=np.int32)
        bbox_max_xyz = np.array(crop_bbox_xyz[1], dtype=np.int32)

        bbox_min_xyz = np.maximum(
            bbox_min_xyz - int(bbox_pad),
            0,
        )
        bbox_max_xyz = np.minimum(
            bbox_max_xyz + int(bbox_pad),
            full_shape_xyz - 1,
        )

    x0, y0, z0 = [int(v) for v in bbox_min_xyz]
    x1, y1, z1 = [int(v) for v in bbox_max_xyz]

    mask_crop_xyz = mask_full_xyz[
        x0:x1 + 1,
        y0:y1 + 1,
        z0:z1 + 1,
    ].astype(np.uint8)

    if img_full_xyz is not None:
        img_crop_xyz = img_full_xyz[
            x0:x1 + 1,
            y0:y1 + 1,
            z0:z1 + 1,
        ].astype(np.float32)
    else:
        img_crop_xyz = None

    source_crop_shape_xyz = np.array(
        mask_crop_xyz.shape,
        dtype=np.int32,
    )
    source_physical_size_mm_xyz = (
        source_crop_shape_xyz.astype(np.float64)
        * spacing_xyz.astype(np.float64)
    )

    if exact_grid_shape_xyz is not None:
        target_shape_xyz = tuple(int(v) for v in exact_grid_shape_xyz)
        if len(target_shape_xyz) != 3 or any(v <= 0 for v in target_shape_xyz):
            raise ValueError(
                f"Invalid exact_grid_shape_xyz={exact_grid_shape_xyz}"
            )

        mode = str(exact_mask_resample_mode).lower()
        if training_matched:
            mask_physics_xyz = resize_training_mask(mask_crop_xyz, out_shape=target_shape_xyz)
        elif mode == "nearest":
            mask_physics_xyz = _resize_volume_to_exact_shape(
                mask_crop_xyz.astype(np.float32),
                target_shape_xyz,
                mode="nearest",
            )
            mask_physics_xyz = (
                mask_physics_xyz >= 0.5
            ).astype(np.float32)
        elif mode == "soft":
            mask_physics_xyz = _resize_volume_to_exact_shape(
                mask_crop_xyz.astype(np.float32),
                target_shape_xyz,
                mode="trilinear",
            )
            mask_physics_xyz = np.clip(
                mask_physics_xyz,
                0.0,
                1.0,
            ).astype(np.float32)
        else:
            raise ValueError(
                "exact_mask_resample_mode must be nearest or soft"
            )

        mask_xyz = (
            mask_physics_xyz >= float(exact_mask_eval_threshold)
        ).astype(np.uint8)

        if img_crop_xyz is not None:
            img_xyz = _resize_volume_to_exact_shape(
                img_crop_xyz,
                target_shape_xyz,
                mode="trilinear",
            )
        else:
            img_xyz = None

        spacing_eff_xyz = (
            source_physical_size_mm_xyz
            / np.asarray(target_shape_xyz, dtype=np.float64)
        ).astype(np.float32)

        scale_xyz = (
            np.asarray(target_shape_xyz, dtype=np.float64)
            / source_crop_shape_xyz.astype(np.float64)
        ).astype(np.float32)
        scale = float(np.mean(scale_xyz))

        aff_local = _resampled_affine_exact_grid(
            original_affine=original_affine,
            bbox_min_xyz=bbox_min_xyz,
            source_crop_shape_xyz=source_crop_shape_xyz,
            target_shape_xyz=target_shape_xyz,
        )
        exact_grid_applied = True
    else:
        mask_xyz = mask_crop_xyz
        img_xyz = img_crop_xyz

        longest = int(max(mask_xyz.shape))
        scale = min(1.0, float(max_dim) / float(longest))
        scale_xyz = np.array([scale, scale, scale], dtype=np.float32)

        if abs(scale - 1.0) > 1e-8:
            mask_xyz = ndi.zoom(
                mask_xyz.astype(np.float32),
                zoom=(scale, scale, scale),
                order=0,
            )
            mask_xyz = (mask_xyz > 0.5).astype(np.uint8)

            if img_xyz is not None:
                img_xyz = ndi.zoom(
                    img_xyz.astype(np.float32),
                    zoom=(scale, scale, scale),
                    order=1,
                ).astype(np.float32)

        mask_physics_xyz = mask_xyz.astype(np.float32)
        spacing_eff_xyz = spacing_xyz / float(scale)
        aff_local = original_affine
        exact_grid_applied = False

    Nx, Ny, Nz = [int(v) for v in mask_xyz.shape]

    full_center_idx = 0.5 * (
        full_shape_xyz.astype(np.float64) - 1.0
    )
    crop_center_idx = 0.5 * (
        bbox_min_xyz.astype(np.float64)
        + bbox_max_xyz.astype(np.float64)
    )
    center_mm = (
        crop_center_idx - full_center_idx
    ) * spacing_xyz.astype(np.float64)

    volume_size_mm_xyz = (
        np.asarray([Nx, Ny, Nz], dtype=np.float64)
        * spacing_eff_xyz.astype(np.float64)
    )

    vg = ts.volume(
        shape=(Nx, Ny, Nz),
        pos=np.asarray(center_mm, dtype=np.float64),
        size=volume_size_mm_xyz,
    )

    prep = {
        "case_id": case_id,
        "label_path": lbl_path,
        "img_path": img_path,
        "img_xyz": img_xyz,
        "mask_xyz": mask_xyz,
        "mask_zyx": mask_xyz.copy(),
        "mask_physics_xyz": mask_physics_xyz.astype(np.float32),
        "spacing_eff_xyz": spacing_eff_xyz.astype(np.float32),
        "spacing_zyx": spacing_eff_xyz.astype(np.float32),
        "bbox_min_xyz": bbox_min_xyz.astype(np.int32),
        "bbox_max_xyz": bbox_max_xyz.astype(np.int32),
        "scale": float(scale),
        "scale_xyz": scale_xyz.astype(np.float32),
        "vg": vg,
        "aff_local": aff_local,
        "full_shape_xyz": full_shape_xyz.astype(np.int32),
        "spacing_full_xyz": spacing_xyz.astype(np.float32),
        "source_crop_shape_xyz": source_crop_shape_xyz.astype(np.int32),
        "bbox_pad": int(bbox_pad),
        "mask_preprocessing": str(mask_preprocessing),
        "prepare_min_vox_component": None if training_matched else int(min_vox_component),
        "reconstruction_output_min_vox_component": int(min_vox_component),
        "volume_size_mm_xyz": volume_size_mm_xyz.astype(np.float32),
        "exact_grid_shape_xyz": (
            None
            if exact_grid_shape_xyz is None
            else [int(v) for v in exact_grid_shape_xyz]
        ),
        "exact_grid_applied": bool(exact_grid_applied),
        "exact_mask_resample_mode": str(exact_mask_resample_mode),
        "measurements_generated_after_exact_resampling": bool(
            exact_grid_applied
        ),
    }
    return prep



# ============================================================
# Synthetic X-ray helpers (segmentation inference is in segmentation_model.py)
# ============================================================

def hu_to_01_window(hu, hu_min=-1000.0, hu_max=3000.0, dtype=np.float32):
    hu = np.asarray(hu, dtype=dtype)
    hu = np.clip(hu, hu_min, hu_max)
    return ((hu - hu_min) / (hu_max - hu_min + 1e-12)).astype(np.float32)


def save_gray_png(path: str, img_hw: np.ndarray):
    arr = normalize_image_01(img_hw)
    imageio.imwrite(path, (255.0 * arr).astype(np.uint8))


def save_softmask_png(path: str, prob_hw: np.ndarray):
    arr = np.clip(np.asarray(prob_hw, dtype=np.float32), 0.0, 1.0)
    imageio.imwrite(path, (255.0 * arr).astype(np.uint8))


def build_confidence_from_soft_mask(prob_hw: np.ndarray, min_w: float = 0.05) -> np.ndarray:
    p = np.clip(np.asarray(prob_hw, dtype=np.float32), 0.0, 1.0)
    # high confidence near 0/1, low near 0.5
    w = np.abs(2.0 * p - 1.0)
    w = np.clip(w, float(min_w), 1.0)
    return w.astype(np.float32)


def apply_xray_noise_from_line_integrals(line_hw_t: torch.Tensor, photon_count: float, clamp_min: float = 1.0):
    I = torch.exp(-line_hw_t)
    counts = torch.poisson(I * float(photon_count))
    counts = torch.clamp(counts, min=float(clamp_min))
    I_noisy = counts / float(photon_count)
    return I_noisy

@torch.no_grad()
def simulate_multiview_xrays_from_ct(
    img_xyz: np.ndarray,
    mask_xyz: np.ndarray,
    vg,
    pg,
    device="cpu",
    hu_min=-1000.0,
    hu_max=3000.0,
    hu_scale=5e-3,
    delta_mu=0.008,
    photon_count=5e4,
):
    img_xyz = np.asarray(img_xyz, dtype=np.float32)
    # Preserve a soft partial-volume vessel field when direct-grid preparation
    # used exact_mask_resample_mode="soft"; binary inputs remain unchanged.
    mask_xyz = np.clip(
        np.asarray(mask_xyz, dtype=np.float32),
        0.0,
        1.0,
    )

    bg01_xyz = hu_to_01_window(img_xyz, hu_min=hu_min, hu_max=hu_max).astype(np.float32)
    bg_mu_xyz = bg01_xyz * float(hu_scale)
    mu_total_xyz = bg_mu_xyz + float(delta_mu) * mask_xyz

    mu_t = torch.from_numpy(mu_total_xyz).to(device=device, dtype=torch.float32)

    xrays = []
    for v in range(len(pg)):
        A_k = ts.operator(vg, pg[v])
        p = A_k(mu_t)[:, 0, :].contiguous()

        I_noisy = apply_xray_noise_from_line_integrals(
            p.clamp_min(0.0),
            photon_count=float(photon_count),
            clamp_min=1.0,
        )
        xray_np = I_noisy.detach().cpu().numpy().astype(np.float32)
        xray_np = normalize_image_01(xray_np)
        xrays.append(xray_np)

    return xrays


@torch.no_grad()
def simulate_multiview_xrays_from_raw_ct(
    img_path: str,
    lbl_path: str,
    pg,
    device="cpu",
    hu_min=-1000.0,
    hu_max=3000.0,
    hu_scale=5e-3,
    delta_mu=0.008,
    photon_count=5e4,
):
    nii_img = nib.load(img_path)
    vol_hu = nii_img.get_fdata().astype(np.float32)

    nii_lbl = nib.load(lbl_path)
    mask_bool = (nii_lbl.get_fdata() > 0).astype(np.float32)

    sx, sy, sz = [float(x) for x in nii_img.header.get_zooms()[:3]]
    Nx, Ny, Nz = vol_hu.shape
    vg_raw = ts.volume(shape=(Nx, Ny, Nz), size=(Nx * sx, Ny * sy, Nz * sz))

    mu01_xyz = hu_to_01_window(
        vol_hu,
        hu_min=float(hu_min),
        hu_max=float(hu_max),
    ).astype(np.float32)
    mu_xyz = mu01_xyz * float(hu_scale)

    mu_t = torch.from_numpy(mu_xyz).to(device=device, dtype=torch.float32)
    mask_t = torch.from_numpy(mask_bool.astype(np.float32)).to(device=device, dtype=torch.float32)

    xrays = []
    for v in range(len(pg)):
        A_k = ts.operator(vg_raw, pg[v])

        p_mu = A_k(mu_t)[:, 0, :].contiguous()
        p_mask = A_k(mask_t)[:, 0, :].contiguous()

        p = p_mu + float(delta_mu) * p_mask

        I_noisy = apply_xray_noise_from_line_integrals(
            p.clamp_min(0.0),
            photon_count=float(photon_count),
            clamp_min=1.0,
        )
        xray_np = I_noisy.detach().cpu().numpy().astype(np.float32)
        xray_np = normalize_image_01(xray_np)
        xrays.append(xray_np)

    return xrays

# ============================================================
# Patch diffusion prior
# ============================================================

def center_pad_to_shape(arr: np.ndarray, shape_zyx, value: float = 0.0) -> np.ndarray:
    out = np.full(shape_zyx, value, dtype=arr.dtype)
    src_shape = np.array(arr.shape)
    dst_shape = np.array(shape_zyx)
    copy_shape = np.minimum(src_shape, dst_shape)

    src_start = ((src_shape - copy_shape) // 2).astype(int)
    dst_start = ((dst_shape - copy_shape) // 2).astype(int)

    src_end = src_start + copy_shape
    dst_end = dst_start + copy_shape

    out[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = arr[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]
    return out


def compute_sliding_starts(size: int, patch: int, stride: int):
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def center_crop_indices(full_patch, keep_patch):
    pd, ph, pw = full_patch
    kd, kh, kw = keep_patch
    dz = (pd - kd) // 2
    dy = (ph - kh) // 2
    dx = (pw - kw) // 2
    return slice(dz, dz + kd), slice(dy, dy + kh), slice(dx, dx + kw)


def make_weight_window(shape_zyx, eps: float = 1e-3) -> np.ndarray:
    z, y, x = [np.linspace(-1.0, 1.0, s, dtype=np.float32) for s in shape_zyx]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    w = np.clip((1.0 - np.abs(zz)) * (1.0 - np.abs(yy)) * (1.0 - np.abs(xx)), eps, None)
    return w.astype(np.float32)


def prob_to_diffusion_space(prob_zyx: np.ndarray) -> np.ndarray:
    return np.clip(prob_zyx, 0.0, 1.0).astype(np.float32) * 2.0 - 1.0


def diffusion_to_prob(x_zyx: np.ndarray) -> np.ndarray:
    return np.clip((x_zyx + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)


# ============================================================
# DDPM helpers + measurement-guided correction
# ============================================================

def ddpm_predict_x0_from_eps(
    diffusion: GaussianDiffusion,
    x_t: torch.Tensor,
    pred_eps: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    sqrt_ab = diffusion.sqrt_alpha_bars[t][:, None, None, None, None]
    sqrt_1mab = diffusion.sqrt_one_minus_alpha_bars[t][:, None, None, None, None]
    x0_hat = (x_t - sqrt_1mab * pred_eps) / torch.clamp(sqrt_ab, min=1e-8)
    return x0_hat.clamp(-1.0, 1.0)


def ddpm_sample_step_from_x0(
    diffusion: GaussianDiffusion,
    x_t: torch.Tensor,
    x0_hat: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    betas_t = diffusion.betas[t][:, None, None, None, None]
    alpha_t = diffusion.alphas[t][:, None, None, None, None]
    alpha_bar_t = diffusion.alpha_bars[t][:, None, None, None, None]
    alpha_bar_prev_t = diffusion.alpha_bars_prev[t][:, None, None, None, None]

    coef1 = torch.sqrt(alpha_bar_prev_t) * betas_t / torch.clamp(1.0 - alpha_bar_t, min=1e-8)
    coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t) / torch.clamp(1.0 - alpha_bar_t, min=1e-8)
    
    mean = coef1 * x0_hat + coef2 * x_t

    nonzero_mask = (t > 0).float()[:, None, None, None, None]
    noise = torch.randn_like(x_t)
    var = diffusion.posterior_var[t][:, None, None, None, None]
    x_prev = mean + nonzero_mask * torch.sqrt(torch.clamp(var, min=1e-20)) * noise
    return x_prev


def ddim_sample_step_from_x0(
    diffusion: GaussianDiffusion,
    x_t: torch.Tensor,
    x0_hat: torch.Tensor,
    pred_eps: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    t_int = int(t.item())
    if t_int <= 0:
        return x0_hat.clamp(-1.0, 1.0)
    ab_prev = diffusion.alpha_bars[t_int - 1]
    return torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1.0 - ab_prev) * pred_eps


def ddpm_posterior_mean_from_x0(
    diffusion: GaussianDiffusion,
    x_t: torch.Tensor,
    x0_hat: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    betas_t = diffusion.betas[t][:, None, None, None, None]
    alpha_t = diffusion.alphas[t][:, None, None, None, None]
    alpha_bar_t = diffusion.alpha_bars[t][:, None, None, None, None]
    alpha_bar_prev_t = diffusion.alpha_bars_prev[t][:, None, None, None, None]

    coef1 = torch.sqrt(alpha_bar_prev_t) * betas_t / torch.clamp(1.0 - alpha_bar_t, min=1e-8)
    coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t) / torch.clamp(1.0 - alpha_bar_t, min=1e-8)
    mean = coef1 * x0_hat + coef2 * x_t
    return mean


def center_pad_tensor_5d(x: torch.Tensor, out_shape_dhw, value: float = 0.0) -> torch.Tensor:
    assert x.ndim == 5 and x.shape[0] == 1 and x.shape[1] == 1
    _, _, D, H, W = x.shape
    DO, HO, WO = out_shape_dhw

    out = x.new_full((1, 1, DO, HO, WO), fill_value=value)

    src_shape = np.array([D, H, W], dtype=np.int64)
    dst_shape = np.array([DO, HO, WO], dtype=np.int64)
    copy_shape = np.minimum(src_shape, dst_shape)

    src_start = ((src_shape - copy_shape) // 2).astype(int)
    dst_start = ((dst_shape - copy_shape) // 2).astype(int)

    src_end = src_start + copy_shape
    dst_end = dst_start + copy_shape

    out[
        :,
        :,
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = x[
        :,
        :,
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]
    return out


def _ensure_tensor_on_device(x, device):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, device=device, dtype=torch.float32)



def compute_tv_gradient_from_x0hat_zyx(
    x0_hat_zyx: torch.Tensor,
    device,
    tv_mode: str = "anisotropic",
    tv_eps: float = 1e-4,
):
    with torch.enable_grad():
        x = x0_hat_zyx.detach().clone().to(device=device, dtype=torch.float32)
        x.requires_grad_(True)

        prob = ((x + 1.0) * 0.5).clamp(0.0, 1.0)

        dz = prob[:, :, 1:, :, :] - prob[:, :, :-1, :, :]
        dy = prob[:, :, :, 1:, :] - prob[:, :, :, :-1, :]
        dx = prob[:, :, :, :, 1:] - prob[:, :, :, :, :-1]

        if tv_mode == "anisotropic":
            tv_loss = dz.abs().mean() + dy.abs().mean() + dx.abs().mean()
        elif tv_mode == "isotropic":
            dzc = dz[:, :, :, :-1, :-1]
            dyc = dy[:, :, :-1, :, :-1]
            dxc = dx[:, :, :-1, :-1, :]
            tv_loss = torch.sqrt(dzc * dzc + dyc * dyc + dxc * dxc + float(tv_eps)).mean()
        else:
            raise ValueError(f"Unknown tv_mode: {tv_mode}")

        tv_loss.backward()
        grad_x = x.grad.detach()

    aux = {
        "tv_loss": float(tv_loss.detach().item()),
        "tv_grad_mean_abs": float(grad_x.abs().mean().item()),
        "tv_grad_max_abs": float(grad_x.abs().max().item()),
    }
    return grad_x, aux



@torch.no_grad()
def compute_measurement_gradient_from_x0hat_zyx(
    x0_hat_zyx: torch.Tensor,
    masks_2d,
    A_op,
    sil_scale_mm: float,
    device,
    loss_type: str = "bce",
    eps: float = 1e-6,
    mask_weights_2d=None,
    alpha_mode: str = "prob",
):
    def _ensure_tensor(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.float32)
        return torch.as_tensor(x, device=device, dtype=torch.float32)

    def _find_perm(src_shape, dst_shape):
        import itertools
        for p in itertools.permutations(range(len(src_shape))):
            if tuple(src_shape[i] for i in p) == tuple(dst_shape):
                return p
        return None

    assert x0_hat_zyx.ndim == 5
    assert x0_hat_zyx.shape[0] == 1
    assert x0_hat_zyx.shape[1] == 1

    prob_zyx = ((x0_hat_zyx + 1.0) * 0.5).clamp(eps, 1.0 - eps)[0, 0]

    if alpha_mode == "prob":
        alpha_zyx = prob_zyx
        d_alpha_d_prob = torch.ones_like(prob_zyx)
    elif alpha_mode == "neglog":
        alpha_zyx = -torch.log(torch.clamp(1.0 - prob_zyx, min=eps))
        d_alpha_d_prob = 1.0 / torch.clamp(1.0 - prob_zyx, min=eps)
    else:
        raise ValueError(f"Unknown alpha_mode: {alpha_mode}")

    line_raw = A_op(alpha_zyx)
    line_raw = _ensure_tensor(line_raw)

    y_vhw = torch.as_tensor(
        np.stack(masks_2d, axis=0).astype(np.float32),
        device=device,
        dtype=torch.float32,
    )

    if mask_weights_2d is None:
        w_vhw = torch.ones_like(y_vhw)
    else:
        w_vhw = torch.as_tensor(
            np.stack(mask_weights_2d, axis=0).astype(np.float32),
            device=device,
            dtype=torch.float32,
        )
        if tuple(w_vhw.shape) != tuple(y_vhw.shape):
            raise RuntimeError(
                f"mask_weights_2d shape {tuple(w_vhw.shape)} does not match "
                f"masks shape {tuple(y_vhw.shape)}"
            )

    forward_perm = None
    if tuple(line_raw.shape) != tuple(y_vhw.shape):
        forward_perm = _find_perm(tuple(line_raw.shape), tuple(y_vhw.shape))
        if forward_perm is None:
            raise RuntimeError(
                f"Cannot match projector output shape {tuple(line_raw.shape)} "
                f"to mask shape {tuple(y_vhw.shape)}"
            )
        line_vhw = line_raw.permute(*forward_perm).contiguous()
    else:
        line_vhw = line_raw

    q_fg_vhw = 1.0 - torch.exp(-line_vhw / float(sil_scale_mm))
    norm = torch.clamp(w_vhw.sum(), min=1.0)

    if loss_type == "bce":
        q_safe = q_fg_vhw.clamp(eps, 1.0 - eps)
        dL_dline_vhw = w_vhw * (q_safe - y_vhw) / (
            float(sil_scale_mm) * q_safe
        )
    elif loss_type == "l2":
        dq_dline = torch.exp(-line_vhw / float(sil_scale_mm)) / float(sil_scale_mm)
        dL_dline_vhw = w_vhw * (q_fg_vhw - y_vhw) * dq_dline
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    dL_dline_vhw = dL_dline_vhw / norm

    if forward_perm is not None:
        inv_perm = [0] * len(forward_perm)
        for i, p in enumerate(forward_perm):
            inv_perm[p] = i
        dL_dline_native = dL_dline_vhw.permute(*inv_perm).contiguous()
    else:
        dL_dline_native = dL_dline_vhw

    grad_alpha_zyx = A_op.T(dL_dline_native)
    grad_alpha_zyx = _ensure_tensor(grad_alpha_zyx)

    grad_prob_zyx = grad_alpha_zyx * d_alpha_d_prob

    # p = (x0 + 1) / 2
    grad_x0_zyx = 0.5 * grad_prob_zyx
    grad_x0_zyx = grad_x0_zyx[None, None]

    aux = {
        "alpha_mode": alpha_mode,
        "line_raw_shape": list(line_raw.shape),
        "line_used_shape": list(line_vhw.shape),
        "mask_shape": list(y_vhw.shape),
        "forward_perm": None if forward_perm is None else list(forward_perm),
        "pred_mask_mean": float(q_fg_vhw.mean().item()),
        "pred_mask_max": float(q_fg_vhw.max().item()),
        "line_mean": float(line_vhw.mean().item()),
        "line_max": float(line_vhw.max().item()),
        "prob_mean": float(prob_zyx.mean().item()),
        "prob_max": float(prob_zyx.max().item()),
        "grad_mean_abs": float(grad_x0_zyx.abs().mean().item()),
        "grad_max_abs": float(grad_x0_zyx.abs().max().item()),
        "weight_mean": float(w_vhw.mean().item()),
        "weight_min": float(w_vhw.min().item()),
        "weight_max": float(w_vhw.max().item()),
    }

    return grad_x0_zyx, aux

class PatchDiffusionPrior:
    """
    Unified diffusion prior loader.

    Supports two checkpoint types:
      1) old patch checkpoint with cfg["patch_size"]
         -> sliding-window patch denoiser
      2) new full-volume 128^3 checkpoint with cfg["volume_shape"]
         -> downsample current volume to 128^3, denoise/sample there,
            then upsample predictions back to reconstruction grid.
    """
    def __init__(self, ckpt_path: str, device="cuda"):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        cfg = ckpt["cfg"]
        self.ckpt_path = ckpt_path
        self.cfg = cfg
        self.device = torch.device(device)

        self.base_ch = int(cfg["base_ch"])
        self.time_dim = int(cfg["time_dim"])

        if "patch_size" in cfg:
            self.prior_type = "patch"
            self.patch_size = tuple(int(v) for v in cfg["patch_size"])
            self.volume_shape = None
        elif "volume_shape" in cfg:
            self.prior_type = "full_volume"
            self.volume_shape = tuple(int(v) for v in cfg["volume_shape"])
            self.patch_size = self.volume_shape
        else:
            raise KeyError(
                "Diffusion checkpoint cfg must contain either 'patch_size' or 'volume_shape'. "
                f"Got keys: {sorted(cfg.keys())}"
            )

        self.model = UNet3DDiffusion(
            in_ch=1,
            base_ch=self.base_ch,
            time_dim=self.time_dim,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()

        self.diffusion = GaussianDiffusion(
            DiffusionConfig(
                num_steps=cfg["num_steps"],
                beta_start=cfg["beta_start"],
                beta_end=cfg["beta_end"],
            )
        ).to(self.device)
        self.num_steps = int(cfg["num_steps"])

        print(
            f"  [diffusion prior] loaded {self.prior_type} prior from {ckpt_path} | "
            f"base_ch={self.base_ch}, steps={self.num_steps}, "
            f"patch_size={self.patch_size}, volume_shape={self.volume_shape}"
        )

    def _resize_np_zyx(self, arr_zyx: np.ndarray, out_shape_zyx, order: int = 1) -> np.ndarray:
        arr_zyx = np.asarray(arr_zyx, dtype=np.float32)
        out_shape_zyx = tuple(int(v) for v in out_shape_zyx)
        if tuple(arr_zyx.shape) == out_shape_zyx:
            return arr_zyx.astype(np.float32)
        zoom = [out_shape_zyx[i] / float(arr_zyx.shape[i]) for i in range(3)]
        out = ndi.zoom(arr_zyx, zoom=zoom, order=order).astype(np.float32)
        # ndi.zoom can be off by one due to rounding; fix by center pad/crop.
        if tuple(out.shape) != out_shape_zyx:
            out = center_pad_to_shape(out, out_shape_zyx, value=0.0)
        return out.astype(np.float32)

    def _resize_tensor_5d(self, x: torch.Tensor, out_shape_zyx, mode: str = "trilinear") -> torch.Tensor:
        out_shape_zyx = tuple(int(v) for v in out_shape_zyx)
        if tuple(x.shape[2:]) == out_shape_zyx:
            return x
        return F.interpolate(x, size=out_shape_zyx, mode=mode, align_corners=False)

    @torch.no_grad()
    def denoise_patch_from_prob(self, prob_patch_zyx: np.ndarray, t_denoise: int = 80) -> np.ndarray:
        """Denoise one patch or, for full-volume prior, denoise a resized whole volume."""
        prob_patch_zyx = np.clip(prob_patch_zyx, 0.0, 1.0).astype(np.float32)

        original_shape = tuple(prob_patch_zyx.shape)
        if self.prior_type == "full_volume":
            prob_in_zyx = self._resize_np_zyx(prob_patch_zyx, self.volume_shape, order=1)
        else:
            prob_in_zyx = prob_patch_zyx

        x0 = torch.from_numpy(prob_to_diffusion_space(prob_in_zyx)[None, None]).to(self.device)

        t_denoise = int(np.clip(t_denoise, 0, self.num_steps - 1))
        t = torch.full((1,), t_denoise, device=self.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x = self.diffusion.q_sample(x0, t, noise)

        for ti in reversed(range(t_denoise + 1)):
            tt = torch.full((1,), ti, device=self.device, dtype=torch.long)
            pred_eps = self.model(x, tt)
            x0_hat = ddpm_predict_x0_from_eps(self.diffusion, x, pred_eps, tt)
            x = ddpm_sample_step_from_x0(self.diffusion, x, x0_hat, tt)

        prob_out = diffusion_to_prob(x[0, 0].clamp(-1.0, 1.0).detach().cpu().numpy())

        if self.prior_type == "full_volume" and tuple(prob_out.shape) != original_shape:
            prob_out = self._resize_np_zyx(prob_out, original_shape, order=1)

        return np.clip(prob_out, 0.0, 1.0).astype(np.float32)

    @torch.no_grad()
    def denoise_full_from_prob(self, prob_zyx: np.ndarray, t_denoise: int = 80) -> np.ndarray:
        """Full-volume denoising interface used by PnP/post-denoise cleanup."""
        return self.denoise_patch_from_prob(prob_zyx, t_denoise=t_denoise)

    @torch.no_grad()
    def predict_eps_full(
        self,
        x_zyx: torch.Tensor,
        t: torch.Tensor,
        stride_zyx=None,
        center_keep_size_zyx=None,
    ) -> torch.Tensor:
        assert x_zyx.ndim == 5 and x_zyx.shape[0] == 1 and x_zyx.shape[1] == 1

        if self.prior_type == "full_volume":
            # Downsample the current reconstruction state to the training grid,
            # predict epsilon there, then upsample epsilon back to the current grid.
            orig_shape = tuple(int(v) for v in x_zyx.shape[2:])
            x_low = self._resize_tensor_5d(x_zyx, self.volume_shape, mode="trilinear")
            pred_low = self.model(x_low, t)
            pred_full = self._resize_tensor_5d(pred_low, orig_shape, mode="trilinear")
            return pred_full

        patch_size = self.patch_size
        if stride_zyx is None:
            stride_zyx = tuple(max(1, p // 2) for p in patch_size)
        if center_keep_size_zyx is None:
            center_keep_size_zyx = tuple(max(1, int(round(p * 0.75))) for p in patch_size)

        D, H, W = x_zyx.shape[2:]
        pd, ph, pw = patch_size
        sd, sh, sw = stride_zyx

        padded = center_pad_tensor_5d(
            x_zyx,
            (max(D, pd), max(H, ph), max(W, pw)),
            value=0.0,
        )

        acc = torch.zeros_like(padded)
        wsum = torch.zeros_like(padded)

        starts_d = compute_sliding_starts(padded.shape[2], pd, sd)
        starts_h = compute_sliding_starts(padded.shape[3], ph, sh)
        starts_w = compute_sliding_starts(padded.shape[4], pw, sw)

        keep_slc = center_crop_indices(patch_size, center_keep_size_zyx)
        kd = keep_slc[0].stop - keep_slc[0].start
        kh = keep_slc[1].stop - keep_slc[1].start
        kw = keep_slc[2].stop - keep_slc[2].start

        weight = torch.from_numpy(make_weight_window((kd, kh, kw))).to(self.device)[None, None]

        for d0 in starts_d:
            for h0 in starts_h:
                for w0 in starts_w:
                    patch = padded[:, :, d0:d0+pd, h0:h0+ph, w0:w0+pw]
                    pred_eps_patch = self.model(patch, t)
                    pred_eps_keep = pred_eps_patch[:, :, keep_slc[0], keep_slc[1], keep_slc[2]]

                    od = d0 + keep_slc[0].start
                    oh = h0 + keep_slc[1].start
                    ow = w0 + keep_slc[2].start

                    acc[:, :, od:od+kd, oh:oh+kh, ow:ow+kw] += pred_eps_keep * weight
                    wsum[:, :, od:od+kd, oh:oh+kh, ow:ow+kw] += weight

        out = acc / torch.clamp(wsum, min=1e-8)

        sd0 = (out.shape[2] - D) // 2
        sh0 = (out.shape[3] - H) // 2
        sw0 = (out.shape[4] - W) // 2
        out = out[:, :, sd0:sd0+D, sh0:sh0+H, sw0:sw0+W]
        return out

    @torch.no_grad()
    def ddim_invert_from_prob(
        self,
        init_prob_zyx: np.ndarray,
        start_t: int = 80,
    ):
        device = self.device
        start_t = int(np.clip(start_t, 0, self.num_steps - 1))

        init_prob_zyx = np.clip(init_prob_zyx, 0.0, 1.0).astype(np.float32)
        if self.prior_type == "full_volume":
            work_shape_zyx = tuple(int(v) for v in self.volume_shape)
            init_prob_work = self._resize_np_zyx(init_prob_zyx, work_shape_zyx, order=1)
        else:
            init_prob_work = init_prob_zyx

        x = torch.from_numpy(prob_to_diffusion_space(init_prob_work)[None, None]).to(
            device=device, dtype=torch.float32
        )

        for ti in range(0, start_t):
            tt = torch.full((1,), ti, device=device, dtype=torch.long)
            eps = self.predict_eps_full(x, tt)

            ab_t = self.diffusion.alpha_bars[ti]
            ab_next = self.diffusion.alpha_bars[ti + 1]

            x0_hat = (x - torch.sqrt(1.0 - ab_t) * eps) / torch.clamp(torch.sqrt(ab_t), min=1e-8)
            x0_hat = x0_hat.clamp(-1.0, 1.0)
            x = torch.sqrt(ab_next) * x0_hat + torch.sqrt(1.0 - ab_next) * eps

        return x.clamp(-1.5, 1.5)


    @torch.no_grad()
    def conditioned_sample_from_measurements(
        self,
        prob_shape_zyx,
        masks_2d,
        A_op,
        sil_scale_mm: float,
        start_t: int = 80,
        init_prob_zyx=None,
        dc_step_size: float = 0.03,
        dc_loss_type: str = "bce",
        grad_normalize: bool = True,
        stride_zyx=None,
        center_keep_size_zyx=None,
        clamp_x: bool = True,
        guidance_inner_steps: int = 1,
        n_stage_snapshots: int = 4,
        mask_weights_2d=None,
        lambda_tv_cond: float = 0.0,
        tv_mode: str = "anisotropic",
        tv_grad_normalize: bool = True,
        alpha_mode: str = "prob",
        init_method: str = "q_sample",
        guidance_mode: str = "full",
    ):
        device = self.device
        init_method = str(init_method).lower()
        guidance_mode = str(guidance_mode).strip().lower()
        if guidance_mode not in {"full", "prior_only"}:
            raise ValueError(
                f"Unknown guidance_mode={guidance_mode!r}. Valid: full, prior_only"
            )
        start_t = int(np.clip(start_t, 0, self.num_steps - 1))
        guidance_inner_steps = int(max(1, guidance_inner_steps))
        n_stage_snapshots = int(max(2, n_stage_snapshots))

        orig_shape_zyx = tuple(int(v) for v in prob_shape_zyx)

        # ------------------------------------------------------------
        # Important:
        # If this is a full-volume prior trained on 128^3,
        # all DDPM states should live in the training grid.
        # ------------------------------------------------------------
        if self.prior_type == "full_volume":
            work_shape_zyx = tuple(int(v) for v in self.volume_shape)
        else:
            work_shape_zyx = orig_shape_zyx

        def prob_orig_to_work(p):
            if p is None:
                return None
            p = np.clip(np.asarray(p, dtype=np.float32), 0.0, 1.0)
            if tuple(p.shape) == work_shape_zyx:
                return p
            return self._resize_np_zyx(p, work_shape_zyx, order=1)

        def prob_work_to_orig(p):
            p = np.clip(np.asarray(p, dtype=np.float32), 0.0, 1.0)
            if tuple(p.shape) == orig_shape_zyx:
                return p
            return self._resize_np_zyx(p, orig_shape_zyx, order=1)

        def x_work_to_x_orig(x_work):
            if tuple(x_work.shape[2:]) == orig_shape_zyx:
                return x_work
            return self._resize_tensor_5d(x_work, orig_shape_zyx, mode="trilinear")

        def grad_orig_to_work(g_orig):
            if tuple(g_orig.shape[2:]) == work_shape_zyx:
                return g_orig
            return self._resize_tensor_5d(g_orig, work_shape_zyx, mode="trilinear")

        init_prob_work = prob_orig_to_work(init_prob_zyx)
        initialization_noise = None

        if init_prob_work is None:
            x = torch.randn((1, 1, *work_shape_zyx), device=device, dtype=torch.float32)
            init_prob_work_vis = diffusion_to_prob(
                x[0, 0].clamp(-1.0, 1.0).detach().cpu().numpy()
            ).astype(np.float32)
        else:
            if init_method in {"q_sample", "random_noising"}:
                x0 = torch.from_numpy(prob_to_diffusion_space(init_prob_work)[None, None]).to(device)
                tt = torch.full((1,), start_t, device=device, dtype=torch.long)
                noise = torch.randn_like(x0)
                initialization_noise = noise
                x = self.diffusion.q_sample(x0, tt, noise)
            elif init_method in {"ddim_inversion", "ddim_inv"}:
                x = self.ddim_invert_from_prob(init_prob_zyx=init_prob_work, start_t=start_t)
            else:
                raise ValueError(
                    f"Unknown init_method={init_method}. "
                    "Valid: q_sample, random_noising, ddim_inversion, ddim_inv"
                )
            init_prob_work_vis = init_prob_work.copy()

        def _tensor_sha256(value):
            array = np.ascontiguousarray(value.detach().cpu().numpy())
            digest = hashlib.sha256()
            digest.update(str(tuple(array.shape)).encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(array.tobytes(order="C"))
            return digest.hexdigest()

        initial_noised_state_sha256 = _tensor_sha256(x)
        initialization_noise_sha256 = (
            None if initialization_noise is None else _tensor_sha256(initialization_noise)
        )

        history = []

        # Save snapshots in original reconstruction grid for plotting.
        stage_prob_zyx_list = [prob_work_to_orig(init_prob_work_vis)]
        stage_info = [{
            "stage_index": 0,
            "tag": "init_prior" if init_prob_zyx is not None else "initial_noise_state",
            "t": int(start_t),
            "progress": 0.0,
            "work_shape_zyx": list(work_shape_zyx),
            "orig_shape_zyx": list(orig_shape_zyx),
            "init_method": init_method,
            "guidance_mode": guidance_mode,
        }]

        target_progress = np.linspace(0.0, 1.0, n_stage_snapshots)[1:]
        next_stage_idx = 0
        measurement_guidance_evaluations = 0
        measurement_guidance_updates = 0

        for ti in reversed(range(start_t + 1)):
            tt = torch.full((1,), ti, device=device, dtype=torch.long)

            # For full-volume prior this is now directly 128^3 -> 128^3.
            pred_eps = self.predict_eps_full(
                x_zyx=x,
                t=tt,
                stride_zyx=stride_zyx,
                center_keep_size_zyx=center_keep_size_zyx,
            )

            x0_hat = ddpm_predict_x0_from_eps(self.diffusion, x, pred_eps, tt)
            aux_last = None

            if guidance_mode == "full" and dc_step_size > 0.0:
                for _ in range(guidance_inner_steps):

                    # ------------------------------------------------------------
                    # Data fidelity is defined on original projector grid.
                    # So upsample x0_hat to original grid, compute gradient there,
                    # then downsample gradient back to diffusion working grid.
                    # ------------------------------------------------------------
                    x0_hat_orig = x_work_to_x_orig(x0_hat)

                    grad_meas_orig, aux = compute_measurement_gradient_from_x0hat_zyx(
                        x0_hat_zyx=x0_hat_orig,
                        masks_2d=masks_2d,
                        A_op=A_op,
                        sil_scale_mm=sil_scale_mm,
                        device=device,
                        loss_type=dc_loss_type,
                        mask_weights_2d=mask_weights_2d,
                        alpha_mode=alpha_mode,
                    )
                    measurement_guidance_evaluations += 1

                    grad_meas = grad_orig_to_work(grad_meas_orig)

                    if grad_normalize:
                        scale_meas = torch.clamp(grad_meas.abs().mean(), min=1e-8)
                        grad_meas = grad_meas / scale_meas

                    grad_total = grad_meas

                    if float(lambda_tv_cond) > 0.0:
                        grad_tv, aux_tv = compute_tv_gradient_from_x0hat_zyx(
                            x0_hat_zyx=x0_hat,
                            device=device,
                            tv_mode=tv_mode,
                        )

                        if tv_grad_normalize:
                            scale_tv = torch.clamp(grad_tv.abs().mean(), min=1e-8)
                            grad_tv = grad_tv / scale_tv

                        grad_total = grad_total + float(lambda_tv_cond) * grad_tv
                        aux["tv_loss"] = aux_tv["tv_loss"]
                        aux["tv_grad_mean_abs"] = aux_tv["tv_grad_mean_abs"]
                        aux["tv_grad_max_abs"] = aux_tv["tv_grad_max_abs"]
                    else:
                        aux["tv_loss"] = None
                        aux["tv_grad_mean_abs"] = None
                        aux["tv_grad_max_abs"] = None

                    x0_hat = x0_hat - float(dc_step_size) * grad_total
                    x0_hat = x0_hat.clamp(-1.0, 1.0)
                    measurement_guidance_updates += 1
                    aux_last = aux

            if init_method in {"ddim_inversion", "ddim_inv"}:
                x_prev = ddim_sample_step_from_x0(
                    diffusion=self.diffusion,
                    x_t=x,
                    x0_hat=x0_hat,
                    pred_eps=pred_eps,
                    t=tt,
                )
            else:
                x_prev = ddpm_sample_step_from_x0(self.diffusion, x, x0_hat, tt)

            if clamp_x:
                x_prev = x_prev.clamp(-1.5, 1.5)

            x = x_prev

            if aux_last is None:
                history.append({
                    "t": int(ti),
                    "pred_mask_mean": None,
                    "pred_mask_max": None,
                    "line_mean": None,
                    "grad_mean_abs": None,
                    "grad_max_abs": None,
                    "weight_mean": None,
                    "weight_min": None,
                    "weight_max": None,
                    "tv_loss": None,
                    "tv_grad_mean_abs": None,
                    "tv_grad_max_abs": None,
                })
            else:
                history.append({
                    "t": int(ti),
                    "pred_mask_mean": aux_last["pred_mask_mean"],
                    "pred_mask_max": aux_last["pred_mask_max"],
                    "line_mean": aux_last["line_mean"],
                    "grad_mean_abs": aux_last["grad_mean_abs"],
                    "grad_max_abs": aux_last["grad_max_abs"],
                    "weight_mean": aux_last["weight_mean"],
                    "weight_min": aux_last["weight_min"],
                    "weight_max": aux_last["weight_max"],
                    "tv_loss": aux_last["tv_loss"],
                    "tv_grad_mean_abs": aux_last["tv_grad_mean_abs"],
                    "tv_grad_max_abs": aux_last["tv_grad_max_abs"],
                })

            progress = float((start_t - ti + 1) / (start_t + 1))

            while next_stage_idx < len(target_progress) and progress >= float(target_progress[next_stage_idx]):
                prob_stage_work = diffusion_to_prob(
                    x0_hat[0, 0].clamp(-1.0, 1.0).detach().cpu().numpy()
                ).astype(np.float32)

                prob_stage_orig = prob_work_to_orig(prob_stage_work)

                tag = "final" if (next_stage_idx == len(target_progress) - 1) else f"stage_{next_stage_idx + 1}"
                stage_prob_zyx_list.append(prob_stage_orig.astype(np.float32))
                stage_info.append({
                    "stage_index": len(stage_prob_zyx_list) - 1,
                    "tag": tag,
                    "t": int(ti),
                    "progress": float(progress),
                    "work_shape_zyx": list(work_shape_zyx),
                    "orig_shape_zyx": list(orig_shape_zyx),
                    "init_method": init_method,
                    "guidance_mode": guidance_mode,
                })
                next_stage_idx += 1

        prob_out_work = diffusion_to_prob(
            x0_hat[0, 0].clamp(-1.0, 1.0).detach().cpu().numpy()
        ).astype(np.float32)

        prob_out_zyx = prob_work_to_orig(prob_out_work).astype(np.float32)

        if len(stage_prob_zyx_list) < n_stage_snapshots:
            stage_prob_zyx_list.append(prob_out_zyx.copy())
            stage_info.append({
                "stage_index": len(stage_prob_zyx_list) - 1,
                "tag": "final",
                "t": 0,
                "progress": 1.0,
                "work_shape_zyx": list(work_shape_zyx),
                "orig_shape_zyx": list(orig_shape_zyx),
                "init_method": init_method,
                "guidance_mode": guidance_mode,
            })

        return {
            "prob_zyx": prob_out_zyx,
            "history": history,
            "stage_prob_zyx_list": stage_prob_zyx_list,
            "stage_info": stage_info,
            "guidance_mode": guidance_mode,
            "measurement_guidance_evaluations": int(measurement_guidance_evaluations),
            "measurement_guidance_updates": int(measurement_guidance_updates),
            "initial_noised_state_sha256": initial_noised_state_sha256,
            "initialization_noise_sha256": initialization_noise_sha256,
        }
# ============================================================
# Legacy unconditional refinement helper
# ============================================================



# ============================================================
# Remaining helper functions from your script
# ============================================================

import scipy.ndimage as ndi


def make_soft_mask_from_binary(mask_2d: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Convert a binary 2D mask {0,1} into a soft mask in [0,1].
    """
    soft = ndi.gaussian_filter(mask_2d.astype(np.float32), sigma=float(sigma))
    mx = float(soft.max())
    if mx > 1e-8:
        soft = soft / mx
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def maybe_make_soft_masks(
    masks_2d,
    use_soft_masks: bool = False,
    soft_mask_sigma: float = 1.5,
):
    if not use_soft_masks:
        return [m.astype(np.float32) for m in masks_2d]
    return [make_soft_mask_from_binary(m, sigma=soft_mask_sigma) for m in masks_2d]



def _disk_mask(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)


def corrupt_single_binary_mask(
    mask,
    rng,
    severity=0.20,
    mode="sparse_like_figure",
    max_shift_px=2,
    fp_near_vessel_prob=0.85,
):
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape

    if m.sum() == 0 or severity <= 0:
        return m

    if mode == "sparse_like_figure":
        if rng.random() < 0.25:
            max_s = max(0, int(round(max_shift_px * severity)))
            if max_s > 0:
                dy = int(rng.integers(-max_s, max_s + 1))
                dx = int(rng.integers(-max_s, max_s + 1))
                m = ndi.shift(m.astype(np.float32), shift=(dy, dx), order=0, mode="constant", cval=0.0)
                m = (m > 0.5).astype(np.uint8)

        if rng.random() < 0.25:
            sigma = float(rng.uniform(0.0, 0.8 * severity))
            if sigma > 1e-6:
                blurred = ndi.gaussian_filter(m.astype(np.float32), sigma=sigma)
                thr = float(rng.uniform(0.48, 0.58))
                m = (blurred > thr).astype(np.uint8)

        ys, xs = np.where(m > 0)
        n_fn = int(round(severity * 2.5))
        for _ in range(n_fn):
            if len(ys) == 0:
                break
            k = int(rng.integers(0, len(ys)))
            cy, cx = int(ys[k]), int(xs[k])
            r = int(rng.integers(1, max(2, int(round(1 + 2 * severity)))))
            hole = _disk_mask(h, w, cy, cx, r)
            m[hole] = 0

        halo_iters = max(4, int(round(0.02 * min(h, w))))
        halo = ndi.binary_dilation(m.astype(bool), iterations=halo_iters)
        halo_ys, halo_xs = np.where(halo)
        n_fp = int(round(6 + 28 * severity))

        for _ in range(n_fp):
            if len(halo_ys) > 0 and rng.random() < fp_near_vessel_prob:
                k = int(rng.integers(0, len(halo_ys)))
                cy, cx = int(halo_ys[k]), int(halo_xs[k])
            else:
                cy = int(rng.integers(0, h))
                cx = int(rng.integers(0, w))

            r = 1 if rng.random() < 0.85 else 2
            blob = _disk_mask(h, w, cy, cx, r)
            m[blob] = 1

        if rng.random() < (0.15 + 0.35 * severity):
            n_blobs = max(1, int(round(severity * 3)))
            for _ in range(n_blobs):
                if len(halo_ys) > 0 and rng.random() < 0.75:
                    k = int(rng.integers(0, len(halo_ys)))
                    cy, cx = int(halo_ys[k]), int(halo_xs[k])
                else:
                    cy = int(rng.integers(0, h))
                    cx = int(rng.integers(0, w))

                r = int(rng.integers(2, 4))
                blob = _disk_mask(h, w, cy, cx, r)
                m[blob] = 1

        return m.astype(np.uint8)

    elif mode == "strong":
        if rng.random() < 0.7:
            max_s = max(0, int(round(4 * severity)))
            if max_s > 0:
                dy = int(rng.integers(-max_s, max_s + 1))
                dx = int(rng.integers(-max_s, max_s + 1))
                m = ndi.shift(m.astype(np.float32), shift=(dy, dx), order=0, mode="constant", cval=0.0)
                m = (m > 0.5).astype(np.uint8)

        if rng.random() < 0.8:
            struct = ndi.generate_binary_structure(2, 1)
            n_iter = max(1, int(round(1 + 3 * severity)))
            op = rng.choice(["erode", "dilate", "open", "close"])
            if op == "erode":
                m = ndi.binary_erosion(m, structure=struct, iterations=n_iter).astype(np.uint8)
            elif op == "dilate":
                m = ndi.binary_dilation(m, structure=struct, iterations=n_iter).astype(np.uint8)
            elif op == "open":
                m = ndi.binary_opening(m, structure=struct, iterations=max(1, n_iter // 2)).astype(np.uint8)
            else:
                m = ndi.binary_closing(m, structure=struct, iterations=max(1, n_iter // 2)).astype(np.uint8)

        if rng.random() < 0.7:
            sigma = float(rng.uniform(0.0, 2.0 * severity))
            if sigma > 1e-6:
                blurred = ndi.gaussian_filter(m.astype(np.float32), sigma=sigma)
                thr = float(rng.uniform(0.45, 0.60))
                m = (blurred > thr).astype(np.uint8)

        n_fn = int(round(severity * 6))
        ys, xs = np.where(m > 0)
        for _ in range(n_fn):
            if len(ys) == 0:
                break
            k = int(rng.integers(0, len(ys)))
            cy, cx = int(ys[k]), int(xs[k])
            r = int(rng.integers(
                max(2, int(0.01 * min(h, w))),
                max(3, int(0.04 * severity * min(h, w)) + 3)
            ))
            hole = _disk_mask(h, w, cy, cx, r)
            m[hole] = 0

        if rng.random() < 0.6 and m.sum() > 0:
            n_breaks = max(1, int(round(1 + 3 * severity)))
            for _ in range(n_breaks):
                if rng.random() < 0.5:
                    y0 = int(rng.integers(0, h))
                    th = max(1, int(round(severity * 0.03 * h)) + 1)
                    y1 = max(0, y0 - th)
                    y2 = min(h, y0 + th + 1)
                    m[y1:y2, :] = 0
                else:
                    x0 = int(rng.integers(0, w))
                    th = max(1, int(round(severity * 0.03 * w)) + 1)
                    x1 = max(0, x0 - th)
                    x2 = min(w, x0 + th + 1)
                    m[:, x1:x2] = 0

        if rng.random() < 0.8:
            n_fp = int(round(severity * 8))
            for _ in range(n_fp):
                cy = int(rng.integers(0, h))
                cx = int(rng.integers(0, w))
                r = int(rng.integers(1, max(2, int(0.03 * severity * min(h, w)) + 2)))
                blob = _disk_mask(h, w, cy, cx, r)
                m[blob] = 1

        if severity < 0.35:
            m = ndi.binary_opening(
                m,
                structure=ndi.generate_binary_structure(2, 1),
                iterations=1,
            ).astype(np.uint8)

        return m.astype(np.uint8)

    else:
        raise ValueError(f"Unknown corruption mode: {mode}")


def corrupt_multiview_masks(
    masks_2d,
    severity=0.20,
    seed=0,
    per_view_randomness=True,
    corrupt_view_prob=0.35,
    min_corrupt_views=1,
    mode="sparse_like_figure",
    return_info=False,
):
    rng = np.random.default_rng(seed)
    n_views = len(masks_2d)

    if severity <= 0.0 or n_views == 0:
        out = [m.copy().astype(np.uint8) for m in masks_2d]
        info = {
            "corrupted_view_indices": [],
            "clean_view_indices": list(range(n_views)),
            "mode": mode,
            "corrupt_view_prob": float(corrupt_view_prob),
            "severity": float(severity),
        }
        return (out, info) if return_info else out

    corrupt_flags = rng.random(n_views) < float(np.clip(corrupt_view_prob, 0.0, 1.0))

    if min_corrupt_views is not None:
        min_corrupt_views = int(max(0, min_corrupt_views))
        if corrupt_flags.sum() < min_corrupt_views:
            need = min(min_corrupt_views, n_views)
            chosen = rng.choice(n_views, size=need, replace=False)
            corrupt_flags[:] = False
            corrupt_flags[chosen] = True

    out = []
    corrupted_view_indices = []
    clean_view_indices = []

    for i, m in enumerate(masks_2d):
        if corrupt_flags[i]:
            local_rng = np.random.default_rng(rng.integers(0, 2**31 - 1)) if per_view_randomness else rng
            out.append(
                corrupt_single_binary_mask(
                    m,
                    rng=local_rng,
                    severity=severity,
                    mode=mode,
                )
            )
            corrupted_view_indices.append(int(i))
        else:
            out.append(m.copy().astype(np.uint8))
            clean_view_indices.append(int(i))

    info = {
        "corrupted_view_indices": corrupted_view_indices,
        "clean_view_indices": clean_view_indices,
        "mode": mode,
        "corrupt_view_prob": float(corrupt_view_prob),
        "severity": float(severity),
    }

    return (out, info) if return_info else out


def build_mask_confidence_map(
    mask,
    boundary_band_px=3.0,
    small_component_area=24,
    small_component_weight=0.05,
    fg_boundary_weight=0.45,
    bg_boundary_weight=0.65,
    min_weight=0.02,
):
    m = (mask > 0).astype(bool)
    h, w = m.shape
    weights = np.ones((h, w), dtype=np.float32)

    if m.sum() == 0:
        return weights

    dist_in = ndi.distance_transform_edt(m).astype(np.float32)
    if boundary_band_px > 1e-6:
        fg_conf = np.clip(dist_in / float(boundary_band_px), 0.0, 1.0)
        fg_conf = fg_boundary_weight + (1.0 - fg_boundary_weight) * fg_conf
        weights[m] *= fg_conf[m]

    dist_out = ndi.distance_transform_edt(~m).astype(np.float32)
    if boundary_band_px > 1e-6:
        bg_conf = np.clip(dist_out / float(boundary_band_px), 0.0, 1.0)
        bg_conf = bg_boundary_weight + (1.0 - bg_boundary_weight) * bg_conf
        weights[~m] *= bg_conf[~m]

    labeled, ncomp = ndi.label(m)
    if ncomp > 0:
        sizes = np.bincount(labeled.ravel())
        for comp_id in range(1, ncomp + 1):
            area = int(sizes[comp_id])
            if area < int(small_component_area):
                weights[labeled == comp_id] *= float(small_component_weight)

    return np.clip(weights, float(min_weight), 1.0).astype(np.float32)


def build_multiview_mask_confidence_maps(
    masks_2d,
    boundary_band_px=3.0,
    small_component_area=24,
    small_component_weight=0.05,
    fg_boundary_weight=0.45,
    bg_boundary_weight=0.65,
    min_weight=0.02,
):
    out = []
    for m in masks_2d:
        out.append(
            build_mask_confidence_map(
                mask=m,
                boundary_band_px=boundary_band_px,
                small_component_area=small_component_area,
                small_component_weight=small_component_weight,
                fg_boundary_weight=fg_boundary_weight,
                bg_boundary_weight=bg_boundary_weight,
                min_weight=min_weight,
            )
        )
    return out


def project_probability_map_to_multiview_fgprob(
    prob_zyx,
    masks_2d,
    A_op,
    sil_scale_mm,
    device,
    eps=1e-6,
    alpha_mode: str = "prob",
):
    def _find_perm(src_shape, dst_shape):
        import itertools
        for p in itertools.permutations(range(len(src_shape))):
            if tuple(src_shape[i] for i in p) == tuple(dst_shape):
                return p
        return None

    prob_zyx_t = torch.as_tensor(
        prob_zyx,
        device=device,
        dtype=torch.float32,
    ).clamp(eps, 1.0 - eps)

    if alpha_mode == "prob":
        alpha_zyx = prob_zyx_t
    elif alpha_mode == "neglog":
        alpha_zyx = -torch.log(torch.clamp(1.0 - prob_zyx_t, min=eps))
    else:
        raise ValueError(f"Unknown alpha_mode: {alpha_mode}")

    line_raw = A_op(alpha_zyx)

    if not torch.is_tensor(line_raw):
        line_raw = torch.as_tensor(line_raw, device=device, dtype=torch.float32)
    else:
        line_raw = line_raw.to(device=device, dtype=torch.float32)

    y_vhw = np.stack(masks_2d, axis=0).astype(np.float32)

    if tuple(line_raw.shape) != tuple(y_vhw.shape):
        forward_perm = _find_perm(tuple(line_raw.shape), tuple(y_vhw.shape))
        if forward_perm is None:
            raise RuntimeError(
                f"Cannot match projector output shape {tuple(line_raw.shape)} "
                f"to mask shape {tuple(y_vhw.shape)}"
            )
        line_vhw = line_raw.permute(*forward_perm).contiguous()
    else:
        line_vhw = line_raw

    q_fg_vhw = 1.0 - torch.exp(-line_vhw / float(sil_scale_mm))
    pred_vhw = q_fg_vhw.detach().cpu().numpy().astype(np.float32)

    return np.clip(pred_vhw, 0.0, 1.0)


def save_view_consistency_panel(
    obs_mask,
    pred_prob,
    disagreement,
    likely_fp,
    likely_fn,
    entropy,
    out_png,
    title,
):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    axs[0, 0].imshow(obs_mask, cmap="gray", vmin=0, vmax=1)
    axs[0, 0].set_title("Observed mask")
    axs[0, 0].axis("off")

    im1 = axs[0, 1].imshow(pred_prob, cmap="viridis", vmin=0, vmax=1)
    axs[0, 1].set_title("Projected reconstruction")
    axs[0, 1].axis("off")
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im2 = axs[0, 2].imshow(disagreement, cmap="magma", vmin=0, vmax=1)
    axs[0, 2].set_title("|pred - mask|")
    axs[0, 2].axis("off")
    fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

    im3 = axs[1, 0].imshow(likely_fp, cmap="inferno", vmin=0, vmax=1)
    axs[1, 0].set_title("Likely false positive")
    axs[1, 0].axis("off")
    fig.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im4 = axs[1, 1].imshow(likely_fn, cmap="inferno", vmin=0, vmax=1)
    axs[1, 1].set_title("Likely false negative")
    axs[1, 1].axis("off")
    fig.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)

    im5 = axs[1, 2].imshow(entropy, cmap="plasma", vmin=0, vmax=np.log(2.0))
    axs[1, 2].set_title("Predictive entropy")
    axs[1, 2].axis("off")
    fig.colorbar(im5, ax=axs[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_mask_view_inconsistency(
    prob_zyx,
    masks_2d,
    A_op,
    sil_scale_mm,
    device,
    out_dir,
    prefix="reprojection",
    eps=1e-6,
    alpha_mode: str = "prob",
):
    ensure_dir(out_dir)

    pred_vhw = project_probability_map_to_multiview_fgprob(
        prob_zyx=prob_zyx,
        masks_2d=masks_2d,
        A_op=A_op,
        sil_scale_mm=sil_scale_mm,
        device=device,
        eps=eps,
        alpha_mode=alpha_mode,
    )

    np.savez_compressed(
        os.path.join(out_dir, f"{prefix}_predicted_view_probs.npz"),
        **{f"view_{i:03d}": pred_vhw[i].astype(np.float32) for i in range(pred_vhw.shape[0])}
    )

    results = []

    for v in range(len(masks_2d)):
        y = (masks_2d[v] > 0).astype(np.float32)
        pred = pred_vhw[v].astype(np.float32)

        disagreement = np.abs(pred - y)
        likely_fp = y * (1.0 - pred)
        likely_fn = (1.0 - y) * pred

        entropy = -(
            pred * np.log(pred + eps) +
            (1.0 - pred) * np.log(1.0 - pred + eps)
        )

        fg_count = float(y.sum())
        bg_count = float((1.0 - y).sum())

        mean_abs_error = float(disagreement.mean())
        fg_support_error = float(likely_fp.sum() / max(fg_count, 1.0))
        bg_leak_error = float(likely_fn.sum() / max(bg_count, 1.0))
        entropy_mean = float(entropy.mean())

        pred_bin = (pred >= 0.5).astype(np.uint8)
        inter = float(np.logical_and(pred_bin > 0, y > 0).sum())
        union = float(np.logical_or(pred_bin > 0, y > 0).sum())
        iou_bin = inter / max(union, 1.0)

        suspicious_score = float(
            0.55 * fg_support_error +
            0.30 * bg_leak_error +
            0.15 * mean_abs_error
        )

        title = (
            f"View {v:03d} | suspicious={suspicious_score:.4f} | "
            f"FG_err={fg_support_error:.4f} | BG_err={bg_leak_error:.4f} | "
            f"MAE={mean_abs_error:.4f} | IoU@0.5={iou_bin:.4f}"
        )

        save_view_consistency_panel(
            obs_mask=y,
            pred_prob=pred,
            disagreement=disagreement,
            likely_fp=likely_fp,
            likely_fn=likely_fn,
            entropy=entropy,
            out_png=os.path.join(out_dir, f"{prefix}_view_{v:03d}_analysis.png"),
            title=title,
        )

        results.append({
            "view_index": int(v),
            "suspicious_score": suspicious_score,
            "mean_abs_error": mean_abs_error,
            "fg_support_error": fg_support_error,
            "bg_leak_error": bg_leak_error,
            "entropy_mean": entropy_mean,
            "binary_iou_at_0p5": float(iou_bin),
            "fg_pixels": int(fg_count),
            "bg_pixels": int(bg_count),
        })

    results_sorted = sorted(results, key=lambda x: x["suspicious_score"], reverse=True)

    with open(os.path.join(out_dir, f"{prefix}_view_ranking.json"), "w") as f:
        json.dump(results_sorted, f, indent=2)

    fieldnames = [
        "view_index",
        "suspicious_score",
        "mean_abs_error",
        "fg_support_error",
        "bg_leak_error",
        "entropy_mean",
        "binary_iou_at_0p5",
        "fg_pixels",
        "bg_pixels",
    ]
    with open(os.path.join(out_dir, f"{prefix}_view_ranking.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_sorted:
            writer.writerow(r)

    print("\n[view inconsistency ranking: most suspicious first]")
    for r in results_sorted:
        print(
            f"  view {r['view_index']:03d} | suspicious={r['suspicious_score']:.4f} | "
            f"FG_err={r['fg_support_error']:.4f} | BG_err={r['bg_leak_error']:.4f} | "
            f"MAE={r['mean_abs_error']:.4f} | IoU@0.5={r['binary_iou_at_0p5']:.4f}"
        )

    return results_sorted


def self_threshold_prob(prob, min_prob=0.01):
    from skimage.filters import threshold_otsu, threshold_li, threshold_triangle
    import numpy as np

    vals = np.asarray(prob, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > float(min_prob)]

    if len(vals) < 100:
        return 0.10

    t1 = threshold_otsu(vals)
    t2 = threshold_li(vals)
    t3 = threshold_triangle(vals)

    thr = float(np.median([t1, t2, t3]))
    return float(np.clip(thr, 0.05, 0.70))


def _stable_case_view_seed(base_seed, case_id, n_views):
    """Derive a process-order-independent seed for one case/view cell."""
    if base_seed is None:
        return None
    payload = f"{int(base_seed)}|{case_id}|{int(n_views)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False) % (2**31)


def _array_sha256(array):
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


_FILE_SHA256_CACHE = {}


def _file_sha256(path):
    """Hash an immutable model artifact without rereading it for every case."""
    if path in [None, "", "None"]:
        return None
    resolved = os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))
    stat = os.stat(resolved)
    cache_key = (resolved, int(stat.st_size), int(stat.st_mtime_ns))
    cached = _FILE_SHA256_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    _FILE_SHA256_CACHE[cache_key] = result
    return result


def average_precision_binary(y_true, y_score):
    """Exact non-interpolated AP with tied scores evaluated as one threshold."""
    truth = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    score = np.asarray(y_score, dtype=np.float32).reshape(-1)
    if truth.shape != score.shape:
        raise ValueError(
            f"AUPRC truth/score shape mismatch: {truth.shape} vs {score.shape}"
        )
    if not np.isfinite(score).all():
        raise ValueError("AUPRC score contains non-finite values")
    positives = int(np.count_nonzero(truth))
    if positives == 0:
        return float("nan")

    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_truth = truth[order].astype(np.int64, copy=False)
    cumulative_tp = np.cumsum(sorted_truth, dtype=np.int64)
    threshold_ends = np.flatnonzero(
        np.r_[sorted_score[1:] != sorted_score[:-1], True]
    )
    tp = cumulative_tp[threshold_ends].astype(np.float64, copy=False)
    predicted = (threshold_ends + 1).astype(np.float64, copy=False)
    precision = tp / predicted
    recall = tp / float(positives)
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision, dtype=np.float64))


def cldice_binary(prediction, ground_truth):
    """Standard 3-D clDice for two binary masks."""
    from skimage.morphology import skeletonize

    pred = np.asarray(prediction) > 0
    gt = np.asarray(ground_truth) > 0
    pred_count = int(np.count_nonzero(pred))
    gt_count = int(np.count_nonzero(gt))
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0

    pred_skeleton = skeletonize(pred)
    gt_skeleton = skeletonize(gt)
    pred_skeleton_count = int(np.count_nonzero(pred_skeleton))
    gt_skeleton_count = int(np.count_nonzero(gt_skeleton))
    if pred_skeleton_count == 0 or gt_skeleton_count == 0:
        return 0.0
    topology_precision = float(np.count_nonzero(pred_skeleton & gt)) / float(
        pred_skeleton_count
    )
    topology_sensitivity = float(np.count_nonzero(gt_skeleton & pred)) / float(
        gt_skeleton_count
    )
    denominator = topology_precision + topology_sensitivity
    if denominator <= 0.0:
        return 0.0
    return float(2.0 * topology_precision * topology_sensitivity / denominator)


def segmentation_quality_2d(predicted_masks, gt_projection_masks):
    """Return per-view and macro hard-mask Dice/IoU for 2-D segmentation."""
    if predicted_masks is None or gt_projection_masks is None:
        return None
    if len(predicted_masks) != len(gt_projection_masks):
        raise ValueError(
            "2-D segmentation/GT projection view count mismatch: "
            f"{len(predicted_masks)} vs {len(gt_projection_masks)}"
        )
    per_view = []
    for view_index, (prediction, target) in enumerate(
        zip(predicted_masks, gt_projection_masks)
    ):
        pred = np.asarray(prediction) > 0
        gt = np.asarray(target) > 0
        if pred.shape != gt.shape:
            raise ValueError(
                f"2-D segmentation shape mismatch at view {view_index}: "
                f"{pred.shape} vs {gt.shape}"
            )
        dice, iou = dice_iou(pred, gt)
        per_view.append(
            {
                "view_index": int(view_index),
                "dice": float(dice),
                "iou": float(iou),
                "prediction_foreground_pixels": int(np.count_nonzero(pred)),
                "gt_foreground_pixels": int(np.count_nonzero(gt)),
            }
        )
    return {
        "per_view": per_view,
        "mean_dice": float(np.mean([item["dice"] for item in per_view])),
        "mean_iou": float(np.mean([item["iou"] for item in per_view])),
    }


def validate_shared_observation_metadata(metadata, expected):
    """Fail fast when a frozen observation bundle belongs to another setup."""
    if not isinstance(metadata, dict):
        raise ValueError("Shared-observation metadata must be a JSON object")
    for key, expected_value in expected.items():
        if key not in metadata:
            raise ValueError(f"Shared-observation metadata is missing {key!r}")
        actual = metadata[key]
        if isinstance(expected_value, bool):
            matches = bool(actual) is expected_value
        elif isinstance(expected_value, (int, float)):
            matches = math.isclose(
                float(actual),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        elif key.endswith("_ckpt") and expected_value is not None:
            matches = os.path.realpath(str(actual)) == os.path.realpath(
                str(expected_value)
            )
        else:
            matches = actual == expected_value
        if not matches:
            raise ValueError(
                f"Shared-observation metadata mismatch for {key}: "
                f"stored={actual!r}, requested={expected_value!r}"
            )


# ============================================================
# Single reconstruction for one angle count
# ============================================================

def evaluate_prob_map_against_gt(
    prob_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    n_views: int,
    default_thr: float,
    min_vox_component: int = 50,
    recon_mode: str = "bayes",
    threshold_mode: str | None = None,
    fixed_threshold: float = 0.05,
):
    mode = None if threshold_mode is None else str(threshold_mode).strip().lower()
    if mode == "auto":
        mode = None
    if mode not in {None, "self", "fixed", "angle"}:
        raise ValueError(
            f"Unknown threshold_mode={threshold_mode!r}; expected self, fixed, or angle"
        )
    if mode == "fixed":
        thr_eff = float(fixed_threshold)
        resolved_mode = "fixed"
    elif mode == "self":
        thr_eff = self_threshold_prob(prob_xyz, min_prob=0.01)
        resolved_mode = "self"
    elif mode == "angle":
        thr_eff = get_angle_dependent_threshold(
            n_views=n_views,
            default_thr=default_thr,
        )
        resolved_mode = "angle"
    elif recon_mode in ["pnp_diffusion", "cond_diffusion", "bayes_then_diffusion"]:
        thr_eff = self_threshold_prob(prob_xyz, min_prob=0.01)
        resolved_mode = "self"
    else:
        thr_eff = get_angle_dependent_threshold(
            n_views=n_views,
            default_thr=default_thr,
        )
        resolved_mode = "angle"
    if not np.isfinite(thr_eff) or not 0.0 <= float(thr_eff) <= 1.0:
        raise ValueError(f"Invalid evaluation threshold: {thr_eff}")
    fixed_xyz = posterior_to_binary_map(
        prob_xyz=prob_xyz,
        thr=float(thr_eff),
        min_vox_component=min_vox_component,
    )
    dice_fixed, iou_fixed = dice_iou(fixed_xyz > 0, gt_xyz > 0)

    best = find_best_threshold(
        prob_xyz=prob_xyz,
        gt_xyz=gt_xyz,
        thresholds=np.linspace(0.05, 0.95, 19, dtype=np.float32),
        min_vox_component=min_vox_component,
    )

    return {
        "thr_eff": float(thr_eff),
        "fixed_mask_xyz": fixed_xyz.astype(np.uint8),
        "dice_fixed": float(dice_fixed),
        "iou_fixed": float(iou_fixed),
        "oracle_mask_xyz": best["mask"].astype(np.uint8),
        "dice_oracle": float(best["dice"]),
        "iou_oracle": float(best["iou"]),
        "thr_oracle": float(best["thr"]),
        "threshold_mode": resolved_mode,
    }

@torch.no_grad()
def project_label_from_mask_segstyle(mask3d: np.ndarray, vg, pg_i, device: str, thr: float = 1e-8) -> np.ndarray:
    A = ts.operator(vg, pg_i)
    m = torch.from_numpy(mask3d.astype(np.float32)).to(device=device)
    p = A(m).squeeze()
    if p.ndim != 2:
        raise RuntimeError(f"Unexpected mask projector output shape: {tuple(p.shape)}")
    y = (p > thr).to(torch.uint8)
    return y.detach().cpu().numpy()


@torch.no_grad()
def simulate_multiview_xrays_from_local_ct(
    img_xyz,
    mask_xyz,
    vg,
    pg,
    device='cpu',
    hu_min=-1000.0,
    hu_max=3000.0,
    hu_scale=5e-3,
    delta_mu=0.008,
    photon_count=50000.0,
):
    img_xyz = np.asarray(img_xyz, dtype=np.float32)
    mask_xyz = np.asarray(mask_xyz, dtype=np.float32)

    img_xyz_clip = np.clip(img_xyz, float(hu_min), float(hu_max))
    mu01_xyz = (img_xyz_clip - float(hu_min)) / (float(hu_max) - float(hu_min) + 1e-12)
    mu_xyz = mu01_xyz * float(hu_scale)

    mu_zyx = np.transpose(mu_xyz, (2, 1, 0)).astype(np.float32)
    mask_zyx = np.transpose(mask_xyz.astype(np.float32), (2, 1, 0)).astype(np.float32)

    mu_t = torch.from_numpy(mu_zyx).to(device=device, dtype=torch.float32)
    mask_t = torch.from_numpy(mask_zyx).to(device=device, dtype=torch.float32)

    xrays_2d = []
    for v in range(len(pg)):
        A_k = ts.operator(vg, pg[v])
        p_mu = A_k(mu_t).squeeze()
        p_mask = A_k(mask_t).squeeze()

        if p_mu.ndim != 2 or p_mask.ndim != 2:
            raise RuntimeError(
                f"Unexpected projector output shapes: p_mu={tuple(p_mu.shape)}, p_mask={tuple(p_mask.shape)}"
            )

        p = p_mu + float(delta_mu) * p_mask
        I = torch.exp(-p)
        counts = torch.poisson(I * float(photon_count))
        counts = torch.clamp(counts, min=1.0)
        I_noisy = counts / float(photon_count)

        xray_np = I_noisy.detach().cpu().numpy().astype(np.float32)
        xray_np = normalize_image_01(xray_np)
        xrays_2d.append(xray_np)

    return xrays_2d


@torch.no_grad()
def project_label_from_mask_segstyle(mask3d: np.ndarray, vg, pg_i, device: str, thr: float = 1e-8) -> np.ndarray:
    A = ts.operator(vg, pg_i)
    m = torch.from_numpy(mask3d.astype(np.float32)).to(device=device)
    p = A(m).squeeze()
    if p.ndim != 2:
        raise RuntimeError(f"Unexpected mask projector output shape: {tuple(p.shape)}")
    y = (p > thr).to(torch.uint8)
    return y.detach().cpu().numpy()

def resize_mask_list_to_shape(mask_list, out_shape_hw, is_binary=False, binary_thr=0.5):
    out_h, out_w = [int(v) for v in out_shape_hw]
    resized = []

    for m in mask_list:
        m = np.asarray(m, dtype=np.float32)
        zoom = (out_h / m.shape[0], out_w / m.shape[1])
        r = ndi.zoom(m, zoom=zoom, order=1).astype(np.float32)

        if r.shape != (out_h, out_w):
            fixed = np.zeros((out_h, out_w), dtype=np.float32)
            h = min(out_h, r.shape[0])
            w = min(out_w, r.shape[1])
            fixed[:h, :w] = r[:h, :w]
            r = fixed

        if is_binary:
            r = (r >= float(binary_thr)).astype(np.uint8)
        else:
            r = np.clip(r, 0.0, 1.0).astype(np.float32)

        resized.append(r)

    return resized

def process_case_for_n_views_hybrid(
    prep,
    out_dir,
    N_VIEWS=8,
    R_SO_MM=400.0,
    PLANE='yz',
    TILT_DEG=0.0,
    TILT_AXIS='x',
    det_shape=(512, 512),
    det_size_mm=(300.0, 300.0),
    d_OD_mm=400.0,
    proj_eps=1e-8,
    bp_eps=1e-8,
    allow_missing_views=0,
    n_points=20000,
    device='cpu',
    diffusion_prior_ckpt=None,
    diffusion_prior_device=None,
    cond_dc_step_size=0.03,
    cond_vh_soft_max=0.30,
    alpha_mode="prob",
    diffusion_denoise_t=15,
    diffusion_n_averages=1,
    seg_corrupt_severity=0.0,
    seg_corrupt_seed=0,
    seg_corrupt_view_prob=0.35,
    seg_corrupt_mode="sparse_like_figure",
    recon_mode="cond_diffusion",
    guidance_mode="full",
    guidance_inner_steps=1,
    diffusion_n_stage_snapshots=4,
    diffusion_dc_loss_type="bce",
    lambda_tv_cond=0.0,
    tv_mode="anisotropic",
    tv_grad_normalize=True,
    pnp_init_mode="vh",
    use_soft_masks=False,
    soft_mask_sigma=1.0,
    use_trained_segmentation=False,
    soft_seg_ckpt=None,
    soft_seg_device=None,
    soft_seg_base_ch=16,
    soft_seg_pred_thr=0.5,
    xray_photon_count=50000,
    xray_hu_scale=5e-3,
    xray_hu_min=-1000.0,
    xray_hu_max=3000.0,
    use_soft_masks_for_recon=True,
    xray_delta_mu=0.008,
    max_dim=9999,
    min_vox_component=50,
    cond_init_method="q_sample",
    lowres_recon_det_shape=None,
    shared_obs_root=None,
    shared_obs_mode="none",
    exact_grid_shape_xyz=None,
    exact_mask_resample_mode="nearest",
    enforce_diffusion_grid_match=True,
    eval_threshold_mode="auto",
    eval_fixed_thr=0.05,
    reconstruction_seed=None,
    require_shared_obs_aux=False,
):
    if str(recon_mode) != "cond_diffusion":
        raise ValueError("This release supports recon_mode='cond_diffusion' only.")
    if str(pnp_init_mode).lower() not in {"vh", "visual_hull", "visual-hull"}:
        raise ValueError("This release uses visual-hull initialization (pnp_init_mode='vh').")
    if str(cond_init_method) != "q_sample":
        raise ValueError("This release uses cond_init_method='q_sample'.")
    case_id = prep["case_id"]
    img_xyz = prep["img_xyz"]
    mask_xyz = prep["mask_xyz"]
    mask_zyx = prep["mask_zyx"]
    spacing_eff = prep["spacing_eff_xyz"]
    spacing_zyx = prep["spacing_zyx"]
    vg = prep["vg"]
    aff_local = prep["aff_local"]
    bbox_min = prep["bbox_min_xyz"]
    bbox_max = prep["bbox_max_xyz"]
    scale = prep["scale"]
    lbl_path = prep["label_path"]
    img_path = prep.get("img_path", None)

    case_out = os.path.join(out_dir, case_id, f"views_{int(N_VIEWS):03d}")
    views_out = os.path.join(case_out, "views")
    views_clean_out = os.path.join(case_out, "views_clean")
    ensure_dir(case_out)
    ensure_dir(views_out)

    print("\n" + "=" * 90)
    print(f"[case] {case_id} | N_VIEWS={N_VIEWS}")
    print(f"  label: {lbl_path}")
    print(f"  image: {img_path}")
    print(f"  shape             : {mask_xyz.shape}")
    print(f"  spacing           : {spacing_eff.tolist()} mm")
    print(f"  bbox_min          : {bbox_min.tolist()}")
    print(f"  bbox_max          : {bbox_max.tolist()}")
    print(f"  scale             : {scale:.5f}")
    print(f"  recon_mode        : {recon_mode}")
    if recon_mode == "cond_diffusion":
        print(f"  guidance_mode     : {guidance_mode}")
    print(f"  use_trained_seg   : {bool(use_trained_segmentation)}")

    O_mm = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    center_mm = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    S_mm, used_angles_rad = build_circular_trajectory_mm(
        n_views=N_VIEWS,
        R_SO_MM=R_SO_MM,
        plane=PLANE,
        center_mm=center_mm,
        start_angle_rad=0.0,
        end_angle_rad= np.pi,
        endpoint=False,
        tilt_deg=TILT_DEG,
        tilt_axis=TILT_AXIS,
    )

    pg, _ = build_cone_vec_from_world_sources(
        S_mm,
        O_mm,
        d_OD_mm,
        det_shape=det_shape,
        det_size=det_size_mm,
        up_world=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        assume_tomosipo_zyx=True,
    )

    dp = None
    if recon_mode == "cond_diffusion":
        if diffusion_prior_ckpt not in [None, "", "None"]:
            dp_device = (
                diffusion_prior_device
                if diffusion_prior_device is not None
                else device
            )
            dp = PatchDiffusionPrior(
                diffusion_prior_ckpt,
                device=dp_device,
            )

    current_grid_shape = tuple(int(v) for v in mask_xyz.shape)
    print(
        f"  exact_grid_applied : "
        f"{bool(prep.get('exact_grid_applied', False))}"
    )
    print(f"  common 3-D grid    : {current_grid_shape}")

    if bool(enforce_diffusion_grid_match) and dp is not None:
        if getattr(dp, "prior_type", None) == "full_volume":
            prior_shape = tuple(int(v) for v in dp.volume_shape)
            if current_grid_shape != prior_shape:
                raise RuntimeError(
                    "Direct-grid mode requires measurement/reconstruction grid "
                    f"{current_grid_shape} to equal diffusion training grid "
                    f"{prior_shape}. Otherwise the sampler would resize internally."
                )
            print(
                "  diffusion grid     : exact match; "
                "no 3-D interpolation"
            )

    grid_meta = {
        "exact_grid_applied": bool(
            prep.get("exact_grid_applied", False)
        ),
        "common_grid_shape_xyz": list(current_grid_shape),
        "spacing_mm_xyz": [
            float(v) for v in np.asarray(spacing_eff).reshape(-1)
        ],
        "physical_size_mm_xyz": [
            float(v)
            for v in np.asarray(
                prep.get(
                    "volume_size_mm_xyz",
                    np.asarray(current_grid_shape)
                    * np.asarray(spacing_eff),
                )
            ).reshape(-1)
        ],
        "measurements_generated_after_exact_resampling": bool(
            prep.get(
                "measurements_generated_after_exact_resampling",
                False,
            )
        ),
        "diffusion_prior_shape": (
            None
            if dp is None or getattr(dp, "volume_shape", None) is None
            else [int(v) for v in dp.volume_shape]
        ),
        "no_3d_resize_inside_conditional_diffusion": bool(
            dp is None
            or getattr(dp, "volume_shape", None) is None
            or current_grid_shape == tuple(dp.volume_shape)
        ),
    }
    with open(
        os.path.join(case_out, "direct_grid_metadata.json"),
        "w",
    ) as f:
        json.dump(grid_meta, f, indent=2)

    effective_reconstruction_seed = _stable_case_view_seed(
        reconstruction_seed,
        case_id,
        N_VIEWS,
    )
    if effective_reconstruction_seed is not None:
        random.seed(effective_reconstruction_seed)
        np.random.seed(effective_reconstruction_seed)
        torch.manual_seed(effective_reconstruction_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective_reconstruction_seed)
        print(
            "  reconstruction RNG : "
            f"base={int(reconstruction_seed)} | "
            f"case/view={effective_reconstruction_seed}"
        )

    shared_obs_mode_norm = normalize_shared_obs_mode(shared_obs_mode)
    shared_obs_dir_used = None
    shared_obs_action = "none"
    shared_obs_loaded = shared_obs_is_load(shared_obs_root, shared_obs_mode_norm)
    _shared_meta = {}
    shared_aux = {}
    xrays_2d = None
    soft_probs = None
    gt_proj_masks_2d = None

    # For a fair VH-initialization sweep, observation generation/loading must
    # not change the stochastic state seen by the conditional diffusion step.
    # Saving observations uses model initialization + Poisson sampling, while
    # loading them does not. Snapshot the RNG state here and restore it after
    # the observation branch whenever shared observations are being used.
    _shared_obs_preserve_rng = shared_obs_mode_norm != "none"
    if _shared_obs_preserve_rng:
        _python_rng_state = random.getstate()
        _np_rng_state = np.random.get_state()
        _torch_cpu_rng_state = torch.get_rng_state()
        _torch_cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )

    if shared_obs_loaded:

        # Shared observations are saved on the final reconstruction detector grid.
        # Therefore, if lowres_recon_det_shape is requested, rebuild the projector
        # before visual hull / data consistency and skip any local mask resizing.
        if lowres_recon_det_shape is not None:
            recon_det_shape = tuple(int(v) for v in lowres_recon_det_shape)
            if len(recon_det_shape) != 2:
                raise ValueError(f"lowres_recon_det_shape must have length 2, got {lowres_recon_det_shape}")
            if tuple(det_shape) != tuple(recon_det_shape):
                print(f"  [shared obs] load final detector masks; rebuild projector: {tuple(det_shape)} -> {tuple(recon_det_shape)}")
                pg, _ = build_cone_vec_from_world_sources(
                    S_mm,
                    O_mm,
                    d_OD_mm,
                    det_shape=recon_det_shape,
                    det_size=det_size_mm,
                    up_world=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    assume_tomosipo_zyx=True,
                )
                det_shape = recon_det_shape

        masks_2d, recon_masks_2d, mask_weights_2d, _shared_meta, shared_obs_dir_used = load_shared_observations(
            shared_obs_root=shared_obs_root,
            case_id=case_id,
            n_views=N_VIEWS,
        )
        validate_observation_target(_shared_meta, prep)
        shared_aux = load_shared_observation_auxiliaries(
            shared_obs_root=shared_obs_root,
            case_id=case_id,
            n_views=N_VIEWS,
            require_complete=bool(require_shared_obs_aux),
        )
        xrays_2d = shared_aux.get("xrays_2d")
        soft_probs = shared_aux.get("segmentation_probabilities_2d")
        gt_proj_masks_2d = shared_aux.get("gt_projection_masks_2d")
        shared_obs_action = "load"
        masks_2d_clean = [m.copy() for m in masks_2d]
        corrupt_info = {
            "corrupted_view_indices": [],
            "clean_view_indices": list(range(len(masks_2d))),
            "mode": "shared_observations",
            "corrupt_view_prob": 0.0,
            "severity": 0.0,
        }

        for v, y in enumerate(masks_2d):
            save_mask_png(os.path.join(views_out, f"mask_view_{v:03d}.png"), y)
        np.savez_compressed(
            os.path.join(case_out, "masks_2d.npz"),
            **{f"view_{i:03d}": masks_2d[i].astype(np.uint8) for i in range(len(masks_2d))}
        )
        np.savez_compressed(
            os.path.join(case_out, "recon_masks_2d_resized.npz"),
            **{f"view_{i:03d}": recon_masks_2d[i].astype(np.float32) for i in range(len(recon_masks_2d))}
        )
        np.savez_compressed(
            os.path.join(case_out, "mask_confidence_maps.npz"),
            **{f"view_{i:03d}": mask_weights_2d[i].astype(np.float32) for i in range(len(mask_weights_2d))}
        )

    else:
        seg_model = None
        seg_model_device = None

        if use_trained_segmentation:
            if soft_seg_ckpt is None:
                raise ValueError("use_trained_segmentation=True requires soft_seg_ckpt")
            if img_xyz is None:
                raise ValueError("prepare_case must provide img_xyz when using trained segmentation")

            seg_dev_str = soft_seg_device if soft_seg_device is not None else device
            seg_model, seg_model_device = load_soft_seg_model(
                ckpt_path=soft_seg_ckpt,
                device=seg_dev_str,
                base_ch=int(soft_seg_base_ch),
            )

        # ------------------------------------------------------------
        # 2D observations on full/current volume
        # ------------------------------------------------------------
        if use_trained_segmentation:
            seg_out = os.path.join(case_out, "segmentation_from_xray")
            ensure_dir(seg_out)

            xrays_2d = simulate_multiview_xrays_from_ct(
                img_xyz=img_xyz,
                mask_xyz=prep.get("mask_physics_xyz", mask_xyz),
                vg=vg,
                pg=pg,
                device=device,
                hu_min=float(xray_hu_min),
                hu_max=float(xray_hu_max),
                hu_scale=float(xray_hu_scale),
                delta_mu=float(xray_delta_mu),
                photon_count=float(xray_photon_count),
            )

            soft_masks_2d_clean = []
            hard_masks_2d_clean = []
            mask_weights_2d = []

            gt_proj_masks_2d = generate_multiview_masks_from_label(
                mask_zyx=mask_zyx.astype(np.float32),
                vg=vg,
                pg=pg,
                device=device,
                proj_eps=proj_eps,
            )
            gt_proj_masks_2d = [m.astype(np.uint8) for m in gt_proj_masks_2d]

            soft_probs = predict_soft_masks_from_xrays(
                seg_model,
                seg_model_device,
                xrays_2d,
                seg_input_hw=(256, 256),
            )

            for v, (xray_hw, prob_hw) in enumerate(zip(xrays_2d, soft_probs)):
                hard_hw = (prob_hw >= float(soft_seg_pred_thr)).astype(np.uint8)
                conf_hw = build_confidence_from_soft_mask(prob_hw, min_w=0.05)

                soft_masks_2d_clean.append(prob_hw.astype(np.float32))
                hard_masks_2d_clean.append(hard_hw.astype(np.uint8))
                mask_weights_2d.append(conf_hw.astype(np.float32))

                np.save(os.path.join(seg_out, f"xray_view_{v:03d}.npy"), xray_hw.astype(np.float32))
                np.save(os.path.join(seg_out, f"softmask_view_{v:03d}.npy"), prob_hw.astype(np.float32))

                save_gray_png(os.path.join(seg_out, f"xray_view_{v:03d}.png"), xray_hw)
                save_softmask_png(os.path.join(seg_out, f"softmask_view_{v:03d}.png"), prob_hw)
                save_mask_png(os.path.join(seg_out, f"hardmask_view_{v:03d}.png"), hard_hw)
                save_mask_png(os.path.join(seg_out, f"gtproj_view_{v:03d}.png"), gt_proj_masks_2d[v])

            masks_2d = [m.copy() for m in hard_masks_2d_clean]
            masks_2d_clean = [m.copy() for m in hard_masks_2d_clean]

            if use_soft_masks_for_recon:
                recon_masks_2d = [m.copy().astype(np.float32) for m in soft_masks_2d_clean]
            else:
                recon_masks_2d = [m.copy().astype(np.float32) for m in hard_masks_2d_clean]

            corrupt_info = {
                "corrupted_view_indices": [],
                "clean_view_indices": list(range(len(masks_2d))),
                "mode": "trained_segmentation",
                "corrupt_view_prob": 0.0,
                "severity": 0.0,
            }

            np.savez_compressed(
                os.path.join(case_out, "masks_2d_hard_from_seg.npz"),
                **{f"view_{i:03d}": masks_2d[i].astype(np.uint8) for i in range(len(masks_2d))}
            )

            np.savez_compressed(
                os.path.join(case_out, "masks_2d_soft_from_seg.npz"),
                **{f"view_{i:03d}": recon_masks_2d[i].astype(np.float32) for i in range(len(recon_masks_2d))}
            )

            np.savez_compressed(
                os.path.join(case_out, "mask_confidence_maps.npz"),
                **{f"view_{i:03d}": mask_weights_2d[i].astype(np.float32) for i in range(len(mask_weights_2d))}
            )

            for v, y in enumerate(masks_2d):
                save_mask_png(os.path.join(views_out, f"mask_view_{v:03d}.png"), y)

        else:
            masks_2d_clean = generate_multiview_masks_from_label(
                mask_zyx=mask_zyx.astype(np.float32),
                vg=vg,
                pg=pg,
                device=device,
                proj_eps=proj_eps,
            )
            masks_2d_clean = [m.astype(np.uint8) for m in masks_2d_clean]

            if seg_corrupt_severity > 0.0:
                masks_2d, corrupt_info = corrupt_multiview_masks(
                    masks_2d_clean,
                    severity=float(seg_corrupt_severity),
                    seed=int(seg_corrupt_seed),
                    corrupt_view_prob=float(seg_corrupt_view_prob),
                    min_corrupt_views=1,
                    mode=str(seg_corrupt_mode),
                    return_info=True,
                )

                ensure_dir(views_clean_out)
                for v, y in enumerate(masks_2d_clean):
                    save_mask_png(os.path.join(views_clean_out, f"mask_view_{v:03d}.png"), y)

                np.savez_compressed(
                    os.path.join(case_out, "masks_2d_clean.npz"),
                    **{f"view_{i:03d}": masks_2d_clean[i].astype(np.uint8) for i in range(len(masks_2d_clean))}
                )

                with open(os.path.join(case_out, "mask_corruption_info.json"), "w") as f:
                    json.dump(corrupt_info, f, indent=2)
            else:
                masks_2d = [m.copy() for m in masks_2d_clean]
                corrupt_info = {
                    "corrupted_view_indices": [],
                    "clean_view_indices": list(range(len(masks_2d_clean))),
                    "mode": seg_corrupt_mode,
                    "corrupt_view_prob": float(seg_corrupt_view_prob),
                    "severity": float(seg_corrupt_severity),
                }

            for v, y in enumerate(masks_2d):
                save_mask_png(os.path.join(views_out, f"mask_view_{v:03d}.png"), y)

            np.savez_compressed(
                os.path.join(case_out, "masks_2d.npz"),
                **{f"view_{i:03d}": masks_2d[i].astype(np.uint8) for i in range(len(masks_2d))}
            )

            recon_masks_2d = maybe_make_soft_masks(
                masks_2d=masks_2d,
                use_soft_masks=bool(use_soft_masks),
                soft_mask_sigma=float(soft_mask_sigma),
            )

            if seg_corrupt_severity > 0.0:
                mask_weights_2d = build_multiview_mask_confidence_maps(
                    recon_masks_2d,
                    boundary_band_px=3.0,
                    small_component_area=24,
                    small_component_weight=0.05,
                    fg_boundary_weight=0.45,
                    bg_boundary_weight=0.65,
                    min_weight=0.02,
                )
            else:
                mask_weights_2d = [np.ones_like(m, dtype=np.float32) for m in recon_masks_2d]

            np.savez_compressed(
                os.path.join(case_out, "mask_confidence_maps.npz"),
                **{f"view_{i:03d}": mask_weights_2d[i].astype(np.float32) for i in range(len(mask_weights_2d))}
            )

    if _shared_obs_preserve_rng:
        random.setstate(_python_rng_state)
        np.random.set_state(_np_rng_state)
        torch.set_rng_state(_torch_cpu_rng_state)
        if _torch_cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(_torch_cuda_rng_states)



    # ------------------------------------------------------------
    # reconstruction
    # ------------------------------------------------------------
    # Optional low-resolution detector masks for reconstruction only.
    # The segmentation/X-ray generation can stay at the original detector size,
    # while VH/Bayes/PnP/conditional diffusion use resized masks and a matching projector.
    if lowres_recon_det_shape is not None:
        recon_det_shape = tuple(int(v) for v in lowres_recon_det_shape)
        if len(recon_det_shape) != 2:
            raise ValueError(f"lowres_recon_det_shape must have length 2, got {lowres_recon_det_shape}")

        if tuple(det_shape) != tuple(recon_det_shape):
            print(
                f"  [low-res recon] resize masks and rebuild projector: "
                f"{tuple(det_shape)} -> {tuple(recon_det_shape)}"
            )

            masks_2d = resize_mask_list_to_shape(
                masks_2d,
                out_shape_hw=recon_det_shape,
                is_binary=True,
                binary_thr=0.05,
            )

            recon_masks_2d = resize_mask_list_to_shape(
                recon_masks_2d,
                out_shape_hw=recon_det_shape,
                is_binary=False,
            )

            mask_weights_2d = resize_mask_list_to_shape(
                mask_weights_2d,
                out_shape_hw=recon_det_shape,
                is_binary=False,
            )

            if xrays_2d is not None:
                xrays_2d = resize_mask_list_to_shape(
                    xrays_2d,
                    out_shape_hw=recon_det_shape,
                    is_binary=False,
                )
            if soft_probs is not None:
                soft_probs = resize_mask_list_to_shape(
                    soft_probs,
                    out_shape_hw=recon_det_shape,
                    is_binary=False,
                )
            if gt_proj_masks_2d is not None:
                gt_proj_masks_2d = resize_mask_list_to_shape(
                    gt_proj_masks_2d,
                    out_shape_hw=recon_det_shape,
                    is_binary=True,
                    binary_thr=0.5,
                )

            pg, _ = build_cone_vec_from_world_sources(
                S_mm,
                O_mm,
                d_OD_mm,
                det_shape=recon_det_shape,
                det_size=det_size_mm,
                up_world=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                assume_tomosipo_zyx=True,
            )
            det_shape = recon_det_shape

            np.savez_compressed(
                os.path.join(case_out, "recon_masks_2d_resized.npz"),
                **{f"view_{i:03d}": recon_masks_2d[i].astype(np.float32) for i in range(len(recon_masks_2d))}
            )

    if (not shared_obs_loaded) and shared_obs_should_save(shared_obs_root, shared_obs_mode_norm):
        shared_obs_dir_used = save_shared_observations(
            shared_obs_root=shared_obs_root,
            case_id=case_id,
            n_views=N_VIEWS,
            masks_2d=masks_2d,
            recon_masks_2d=recon_masks_2d,
            mask_weights_2d=mask_weights_2d,
            xrays_2d=xrays_2d,
            segmentation_probabilities_2d=soft_probs,
            gt_projection_masks_2d=gt_proj_masks_2d,
            metadata={
                "source_script": "pipeline_full128_prior_direct128_test50_disjoint.py",
                **observation_target_metadata(prep),
                "det_shape_hw": list(det_shape),
                "lowres_recon_det_shape": None if lowres_recon_det_shape is None else [int(v) for v in lowres_recon_det_shape],
                "use_trained_segmentation": bool(use_trained_segmentation),
                "use_soft_masks_for_recon": bool(use_soft_masks_for_recon),
                "soft_seg_ckpt": None if soft_seg_ckpt is None else str(soft_seg_ckpt),
                "soft_seg_base_ch": int(soft_seg_base_ch),
                "soft_seg_pred_thr": float(soft_seg_pred_thr),
                "xray_photon_count": float(xray_photon_count),
                "xray_hu_scale": float(xray_hu_scale),
                "xray_hu_min": float(xray_hu_min),
                "xray_hu_max": float(xray_hu_max),
                "xray_delta_mu": float(xray_delta_mu),
                "reconstruction_seed_base": (
                    None
                    if reconstruction_seed is None
                    else int(reconstruction_seed)
                ),
                "reconstruction_seed_case_view": (
                    None
                    if effective_reconstruction_seed is None
                    else int(effective_reconstruction_seed)
                ),
                "projection_geometry": {
                    "R_SO_MM": float(R_SO_MM),
                    "PLANE": str(PLANE),
                    "TILT_DEG": float(TILT_DEG),
                    "TILT_AXIS": str(TILT_AXIS),
                    "d_OD_mm": float(d_OD_mm),
                    "det_shape_hw": [int(v) for v in det_shape],
                    "det_size_mm": [float(v) for v in det_size_mm],
                },
            },
            overwrite=shared_obs_mode_norm in {"save", "overwrite"},
        )
        shared_obs_action = "save" if shared_obs_mode_norm in {"save", "overwrite"} else "save_if_missing"
        with open(os.path.join(shared_obs_dir_used, "metadata.json"), "r") as f:
            _shared_meta = json.load(f)

    if _shared_meta:
        validate_observation_target(_shared_meta, prep)
    if bool(require_shared_obs_aux):
        validate_shared_observation_metadata(
            _shared_meta,
            {
                "use_trained_segmentation": bool(use_trained_segmentation),
                "use_soft_masks_for_recon": bool(use_soft_masks_for_recon),
                "soft_seg_ckpt": (
                    None if soft_seg_ckpt is None else str(soft_seg_ckpt)
                ),
                "soft_seg_base_ch": int(soft_seg_base_ch),
                "soft_seg_pred_thr": float(soft_seg_pred_thr),
                "xray_photon_count": float(xray_photon_count),
                "xray_hu_scale": float(xray_hu_scale),
                "xray_hu_min": float(xray_hu_min),
                "xray_hu_max": float(xray_hu_max),
                "xray_delta_mu": float(xray_delta_mu),
                "reconstruction_seed_base": (
                    None
                    if reconstruction_seed is None
                    else int(reconstruction_seed)
                ),
                "reconstruction_seed_case_view": (
                    None
                    if effective_reconstruction_seed is None
                    else int(effective_reconstruction_seed)
                ),
                "projection_geometry": {
                    "R_SO_MM": float(R_SO_MM),
                    "PLANE": str(PLANE),
                    "TILT_DEG": float(TILT_DEG),
                    "TILT_AXIS": str(TILT_AXIS),
                    "d_OD_mm": float(d_OD_mm),
                    "det_shape_hw": [int(v) for v in det_shape],
                    "det_size_mm": [float(v) for v in det_size_mm],
                },
            },
        )

    segmentation_2d_quality = segmentation_quality_2d(
        masks_2d,
        gt_proj_masks_2d,
    )

    vh_zyx, votes_zyx = reconstruct_visual_hull_from_masks_tomosipo(
        masks_2d=masks_2d,
        pg=pg,
        vol_shape_zyx=mask_xyz.shape,
        spacing_zyx=spacing_zyx,
        allow_missing_views=allow_missing_views,
    )

    vh = vh_zyx.astype(np.uint8)
    votes = votes_zyx.astype(np.uint16)
    dice_vh, iou_vh = dice_iou(vh > 0, mask_xyz > 0)

    # Visual-hull probability-style initialization.
    # Used when --pnp_init_mode visual_hull / vh is selected.
    prob_zyx_vh_init = vh.astype(np.float32)

    # Visual-hull initialization before diffusion refinement.
    save_volume_views_xyz_with_contours(
        prob_zyx_vh_init,
        os.path.join(case_out, "visual_hull_prob_views.png"),
        title=f"{case_id} | N={N_VIEWS} | Visual hull initialization",
    )

    sil_scale_mm = float(np.min(spacing_zyx))
    A_op = ts.operator(vg, pg)

    valid_recon_modes = {"cond_diffusion"}
    
    if recon_mode not in valid_recon_modes:
        raise ValueError(f"Unknown recon_mode: {recon_mode}. Valid: {sorted(valid_recon_modes)}")

    guidance_mode = str(guidance_mode).strip().lower()
    if guidance_mode not in {"full", "prior_only"}:
        raise ValueError(
            f"Unknown guidance_mode={guidance_mode!r}. Valid: full, prior_only"
        )

    denoise_hist = []
    stage_prob_zyx_list = []
    stage_info = []
    measurement_guidance_evaluations = None
    measurement_guidance_updates = None
    initial_noised_state_sha256 = None
    initialization_noise_sha256 = None

    # ------------------------------------------------------------
    # Explicit bookkeeping for VH-based conditional initialization.
    # Important: the true visual-hull baseline Dice is dice_vh from
    # binary vh > 0 and must not depend on vh_soft_max.
    # These fields only describe the softened initialization map.
    # ------------------------------------------------------------
    cond_vh_soft_max_used = None
    cond_init_vh_support_dice = None
    cond_init_vh_support_iou = None
    cond_init_vh_thr010_dice = None
    cond_init_vh_thr010_iou = None



    if dp is None:
        raise ValueError("recon_mode='cond_diffusion' requires diffusion_prior_ckpt.")

    if guidance_mode == "full":
        print(
            "  [conditional diffusion] reverse diffusion with "
            "measurement-guided x0_hat correction"
        )
    else:
        print(
            "  [prior-only diffusion] reverse diffusion from the "
            "patient-specific initialization without measurement guidance"
        )
    mode = str(pnp_init_mode).lower()

    vh_soft_max = float(cond_vh_soft_max)
    if not (0.0 <= vh_soft_max <= 1.0):
        raise ValueError(
            f"cond_vh_soft_max must be in [0, 1], got {vh_soft_max}"
        )
    vh_bg = 0.0
    init_prob_cond = vh_bg + vh_soft_max * prob_zyx_vh_init.astype(np.float32)
    init_prob_cond = np.clip(init_prob_cond, 0.0, 1.0).astype(np.float32)

    cond_vh_soft_max_used = float(vh_soft_max)

    # Support Dice: independent of vh_soft_max as long as vh_soft_max > 0.
    init_support_mask = init_prob_cond > 0.0
    cond_init_vh_support_dice, cond_init_vh_support_iou = dice_iou(
        init_support_mask,
        mask_xyz > 0,
    )

    # Thresholded soft-init Dice: this can change with vh_soft_max.
    # It is saved separately so it is not confused with true VH Dice.
    init_thr010_mask = init_prob_cond >= 0.10
    cond_init_vh_thr010_dice, cond_init_vh_thr010_iou = dice_iou(
        init_thr010_mask,
        mask_xyz > 0,
    )

    print(
        f"  [cond VH init check] true VH Dice={dice_vh:.4f} | "
        f"vh_soft_max={cond_vh_soft_max_used:.3f} | "
        f"support Dice={cond_init_vh_support_dice:.4f} | "
        f"thr@0.10 Dice={cond_init_vh_thr010_dice:.4f}"
    )

    # This is x0-guided measurement conditioning: at each reverse step, the code
    # corrects the predicted clean image x0_hat using the silhouette data gradient,
    # then uses the corrected x0_hat in the DDPM posterior mean. It does not
    # backpropagate the measurement loss through the denoiser to obtain grad wrt x_t.
    cond_out = dp.conditioned_sample_from_measurements(
        prob_shape_zyx=mask_xyz.shape,
        masks_2d=recon_masks_2d,
        A_op=A_op,
        sil_scale_mm=sil_scale_mm,
        start_t=int(diffusion_denoise_t),
        init_prob_zyx=init_prob_cond,
        dc_step_size=float(cond_dc_step_size),
        dc_loss_type=str(diffusion_dc_loss_type),
        grad_normalize=True,
        stride_zyx=tuple(max(1, p // 4) for p in dp.patch_size),
        center_keep_size_zyx=tuple(max(1, int(round(p * 0.80))) for p in dp.patch_size),
        clamp_x=True,
        guidance_inner_steps=int(guidance_inner_steps),
        n_stage_snapshots=int(diffusion_n_stage_snapshots),
        mask_weights_2d=mask_weights_2d,
        lambda_tv_cond=float(lambda_tv_cond),
        tv_mode=str(tv_mode),
        tv_grad_normalize=bool(tv_grad_normalize),
        alpha_mode=str(alpha_mode),
        init_method=str(cond_init_method),
        guidance_mode=str(guidance_mode),
    )

    prob_zyx = cond_out["prob_zyx"].astype(np.float32)
    denoise_hist = cond_out["history"]
    stage_prob_zyx_list = cond_out.get("stage_prob_zyx_list", [])
    stage_info = cond_out.get("stage_info", [])
    measurement_guidance_evaluations = int(
        cond_out["measurement_guidance_evaluations"]
    )
    measurement_guidance_updates = int(
        cond_out["measurement_guidance_updates"]
    )
    initial_noised_state_sha256 = cond_out["initial_noised_state_sha256"]
    initialization_noise_sha256 = cond_out["initialization_noise_sha256"]

    if init_prob_cond is not None:
        nib.save(
            nib.Nifti1Image(init_prob_cond.astype(np.float32), aff_local),
            os.path.join(case_out, "cond_diffusion_init_prob_map.nii.gz"),
        )

    nib.save(
        nib.Nifti1Image(prob_zyx.astype(np.float32), aff_local),
        os.path.join(case_out, "cond_diffusion_final_prob_map.nii.gz"),
    )

    prob_xyz = prob_zyx.astype(np.float32)

    view_analysis_dir = os.path.join(case_out, "view_consistency_analysis")
    view_ranking = analyze_mask_view_inconsistency(
        prob_zyx=prob_zyx,
        masks_2d=recon_masks_2d,
        A_op=A_op,
        sil_scale_mm=sil_scale_mm,
        device=device,
        out_dir=view_analysis_dir,
        prefix="final_reconstruction",
        alpha_mode=str(alpha_mode),
    )

    if len(stage_prob_zyx_list) > 0:
        np.savez_compressed(
            os.path.join(case_out, "diffusion_stage_prob_maps.npz"),
            **{f"stage_{i:02d}": stage_prob_zyx_list[i].astype(np.float32) for i in range(len(stage_prob_zyx_list))}
        )
        with open(os.path.join(case_out, "diffusion_stage_info.json"), "w") as f:
            json.dump(stage_info, f, indent=2)

        stage_prob_xyz_list = [p.astype(np.float32) for p in stage_prob_zyx_list]
        stage_titles = []
        for i, info in enumerate(stage_info):
            tag = info.get("tag", f"stage_{i}")
            if tag == "init_prior":
                ttl = (
                    "Stage 0\nGuided init"
                    if guidance_mode == "full"
                    else "Stage 0\nPrior-only init"
                )
            elif tag == "initial_noise_state":
                ttl = "Stage 0\nInitial noise"
            elif tag == "final":
                ttl = "Final reconstruction"
            else:
                ttl = f"Stage {i}\n{tag}"
            stage_titles.append(ttl)

        plot_diffusion_stage_evolution_3d(
            stage_prob_xyz_list=stage_prob_xyz_list,
            out_png=os.path.join(case_out, "diffusion_probability_evolution_4stages.png"),
            stage_titles=stage_titles,
            main_title=f"{recon_mode}: {case_id}",
            prob_thr=0.05,
            max_points_per_panel=12000,
            seed=0,
            elev=25.0,
            azim=45.0,
        )

    cond_title = (
        "measurement-guided diffusion sample"
        if guidance_mode == "full"
        else "prior-only diffusion reconstruction"
    )
    save_volume_views_xyz_with_contours(
        prob_xyz,
        os.path.join(case_out, "cond_diffusion_prob_views.png"),
        title=f"{case_id} | N={N_VIEWS} | {cond_title}",
    )


    probability_map_title = "Posterior probability map"
    if recon_mode == "cond_diffusion":
        probability_map_title = (
            "Measurement-guided probability map"
            if guidance_mode == "full"
            else "Prior-only probability map"
        )
    save_volume_views_xyz_with_contours(
        prob_xyz,
        os.path.join(case_out, "bayes_prob_views.png"),
        title=f"{case_id} | N={N_VIEWS} | {probability_map_title}",
    )

    final_eval = evaluate_prob_map_against_gt(
        prob_xyz=prob_xyz,
        gt_xyz=mask_xyz,
        n_views=N_VIEWS,
        default_thr=0.50,
        min_vox_component=min_vox_component,
        recon_mode=recon_mode,
        threshold_mode=eval_threshold_mode,
        fixed_threshold=eval_fixed_thr,
    )

    bayes_thr_eff = final_eval["thr_eff"]
    bayes_fixed_xyz = final_eval["fixed_mask_xyz"]
    dice_bf = final_eval["dice_fixed"]
    iou_bf = final_eval["iou_fixed"]
    bayes_oracle_xyz = final_eval["oracle_mask_xyz"]
    dice_bo = final_eval["dice_oracle"]
    iou_bo = final_eval["iou_oracle"]
    thr_bo = final_eval["thr_oracle"]
    auprc_final = average_precision_binary(mask_xyz > 0, prob_xyz)
    cldice_final = cldice_binary(bayes_fixed_xyz > 0, mask_xyz > 0)

    # Compare the target, visual-hull initialization, and final reconstruction.
    method_name_map = {"cond_diffusion": "Conditional diffusion"}
    method_name = method_name_map.get(str(recon_mode), str(recon_mode))
    if recon_mode == "cond_diffusion" and guidance_mode == "prior_only":
        method_name = "Prior-only diffusion"

    if init_prob_cond is not None:
        bayes_prob_for_plot = init_prob_cond.astype(np.float32)
        bayes_fixed_for_plot = posterior_to_binary_map(
            prob_xyz=bayes_prob_for_plot,
            thr=0.05,
            min_vox_component=min_vox_component,
        ).astype(np.uint8)
    else:
        bayes_prob_for_plot = np.zeros_like(prob_xyz, dtype=np.float32)
        bayes_fixed_for_plot = np.zeros_like(mask_xyz, dtype=np.uint8)

    final_fixed_xyz = final_eval["fixed_mask_xyz"].astype(np.uint8)

    plot_gt_bayes_and_final_reconstruction_3d(
        gt_xyz=mask_xyz,
        bayes_prob_xyz=bayes_prob_for_plot,
        bayes_fixed_xyz=bayes_fixed_for_plot,
        final_prob_xyz=prob_xyz,
        final_fixed_xyz=final_fixed_xyz,
        out_png=os.path.join(case_out, f"gt_bayes_and_{recon_mode}_3d_overlay.png"),
        prob_thr=0.10,
        max_points_per_panel=30000,
        seed=0,
        elev=25.0,
        azim=45.0,
        final_method_name=method_name,
    )



    print(
        f"  Final method ({recon_mode}) fixed={dice_bf:.4f} (thr={bayes_thr_eff:.3f}) | "
        f"Final method ({recon_mode}) oracle={dice_bo:.4f} (thr={thr_bo:.3f})"
    )

    nib.save(nib.Nifti1Image(mask_xyz.astype(np.uint8), aff_local), os.path.join(case_out, "gt_mask.nii.gz"))
    nib.save(nib.Nifti1Image(vh.astype(np.uint8), aff_local), os.path.join(case_out, "visual_hull_mask.nii.gz"))
    nib.save(nib.Nifti1Image(votes.astype(np.uint16), aff_local), os.path.join(case_out, "visual_hull_votes.nii.gz"))
    # Legacy filenames are retained for the existing saved-result readers.
    nib.save(nib.Nifti1Image(prob_xyz.astype(np.float32), aff_local), os.path.join(case_out, "bayes_prob_map.nii.gz"))
    nib.save(nib.Nifti1Image(bayes_fixed_xyz.astype(np.uint8), aff_local), os.path.join(case_out, "bayes_fixed_mask.nii.gz"))
    nib.save(nib.Nifti1Image(bayes_oracle_xyz.astype(np.uint8), aff_local), os.path.join(case_out, "bayes_oracle_mask.nii.gz"))

    with open(os.path.join(case_out, "hybrid_denoise_history.json"), "w") as f:
        json.dump(denoise_hist, f, indent=2)

    gt_verts, gt_faces = mesh_from_binary(mask_xyz.astype(np.uint8), spacing_eff)
    vh_verts, vh_faces = mesh_from_binary(vh.astype(np.uint8), spacing_eff)
    bf_verts, bf_faces = mesh_from_binary(bayes_fixed_xyz.astype(np.uint8), spacing_eff)

    if gt_verts is not None and gt_faces is not None:
        write_ply_mesh(os.path.join(case_out, "gt_mesh.ply"), gt_verts, gt_faces)

    if vh_verts is not None and vh_faces is not None:
        write_ply_mesh(os.path.join(case_out, "visual_hull_mesh.ply"), vh_verts, vh_faces)
        pts = sample_points_from_mesh(vh_verts, vh_faces, n_points=n_points, seed=0)
        write_ply_points(os.path.join(case_out, "visual_hull_points.ply"), pts)

    if bf_verts is not None and bf_faces is not None:
        write_ply_mesh(os.path.join(case_out, "bayes_fixed_mesh.ply"), bf_verts, bf_faces)
        pts = sample_points_from_mesh(bf_verts, bf_faces, n_points=n_points, seed=0)
        write_ply_points(os.path.join(case_out, "bayes_fixed_points.ply"), pts)

    # Hash the checkpoints that were loaded for this reconstruction.
    diffusion_checkpoint_sha256 = (
        _file_sha256(diffusion_prior_ckpt) if dp is not None else None
    )
    segmentation_checkpoint_sha256 = (
        _file_sha256(soft_seg_ckpt) if use_trained_segmentation else None
    )

    meta = {
        "case_id": case_id,
        "label_path": lbl_path,
        "img_path": img_path,
        "shape_local_xyz": [int(x) for x in mask_xyz.shape],
        "mask_preprocessing": prep.get("mask_preprocessing", "legacy"),
        "native_component_filter": prep.get("prepare_min_vox_component"),
        "shape_local_zyx": [int(x) for x in mask_xyz.shape],
        "exact_grid_applied": bool(
            prep.get("exact_grid_applied", False)
        ),
        "exact_grid_shape_xyz": prep.get(
            "exact_grid_shape_xyz",
            None,
        ),
        "exact_mask_resample_mode": str(
            prep.get("exact_mask_resample_mode", "none")
        ),
        "measurements_generated_after_exact_resampling": bool(
            prep.get(
                "measurements_generated_after_exact_resampling",
                False,
            )
        ),
        "diffusion_prior_volume_shape": (
            None
            if dp is None or getattr(dp, "volume_shape", None) is None
            else [int(v) for v in dp.volume_shape]
        ),
        "no_3d_resize_inside_conditional_diffusion": bool(
            dp is None
            or getattr(dp, "volume_shape", None) is None
            or tuple(mask_xyz.shape) == tuple(dp.volume_shape)
        ),
        "spacing_eff_xyz_mm": [float(x) for x in spacing_eff],
        "spacing_eff_zyx_mm": [float(x) for x in spacing_zyx],
        "bbox_min_xyz": [int(x) for x in bbox_min],
        "bbox_max_xyz": [int(x) for x in bbox_max],
        "scale": float(scale),
        "N_VIEWS": int(N_VIEWS),
        "dice_visual_hull_vs_gt": float(dice_vh),
        "iou_visual_hull_vs_gt": float(iou_vh),
        "visual_hull_array_sha256": _array_sha256(vh.astype(np.uint8)),
        "segmentation_2d_mean_dice": (
            None
            if segmentation_2d_quality is None
            else float(segmentation_2d_quality["mean_dice"])
        ),
        "segmentation_2d_mean_iou": (
            None
            if segmentation_2d_quality is None
            else float(segmentation_2d_quality["mean_iou"])
        ),
        "segmentation_2d_per_view": (
            None
            if segmentation_2d_quality is None
            else segmentation_2d_quality["per_view"]
        ),


        # Current/final output metrics for whichever recon_mode is running.
        "dice_current_output_fixed_vs_gt": float(dice_bf),
        "iou_current_output_fixed_vs_gt": float(iou_bf),
        "current_output_fixed_thr": float(bayes_thr_eff),
        "cldice_current_output_fixed_vs_gt": float(cldice_final),
        "auprc_current_output_continuous_vs_gt": float(auprc_final),
        "dice_current_output_oracle_vs_gt": float(dice_bo),
        "iou_current_output_oracle_vs_gt": float(iou_bo),
        "current_output_oracle_thr": float(thr_bo),

        # Conditional-diffusion VH initialization diagnostics.
        # The true VH baseline is dice_visual_hull_vs_gt above.
        "cond_vh_soft_max": cond_vh_soft_max_used,
        "cond_init_vh_support_dice_vs_gt": (
            float(cond_init_vh_support_dice)
            if cond_init_vh_support_dice is not None else None
        ),
        "cond_init_vh_support_iou_vs_gt": (
            float(cond_init_vh_support_iou)
            if cond_init_vh_support_iou is not None else None
        ),
        "cond_init_vh_thr010_dice_vs_gt": (
            float(cond_init_vh_thr010_dice)
            if cond_init_vh_thr010_dice is not None else None
        ),
        "cond_init_vh_thr010_iou_vs_gt": (
            float(cond_init_vh_thr010_iou)
            if cond_init_vh_thr010_iou is not None else None
        ),
        "diffusion_denoise_t": int(diffusion_denoise_t),
        "diffusion_n_averages": int(diffusion_n_averages),
        "diffusion_n_stage_snapshots": int(diffusion_n_stage_snapshots),
        "diffusion_prior_ckpt": (
            None if diffusion_prior_ckpt is None else str(diffusion_prior_ckpt)
        ),
        "diffusion_prior_ckpt_sha256": diffusion_checkpoint_sha256,
        "diffusion_prior_device": (
            None if diffusion_prior_device is None else str(diffusion_prior_device)
        ),
        "cond_dc_step_size": float(cond_dc_step_size),
        "alpha_mode": str(alpha_mode),
        "pnp_init_mode": str(pnp_init_mode),
        "guidance_inner_steps": int(guidance_inner_steps),
        "guidance_mode": (
            str(guidance_mode) if recon_mode == "cond_diffusion" else None
        ),
        "measurement_guidance_evaluations": measurement_guidance_evaluations,
        "measurement_guidance_updates": measurement_guidance_updates,
        "initial_noised_state_sha256": initial_noised_state_sha256,
        "initialization_noise_sha256": initialization_noise_sha256,
        "diffusion_dc_loss_type": str(diffusion_dc_loss_type),
        "lambda_tv_cond": float(lambda_tv_cond),
        "tv_mode": str(tv_mode),
        "tv_grad_normalize": bool(tv_grad_normalize),
        "cond_init_mode": str(pnp_init_mode) if recon_mode == "cond_diffusion" else None,
        "conditioning_type": (
            "x0_guided_measurement_correction"
            if recon_mode == "cond_diffusion" and guidance_mode == "full"
            else (
                "prior_only_no_measurement_update"
                if recon_mode == "cond_diffusion"
                else None
            )
        ),
        "output_dir": case_out,
        "final_probability_map_path": (
            os.path.join(case_out, "cond_diffusion_final_prob_map.nii.gz")
            if recon_mode == "cond_diffusion"
            else os.path.join(case_out, "bayes_prob_map.nii.gz")
        ),
        "final_probability_array_sha256": _array_sha256(
            prob_xyz.astype(np.float32)
        ),
        "R_SO_MM": float(R_SO_MM),
        "PLANE": str(PLANE),
        "TILT_DEG": float(TILT_DEG),
        "TILT_AXIS": str(TILT_AXIS),
        "d_OD_mm": float(d_OD_mm),
        "det_shape": [int(v) for v in det_shape],
        "det_size_mm": [float(v) for v in det_size_mm],
        "proj_eps": float(proj_eps),
        "bp_eps": float(bp_eps),
        "allow_missing_views": int(allow_missing_views),
        "min_vox_component": int(min_vox_component),
        "volume_grid_shape_xyz": [int(v) for v in mask_xyz.shape],
        "direct_common_grid": bool(
            prep.get("exact_grid_applied", False)
            and tuple(int(v) for v in mask_xyz.shape) == (128, 128, 128)
        ),
        "seg_corrupt_severity": float(seg_corrupt_severity),
        "seg_corrupt_seed": int(seg_corrupt_seed),
        "seg_corrupt_view_prob": float(seg_corrupt_view_prob),
        "seg_corrupt_mode": str(seg_corrupt_mode),
        "corrupted_view_indices": corrupt_info["corrupted_view_indices"],
        "recon_mode": str(recon_mode),
        "most_suspicious_views": [int(r["view_index"]) for r in view_ranking[:5]],
        "dice_final_method_fixed_vs_gt": float(dice_bf),
        "iou_final_method_fixed_vs_gt": float(iou_bf),
        "final_method_fixed_thr": float(bayes_thr_eff),
        "cldice_final_method_fixed_vs_gt": float(cldice_final),
        "auprc_final_continuous_vs_gt": float(auprc_final),
        "auprc_definition": (
            "full-grid non-interpolated average precision; tied scores grouped"
        ),
        "cldice_definition": "3-D voxel skeleton clDice on the fixed-threshold mask",
        "dice_final_method_oracle_vs_gt": float(dice_bo),
        "iou_final_method_oracle_vs_gt": float(iou_bo),
        "final_method_oracle_thr": float(thr_bo),
        "eval_threshold_mode": str(final_eval["threshold_mode"]),
        "eval_threshold_mode_requested": str(eval_threshold_mode),
        "eval_fixed_thr_requested": float(eval_fixed_thr),
        "use_soft_masks": bool(use_soft_masks),
        "soft_mask_sigma": float(soft_mask_sigma),
        "use_trained_segmentation": bool(use_trained_segmentation),
        "soft_seg_ckpt": str(soft_seg_ckpt) if soft_seg_ckpt is not None else None,
        "soft_seg_ckpt_sha256": segmentation_checkpoint_sha256,
        "soft_seg_device": (
            None if soft_seg_device is None else str(soft_seg_device)
        ),
        "soft_seg_base_ch": int(soft_seg_base_ch),
        "soft_seg_pred_thr": float(soft_seg_pred_thr),
        "xray_photon_count": float(xray_photon_count),
        "xray_hu_scale": float(xray_hu_scale),
        "xray_hu_min": float(xray_hu_min),
        "xray_hu_max": float(xray_hu_max),
        "xray_delta_mu": float(xray_delta_mu),
        "use_soft_masks_for_recon": bool(use_soft_masks_for_recon),
        "lowres_recon_det_shape": None if lowres_recon_det_shape is None else [int(v) for v in lowres_recon_det_shape],
        "shared_obs_root": None if shared_obs_root in [None, "", "None"] else str(shared_obs_root),
        "shared_obs_mode": str(shared_obs_mode_norm),
        "shared_obs_action": str(shared_obs_action),
        "shared_obs_dir": shared_obs_dir_used,
        "shared_observation_hashes": {
            key: str(_shared_meta[key])
            for key in (
                "masks_2d_sha256",
                "recon_masks_2d_sha256",
                "mask_weights_2d_sha256",
                "xrays_2d_sha256",
                "segmentation_probabilities_2d_sha256",
                "gt_projection_masks_2d_sha256",
            )
            if key in _shared_meta
        },
        "require_shared_obs_aux": bool(require_shared_obs_aux),
        "reconstruction_seed_base": (
            None if reconstruction_seed is None else int(reconstruction_seed)
        ),
        "reconstruction_seed_case_view": (
            None
            if effective_reconstruction_seed is None
            else int(effective_reconstruction_seed)
        ),
        "cond_init_method": str(cond_init_method),
        "reconstruction_parameters": {
            "recon_mode": str(recon_mode),
            "device": str(device),
            "geometry": {
                "R_SO_MM": float(R_SO_MM),
                "PLANE": str(PLANE),
                "TILT_DEG": float(TILT_DEG),
                "TILT_AXIS": str(TILT_AXIS),
                "d_OD_mm": float(d_OD_mm),
                "det_shape_hw": [int(v) for v in det_shape],
                "det_size_mm": [float(v) for v in det_size_mm],
                "proj_eps": float(proj_eps),
                "bp_eps": float(bp_eps),
                "allow_missing_views": int(allow_missing_views),
            },
            "grid": {
                "shape_xyz": [int(v) for v in mask_xyz.shape],
                "mask_resample_mode": str(exact_mask_resample_mode),
                "enforce_diffusion_grid_match": bool(enforce_diffusion_grid_match),
                "bbox_pad": int(prep.get("bbox_pad", 0)),
                "min_vox_component": int(min_vox_component),
                "lowres_recon_det_shape": (
                    None
                    if lowres_recon_det_shape is None
                    else [int(v) for v in lowres_recon_det_shape]
                ),
            },
            "xray": {
                "photon_count": float(xray_photon_count),
                "hu_scale": float(xray_hu_scale),
                "hu_min": float(xray_hu_min),
                "hu_max": float(xray_hu_max),
                "delta_mu": float(xray_delta_mu),
                "attenuation_model": "windowed_hu_background_plus_delta_mu_vessel",
                "line_integral_clamp_min": 0.0,
                "transmission_model": "exp(-line_integral)",
                "noise_model": "poisson_photon_counts",
                "photon_count_clamp_min": 1.0,
                "output_normalization": "per_view_minmax_0_1",
            },
            "segmentation": {
                "use_trained": bool(use_trained_segmentation),
                "checkpoint": (
                    None if soft_seg_ckpt is None else str(soft_seg_ckpt)
                ),
                "checkpoint_sha256": segmentation_checkpoint_sha256,
                "device": (
                    None if soft_seg_device is None else str(soft_seg_device)
                ),
                "base_channels": int(soft_seg_base_ch),
                "input_shape_hw": [256, 256] if use_trained_segmentation else None,
                "input_normalization": (
                    "per_view_minmax_0_1" if use_trained_segmentation else None
                ),
                "output_activation": "sigmoid" if use_trained_segmentation else None,
                "prediction_threshold": float(soft_seg_pred_thr),
                "confidence_weight_formula": (
                    "clip(abs(2*p-1),0.05,1.0)"
                    if use_trained_segmentation
                    else None
                ),
                "confidence_min_weight": 0.05 if use_trained_segmentation else None,
                "use_soft_masks_for_recon": bool(use_soft_masks_for_recon),
                "corruption_severity": float(seg_corrupt_severity),
                "corruption_seed": int(seg_corrupt_seed),
                "corrupt_view_probability": float(seg_corrupt_view_prob),
                "corruption_mode": str(seg_corrupt_mode),
            },
            "conditional_diffusion": {
                "checkpoint": (
                    None
                    if diffusion_prior_ckpt is None
                    else str(diffusion_prior_ckpt)
                ),
                "checkpoint_sha256": diffusion_checkpoint_sha256,
                "device": (
                    None
                    if diffusion_prior_device is None
                    else str(diffusion_prior_device)
                ),
                "start_t": int(diffusion_denoise_t),
                "n_averages": int(diffusion_n_averages),
                "n_stage_snapshots": int(diffusion_n_stage_snapshots),
                "init_mode": str(pnp_init_mode),
                "init_method": str(cond_init_method),
                "vh_soft_max": float(cond_vh_soft_max),
                "dc_step_size": float(cond_dc_step_size),
                "dc_loss_type": str(diffusion_dc_loss_type),
                "guidance_inner_steps": int(guidance_inner_steps),
                "alpha_mode": str(alpha_mode),
                "lambda_tv": float(lambda_tv_cond),
                "tv_mode": str(tv_mode),
                "tv_grad_normalize": bool(tv_grad_normalize),
                "measurement_grad_normalize": (
                    True if recon_mode == "cond_diffusion" else None
                ),
                "clamp_x": True if recon_mode == "cond_diffusion" else None,
                "patch_size_zyx": (
                    None
                    if recon_mode != "cond_diffusion" or dp is None
                    else [int(p) for p in dp.patch_size]
                ),
                "stride_zyx": (
                    None
                    if recon_mode != "cond_diffusion" or dp is None
                    else [int(max(1, p // 4)) for p in dp.patch_size]
                ),
                "center_keep_fraction": (
                    0.8 if recon_mode == "cond_diffusion" else None
                ),
                "center_keep_size_zyx": (
                    None
                    if recon_mode != "cond_diffusion" or dp is None
                    else [int(max(1, round(p * 0.8))) for p in dp.patch_size]
                ),
            },
            "evaluation": {
                "threshold_mode": str(final_eval["threshold_mode"]),
                "threshold_mode_requested": str(eval_threshold_mode),
                "fixed_threshold": float(eval_fixed_thr),
                "min_vox_component": int(min_vox_component),
            },
            "randomness": {
                "base_seed": (
                    None if reconstruction_seed is None else int(reconstruction_seed)
                ),
                "case_view_seed": (
                    None
                    if effective_reconstruction_seed is None
                    else int(effective_reconstruction_seed)
                ),
            },
            "shared_observations": {
                "root": (
                    None
                    if shared_obs_root in [None, "", "None"]
                    else str(shared_obs_root)
                ),
                "mode": str(shared_obs_mode_norm),
                "action": str(shared_obs_action),
                "directory": shared_obs_dir_used,
                "require_auxiliary_arrays": bool(require_shared_obs_aux),
            },
        },
        }

    with open(os.path.join(case_out, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ok] saved to {case_out}")
    return meta

# ============================================================
# Plotting / summary
# ============================================================

def save_summary_csv(results, csv_path):
    fieldnames = [
        "case_id",
        "N_VIEWS",
        "dice_visual_hull_vs_gt",
        "iou_visual_hull_vs_gt",
        "dice_current_output_fixed_vs_gt",
        "iou_current_output_fixed_vs_gt",
        "current_output_fixed_thr",
        "diffusion_denoise_t",
        "cond_dc_step_size",
        "alpha_mode",
        "cond_vh_soft_max",
        "cond_init_vh_support_dice_vs_gt",
        "cond_init_vh_thr010_dice_vs_gt",
        "output_dir",
        "label_path",
        "R_SO_MM",
        "PLANE",
        "TILT_DEG",
        "TILT_AXIS",
        "d_OD_mm",
        "det_shape",
        "det_size_mm",
        "volume_grid_shape_xyz",
        "direct_common_grid",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, None) for k in fieldnames})


# ============================================================
# Angle sweep
# ============================================================

def find_img_in_raw_roots(case_num, raw_roots):
    if raw_roots is None or len(raw_roots) == 0:
        raise FileNotFoundError(f"No raw_roots provided for case {case_num}")

    fname = f"{case_num}.img.nii.gz"
    for root in raw_roots:
        p = os.path.join(root, fname)
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Missing matching CT image for case {case_num} in raw_roots={raw_roots}"
    )



def select_diffusion_test_subset_paths(
    label_paths, subset_size=50, subset_offset=0, split_seed=0,
    train_frac=0.70, val_frac=0.15, test_frac=0.15,
    start_index=1, max_cases=2000, split_json=None,
):
    """Select actual saved test members; legacy split-generation arguments are unused."""
    return select_saved_test_paths(label_paths, split_json, subset_size, subset_offset)


def run_angle_sweep(
    data_dir="slicer_full_pipeline/split_masks",
    raw_roots=None,
    out_dir="vh_tomosipo_sweep",
    case_id=(1, 2, 3, 4, 5, 6, 7),
    angle_list=(1, 2, 4, 8, 16, 32, 64),
    R_SO_MM=400.0,
    PLANE="yz",
    TILT_DEG=0.0,
    TILT_AXIS="x",
    det_shape=(512, 512),
    det_size_mm=(300.0, 300.0),
    d_OD_mm=400.0,
    max_dim=192,
    bbox_pad=10,
    min_vox_component=50,
    proj_eps=1e-8,
    bp_eps=1e-8,
    allow_missing_views=0,
    n_points=20000,
    device="cuda",
    diffusion_prior_ckpt=None,
    diffusion_prior_device=None,
    diffusion_denoise_t=15,
    diffusion_n_averages=1,
    seg_corrupt_severity=0.0,
    seg_corrupt_seed=0,
    seg_corrupt_view_prob=0.35,
    seg_corrupt_mode="sparse_like_figure",
    recon_mode="cond_diffusion",
    guidance_mode="full",
    guidance_inner_steps=1,
    diffusion_n_stage_snapshots=4,
    diffusion_dc_loss_type="bce",
    lambda_tv_cond=0.0,
    tv_mode="anisotropic",
    tv_grad_normalize=True,
    pnp_init_mode="vh",
    use_soft_masks=False,
    soft_mask_sigma=1.0,
    use_trained_segmentation=False,
    soft_seg_ckpt=None,
    soft_seg_device=None,
    soft_seg_base_ch=16,
    soft_seg_pred_thr=0.5,
    xray_photon_count=50000,
    xray_hu_scale=5e-3,
    xray_hu_min=-1000.0,
    xray_hu_max=3000.0,
    use_soft_masks_for_recon=True,
    xray_delta_mu=0.008,
    cond_dc_step_size=0.03,
    cond_vh_soft_max=0.30,
    alpha_mode="prob",
    cond_init_method="q_sample",
    lowres_recon_det_shape=None,
    shared_obs_root=None,
    shared_obs_mode="none",
    exact_grid_shape_xyz=(128, 128, 128),
    exact_mask_resample_mode="nearest",
    mask_preprocessing="legacy",
    enforce_diffusion_grid_match=True,
    use_diffusion_test_subset=False,
    diffusion_test_subset_size=50,
    diffusion_test_subset_offset=0,
    diffusion_split_seed=0,
    diffusion_split_json=None,
    diffusion_train_frac=0.70,
    diffusion_val_frac=0.15,
    diffusion_test_frac=0.15,
    diffusion_split_start_index=1,
    diffusion_split_max_cases=2000,
    eval_threshold_mode="auto",
    eval_fixed_thr=0.05,
    reconstruction_seed=None,
    require_shared_obs_aux=False,
):
    ensure_dir(out_dir)

    label_paths = sorted(
        glob.glob(os.path.join(data_dir, "*.label.nii.gz")),
        key=split_mask_sort_key,
    )
    if len(label_paths) == 0:
        raise RuntimeError(f"No label files found in: {data_dir}")

    diffusion_split_meta = None
    if bool(use_diffusion_test_subset):
        label_paths, diffusion_split_meta = select_diffusion_test_subset_paths(
            label_paths=label_paths,
            subset_size=int(diffusion_test_subset_size),
            subset_offset=int(diffusion_test_subset_offset),
            split_seed=int(diffusion_split_seed),
            split_json=diffusion_split_json,
            train_frac=float(diffusion_train_frac),
            val_frac=float(diffusion_val_frac),
            test_frac=float(diffusion_test_frac),
            start_index=int(diffusion_split_start_index),
            max_cases=int(diffusion_split_max_cases),
        )

        manifest = {
            **diffusion_split_meta,
            "purpose": "conditional-diffusion method-comparison cohort",
            "selected_test_phantoms": [split_mask_stem(p) for p in label_paths],
            "selected_test_label_paths": [str(p) for p in label_paths],
            "run_config": {
                "angle_list": [int(v) for v in angle_list],
                "volume_grid_shape": (
                    None
                    if exact_grid_shape_xyz is None
                    else [int(v) for v in exact_grid_shape_xyz]
                ),
                "volume_mask_resample_mode": str(exact_mask_resample_mode),
                "mask_preprocessing": str(mask_preprocessing),
                "eval_threshold_mode": (
                    str(eval_threshold_mode)
                ),
                "eval_fixed_thr": float(eval_fixed_thr),
                "max_dim": int(max_dim),
                "bbox_pad": int(bbox_pad),
                "det_shape_hw": [int(v) for v in det_shape],
                "det_size_mm": [float(v) for v in det_size_mm],
                "R_SO_MM": float(R_SO_MM),
                "PLANE": str(PLANE),
                "TILT_DEG": float(TILT_DEG),
                "TILT_AXIS": str(TILT_AXIS),
                "d_OD_mm": float(d_OD_mm),
                "xray_photon_count": float(xray_photon_count),
                "xray_hu_scale": float(xray_hu_scale),
                "xray_delta_mu": float(xray_delta_mu),
                "use_soft_masks_for_recon": bool(use_soft_masks_for_recon),
                "shared_obs_mode": str(normalize_shared_obs_mode(shared_obs_mode)),
                "recon_mode": str(recon_mode),
                "guidance_mode": str(guidance_mode),
                "pnp_init_mode": str(pnp_init_mode),
                "cond_vh_soft_max": float(cond_vh_soft_max),
                "cond_init_method": str(cond_init_method),
                "diffusion_denoise_t": int(diffusion_denoise_t),
                "cond_dc_step_size": float(cond_dc_step_size),
                "diffusion_n_averages": int(diffusion_n_averages),
                "diffusion_dc_loss_type": str(diffusion_dc_loss_type),
                "guidance_inner_steps": int(guidance_inner_steps),
                "lambda_tv_cond": float(lambda_tv_cond),
                "tv_mode": str(tv_mode),
                "tv_grad_normalize": bool(tv_grad_normalize),
                "alpha_mode": str(alpha_mode),
                "diffusion_prior_ckpt": (
                    None if diffusion_prior_ckpt is None else str(diffusion_prior_ckpt)
                ),
                "reconstruction_seed": (
                    None if reconstruction_seed is None else int(reconstruction_seed)
                ),
                "require_shared_obs_aux": bool(require_shared_obs_aux),
            },
        }
        with open(os.path.join(out_dir, "diffusion_test_subset_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(
            f"[run] diffusion held-out test subset: "
            f"{len(label_paths)} phantoms | seed={diffusion_split_seed} | "
            f"test_offset={diffusion_test_subset_offset} | "
            f"split={diffusion_train_frac:.2f}/{diffusion_val_frac:.2f}/{diffusion_test_frac:.2f}"
        )
        print(
            "[run] selected test phantoms="
            + ", ".join(split_mask_stem(p) for p in label_paths)
        )

    if case_id is not None:
        if not isinstance(case_id, (list, tuple)):
            case_id = [case_id]

        case_id_str = set(str(x) for x in case_id)
        case_id_num = set()
        for x in case_id:
            try:
                case_id_num.add(int(x))
            except Exception:
                pass

        filtered = []
        for p in label_paths:
            stem = split_mask_stem(p)
            num = split_mask_case_number(p)
            if stem in case_id_str or str(num) in case_id_str or num in case_id_num:
                filtered.append(p)

        label_paths = filtered

        if len(label_paths) == 0:
            raise RuntimeError(f"Case {case_id} not found in {data_dir}")

    angle_list = [int(x) for x in angle_list]

    print(f"[run] device={device}")
    print(f"[run] data_dir={data_dir}")
    print(f"[run] raw_roots={raw_roots}")
    print(f"[run] out_dir={out_dir}")
    print(f"[run] num_cases={len(label_paths)}")
    print(f"[run] angle_list={angle_list}")
    print(f"[run] seg_corrupt_severity={seg_corrupt_severity}")
    print(f"[run] seg_corrupt_seed={seg_corrupt_seed}")
    print(f"[run] seg_corrupt_view_prob={seg_corrupt_view_prob}")
    print(f"[run] seg_corrupt_mode={seg_corrupt_mode}")
    print(f"[run] recon_mode={recon_mode}")
    print(f"[run] guidance_mode={guidance_mode}")
    print(f"[run] cond_init_method={cond_init_method}")
    print(f"[run] lowres_recon_det_shape={lowres_recon_det_shape}")
    print(f"[run] shared_obs_root={shared_obs_root}")
    print(f"[run] shared_obs_mode={shared_obs_mode}")
    print(f"[run] require_shared_obs_aux={bool(require_shared_obs_aux)}")
    print(f"[run] diffusion_denoise_t={diffusion_denoise_t}")
    print(f"[run] diffusion_n_averages={diffusion_n_averages}")
    print(f"[run] diffusion_n_stage_snapshots={diffusion_n_stage_snapshots}")
    print(f"[run] diffusion_dc_loss_type={diffusion_dc_loss_type}")
    print(f"[run] pnp_init_mode={pnp_init_mode}")
    print(f"[run] use_trained_segmentation={use_trained_segmentation}")
    print(f"[run] soft_seg_ckpt={soft_seg_ckpt}")
    print(f"[run] soft_seg_pred_thr={soft_seg_pred_thr}")
    print(f"[run] eval_threshold_mode={eval_threshold_mode}")
    print(f"[run] eval_fixed_thr={eval_fixed_thr}")
    print(f"[run] reconstruction_seed={reconstruction_seed}")
    print(f"[run] xray_photon_count={xray_photon_count}")
    print(f"[run] xray_hu_scale={xray_hu_scale}")
    print(f"[run] use_soft_masks_for_recon={use_soft_masks_for_recon}")
    print(f"[run] exact_grid_shape_xyz={exact_grid_shape_xyz}")
    print(
        f"[run] exact_mask_resample_mode="
        f"{exact_mask_resample_mode}"
    )
    print(
        f"[run] enforce_diffusion_grid_match="
        f"{bool(enforce_diffusion_grid_match)}"
    )


    all_results = []

    for p in label_paths:
        case_name = strip_label_suffix(p)
        print("\n" + "#" * 100)
        print(f"[prepare] case={case_name}")

        try:
            stem = split_mask_stem(p)
            m = re.match(r"^(\d+)(?:_(left|right|single))?$", stem)
            if m is None:
                raise ValueError(f"Cannot parse split-mask stem: {stem}")
            case_num = m.group(1)

            img_path = find_img_in_raw_roots(case_num, raw_roots)

            prep = prepare_case(
                lbl_path=p,
                img_path=img_path,
                min_vox_component=min_vox_component,
                bbox_pad=bbox_pad,
                max_dim=max_dim,
                exact_grid_shape_xyz=exact_grid_shape_xyz,
                exact_mask_resample_mode=exact_mask_resample_mode,
                mask_preprocessing=mask_preprocessing,
            )

            if exact_grid_shape_xyz is not None:
                expected_shape = tuple(
                    int(v) for v in exact_grid_shape_xyz
                )
                actual_shape = tuple(
                    int(v) for v in prep["mask_xyz"].shape
                )
                if actual_shape != expected_shape:
                    raise RuntimeError(
                        f"Exact grid creation failed: got {actual_shape}, "
                        f"expected {expected_shape}"
                    )
                print(
                    "[prepare] exact common grid created BEFORE projection: "
                    f"{actual_shape}"
                )
        except Exception as e:
            print(f"[fail] prepare case={case_name} | {repr(e)}")
            traceback.print_exc()
            continue

        case_results = []
        for n_views in angle_list:
            try:
                res = process_case_for_n_views_hybrid(
                    prep=prep,
                    out_dir=out_dir,
                    N_VIEWS=n_views,
                    R_SO_MM=R_SO_MM,
                    PLANE=PLANE,
                    TILT_DEG=TILT_DEG,
                    TILT_AXIS=TILT_AXIS,
                    det_shape=det_shape,
                    det_size_mm=det_size_mm,
                    d_OD_mm=d_OD_mm,
                    proj_eps=proj_eps,
                    bp_eps=bp_eps,
                    allow_missing_views=allow_missing_views,
                    n_points=n_points,
                    device=device,
                    diffusion_prior_ckpt=diffusion_prior_ckpt,
                    diffusion_prior_device=diffusion_prior_device,
                    diffusion_denoise_t=diffusion_denoise_t,
                    diffusion_n_averages=diffusion_n_averages,
                    seg_corrupt_severity=seg_corrupt_severity,
                    seg_corrupt_seed=seg_corrupt_seed,
                    seg_corrupt_view_prob=seg_corrupt_view_prob,
                    seg_corrupt_mode=seg_corrupt_mode,
                    recon_mode=recon_mode,
                    guidance_mode=guidance_mode,
                    guidance_inner_steps=guidance_inner_steps,
                    diffusion_n_stage_snapshots=diffusion_n_stage_snapshots,
                    diffusion_dc_loss_type=diffusion_dc_loss_type,
                    lambda_tv_cond=lambda_tv_cond,
                    tv_mode=tv_mode,
                    tv_grad_normalize=tv_grad_normalize,
                    pnp_init_mode=pnp_init_mode,
                    use_soft_masks=use_soft_masks,
                    soft_mask_sigma=soft_mask_sigma,
                    use_trained_segmentation=use_trained_segmentation,
                    soft_seg_ckpt=soft_seg_ckpt,
                    soft_seg_device=soft_seg_device,
                    soft_seg_base_ch=soft_seg_base_ch,
                    soft_seg_pred_thr=soft_seg_pred_thr,
                    xray_photon_count=xray_photon_count,
                    xray_hu_scale=xray_hu_scale,
                    xray_hu_min=xray_hu_min,
                    xray_hu_max=xray_hu_max,
                    use_soft_masks_for_recon=use_soft_masks_for_recon,
                    xray_delta_mu=xray_delta_mu,
                    max_dim=max_dim,
                    min_vox_component=min_vox_component,
                    cond_dc_step_size=cond_dc_step_size,
                    cond_vh_soft_max=cond_vh_soft_max,
                    alpha_mode=alpha_mode,
                    cond_init_method=cond_init_method,
                    lowres_recon_det_shape=lowres_recon_det_shape,
                    shared_obs_root=shared_obs_root,
                    shared_obs_mode=shared_obs_mode,
                    exact_grid_shape_xyz=exact_grid_shape_xyz,
                    exact_mask_resample_mode=exact_mask_resample_mode,
                    enforce_diffusion_grid_match=bool(
                        enforce_diffusion_grid_match
                    ),
                    eval_threshold_mode=eval_threshold_mode,
                    eval_fixed_thr=eval_fixed_thr,
                    reconstruction_seed=reconstruction_seed,
                    require_shared_obs_aux=require_shared_obs_aux,
                )
                case_results.append(res)
                all_results.append(res)
            except Exception as e:
                print(f"[fail] case={case_name} | N_VIEWS={n_views} | {repr(e)}")
                traceback.print_exc()

        if len(case_results) > 0:
            case_root = os.path.join(out_dir, case_name)
            ensure_dir(case_root)

            plot_case_dice_curve(
                case_results=case_results,
                out_png=os.path.join(case_root, "dice_vs_angles.png"),
            )


            plot_case_probability_evolution_3d(
                case_results=case_results,
                out_png=os.path.join(case_root, "posterior_probability_evolution_3d.png"),
                prob_kind="posterior",
                init_prior_p=0.50,
                prob_thr=0.05,
                max_points_per_panel=12000,
                seed=0,
                elev=25.0,
                azim=45.0,
            )


            with open(os.path.join(case_root, "sweep_summary.json"), "w") as f:
                json.dump(case_results, f, indent=2)

    if len(all_results) > 0:
        save_summary_csv(all_results, os.path.join(out_dir, "summary.csv"))

    print("\n[done] angle sweep finished.")
