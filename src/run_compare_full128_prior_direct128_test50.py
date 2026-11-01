import os
import re
import json
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


# The CLI uses training-matched mask preprocessing and explicit saved-split
# membership. Legacy preprocessing remains available for historical replay.
from pipeline_full128_prior_direct128_test50_disjoint import run_angle_sweep

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_case_summary(summary_json):
    with open(summary_json, "r") as f:
        data = json.load(f)
    return data


def find_case_dirs_with_summary(root_dir):
    out = {}
    if not os.path.isdir(root_dir):
        return out

    for d in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, d)
        if not os.path.isdir(full):
            continue
        sj = os.path.join(full, "sweep_summary.json")
        if os.path.exists(sj):
            out[d] = full
    return out


def case_matches_requested(case_name, requested_ids):
    if requested_ids is None:
        return True

    requested_str = set(str(x) for x in requested_ids)

    if case_name in requested_str:
        return True

    m = re.match(r"^(\d+)", case_name)
    if m is not None and m.group(1) in requested_str:
        return True

    return False


def summary_to_sorted_curves(summary_list):
    if len(summary_list) == 0:
        return None

    s = sorted(summary_list, key=lambda x: int(x["N_VIEWS"]))
    x = np.array([int(r["N_VIEWS"]) for r in s], dtype=np.int32)


    y_init = np.array([float(r["dice_visual_hull_vs_gt"]) for r in s], dtype=np.float32)
    init_label = "Visual hull"

    y_method = np.array([float(r["dice_final_method_fixed_vs_gt"]) for r in s], dtype=np.float32)

    return {
        "x": x,
        "y_init": y_init,
        "y_method": y_method,
        "init_label": init_label,
    }
    
    
def plot_two_curve_case(
    case_dir,
    out_png,
    method_label="Conditional diffusion",
    title_suffix="imperfect masks",
):
    summary_json = os.path.join(case_dir, "sweep_summary.json")

    if not os.path.exists(summary_json):
        print(f"[warn] missing summary: {summary_json}")
        return False

    summary = load_case_summary(summary_json)
    curves = summary_to_sorted_curves(summary)
    if curves is None:
        return False

    plt.figure(figsize=(7, 5))
    plt.plot(
        curves["x"],
        curves["y_init"],
        marker="o",
        linewidth=2,
        label=curves["init_label"],
    )
    plt.plot(
        curves["x"],
        curves["y_method"],
        marker="s",
        linewidth=2,
        label=method_label,
    )
    plt.xlabel("Number of views")
    plt.ylabel("Dice")
    plt.title(f"Dice vs angle count ({title_suffix})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True


def collect_case_metric(case_dir, metric_key):
    summary_json = os.path.join(case_dir, "sweep_summary.json")
    if not os.path.exists(summary_json):
        return {}

    data = load_case_summary(summary_json)
    out = {}

    for r in data:
        if metric_key not in r or r[metric_key] is None:
            continue
        n = int(r["N_VIEWS"])
        out[n] = float(r[metric_key])

    return out

def plot_mean_two_curve(
    case_dirs,
    requested_case_ids,
    out_png,
    out_csv,
    method_label="Conditional diffusion",
    title_suffix="imperfect masks",
):
    case_names = sorted(case_dirs.keys())
    case_names = [c for c in case_names if case_matches_requested(c, requested_case_ids)]

    if len(case_names) == 0:
        print("[warn] no cases for mean plot")
        return False

    init_all = {}
    method_all = {}
    init_label = "Raw visual hull"

    for cid in case_names:
        case_dir = case_dirs[cid]
        summary_json = os.path.join(case_dir, "sweep_summary.json")
        if not os.path.exists(summary_json):
            continue

        summary = load_case_summary(summary_json)
        if len(summary) == 0:
            continue



        init_key = "dice_visual_hull_vs_gt"
        init_label = "Raw visual hull"

        init_vals = collect_case_metric(case_dir, init_key)
        method_vals = collect_case_metric(case_dir, "dice_final_method_fixed_vs_gt")

        for n, v in init_vals.items():
            init_all.setdefault(n, []).append(v)
        for n, v in method_vals.items():
            method_all.setdefault(n, []).append(v)

    if len(init_all) == 0 or len(method_all) == 0:
        print("[warn] no valid metrics for mean plot")
        return False

    all_angles = sorted(set(init_all.keys()) | set(method_all.keys()))

    mean_init = {
        n: float(np.mean(init_all[n])) for n in init_all
    }
    mean_method = {
        n: float(np.mean(method_all[n])) for n in method_all
    }

    plt.figure(figsize=(7, 5))
    plt.plot(
        sorted(mean_init.keys()),
        [mean_init[n] for n in sorted(mean_init.keys())],
        marker="o",
        linewidth=2,
        label=init_label,
    )
    plt.plot(
        sorted(mean_method.keys()),
        [mean_method[n] for n in sorted(mean_method.keys())],
        marker="s",
        linewidth=2,
        label=method_label,
    )
    plt.xlabel("Number of views")
    plt.ylabel("Mean Dice")
    plt.title(f"Mean Dice vs angle count ({title_suffix})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["N_VIEWS", "mean_dice_init", "mean_dice_method"],
        )
        writer.writeheader()
        for n in all_angles:
            writer.writerow({
                "N_VIEWS": int(n),
                "mean_dice_init": mean_init.get(n, None),
                "mean_dice_method": mean_method.get(n, None),
            })

    return True

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="slicer_full_pipeline/split_masks")
    parser.add_argument("--out_dir", type=str, default="compare_vh_vs_x0guided_condition_imperfect_only")

    parser.add_argument(
        "--case_id",
        type=int,
        nargs="*",
        default=None,
        help="Case IDs to run, e.g. --case_id 1 2 3",
    )

    parser.add_argument(
        "--angle_list",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="View counts to evaluate; experiments use 2, 4, and 8.",
    )

    parser.add_argument("--R_SO_MM", type=float, default=400.0)
    parser.add_argument("--PLANE", type=str, default="yz")
    parser.add_argument("--TILT_DEG", type=float, default=0.0)
    parser.add_argument("--TILT_AXIS", type=str, default="x")

    parser.add_argument("--det_h", type=int, default=512)
    parser.add_argument("--det_w", type=int, default=512)
    parser.add_argument("--det_h_mm", type=float, default=300.0)
    parser.add_argument("--det_w_mm", type=float, default=300.0)

    parser.add_argument("--d_OD_mm", type=float, default=400.0)

    parser.add_argument(
        "--max_dim",
        type=int,
        default=9999,
        help=(
            "Legacy max-dimension setting. It is ignored when the exact "
            "--volume_grid_shape is supplied."
        ),
    )
    parser.add_argument(
        "--volume_grid_shape",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        metavar=("NX", "NY", "NZ"),
        help=(
            "Exact 3-D grid created before projection and used by every "
            "reconstruction stage. Default: 128 128 128."
        ),
    )
    parser.add_argument(
        "--volume_mask_resample_mode",
        choices=["nearest", "soft"],
        default="nearest",
        help=(
            "nearest keeps a binary vessel target; soft uses trilinear "
            "partial-volume occupancy for X-ray simulation."
        ),
    )
    parser.add_argument(
        "--enforce_diffusion_grid_match",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Require the 3-D reconstruction grid to equal the full-volume "
            "diffusion training grid."
        ),
    )
    parser.add_argument("--bbox_pad", type=int, default=0)
    parser.add_argument("--mask_preprocessing", choices=["training_matched", "legacy"],
                        default="training_matched",
                        help="training_matched uses the diffusion trainer's full-volume mask preparation; legacy reproduces earlier runs.")
    parser.add_argument("--min_vox_component", type=int, default=50)

    parser.add_argument("--proj_eps", type=float, default=1e-8)
    parser.add_argument("--bp_eps", type=float, default=1e-8)
    parser.add_argument("--allow_missing_views", type=int, default=0)

    parser.add_argument("--n_points", type=int, default=20000)

    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )


    parser.add_argument("--seg_corrupt_view_prob", type=float, default=0.35)
    parser.add_argument("--seg_corrupt_mode", type=str, default="sparse_like_figure")

    parser.add_argument(
        "--diffusion_prior_ckpt",
        type=str,
        default=None,
        help="Full-volume diffusion checkpoint used in the experiments.",
    )
    parser.add_argument("--diffusion_prior_device", type=str, default=None)



    parser.add_argument(
        "--cond_dc_step_size",
        type=float,
        default=0.03,
        help="Step size for x0-guided measurement correction in conditional diffusion. Do not set this as high as 0.8.",
    )

    parser.add_argument(
        "--cond_vh_soft_max",
        type=float,
        default=0.30,
        help=(
            "Probability assigned to voxels inside the visual hull when "
            "--recon_mode cond_diffusion and --pnp_init_mode vh are used. "
            "Valid range: [0, 1]."
        ),
    )

    parser.add_argument(
        "--alpha_mode",
        type=str,
        default="neglog",
        choices=["prob", "neglog"],
        help="Projection model: alpha=p for stable soft density, or alpha=-log(1-p).",
    )

    parser.add_argument(
        "--diffusion_denoise_t",
        type=int,
        default=15,
        help="Denoising strength / start timestep",
    )
    parser.add_argument(
        "--diffusion_n_averages",
        type=int,
        default=1,
        help="Legacy run metadata (1). Generate posterior samples in separate seeded runs.",
    )

    parser.add_argument(
        "--cond_init_method",
        type=str,
        default="q_sample",
        choices=["q_sample"],
        help="How to initialize conditional diffusion from the chosen init probability map.",
    )
    parser.add_argument(
        "--lowres_recon_det_shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Optional detector mask size used only for reconstruction, e.g. --lowres_recon_det_shape 128 128",
    )
    parser.add_argument(
        "--shared_obs_root",
        type=str,
        default=None,
        help="Root folder for shared 2-D observations. Use with --shared_obs_mode.",
    )
    parser.add_argument(
        "--shared_obs_mode",
        type=str,
        default="none",
        choices=["none", "save", "overwrite", "save_if_missing", "load"],
        help="Shared observation mode. Use overwrite/save for the first method and load for the others.",
    )
    parser.add_argument(
        "--require_shared_obs_aux",
        action="store_true",
        help=(
            "Require shared noisy X-rays, soft segmentation probabilities, and "
            "GT projection masks in addition to masks/confidence maps."
        ),
    )
    parser.add_argument(
        "--reconstruction_seed",
        type=int,
        default=None,
        help=(
            "Optional base seed. A deterministic seed is derived per case/view so "
            "results do not depend on loop order or diffusion timestep."
        ),
    )
    parser.add_argument(
        "--eval_threshold_mode",
        choices=["fixed"],
        default="fixed",
        help=(
            "Use the frozen reconstruction threshold for evaluation."
        ),
    )
    parser.add_argument(
        "--eval_fixed_thr",
        type=float,
        default=0.05,
        help="One global probability threshold used when --eval_threshold_mode=fixed.",
    )

    parser.add_argument(
        "--imperfect_severity",
        type=float,
        default=0.0,
        help="Mask corruption severity for the imperfect-mask run",
    )
    parser.add_argument(
        "--imperfect_seed",
        type=int,
        default=0,
        help="Random seed for imperfect-mask corruption",
    )

    parser.add_argument(
        "--recon_mode",
        type=str,
        default="cond_diffusion",
        choices=["cond_diffusion"],
        help="Visual-hull-initialized conditional diffusion.",
    )

    parser.add_argument(
        "--guidance_mode",
        choices=["full", "prior_only"],
        default="full",
        help=(
            "Conditional-diffusion guidance. full preserves measurement "
            "consistency; prior_only keeps the visual-hull initialization "
            "and DDPM sampler but skips every measurement-guidance update."
        ),
    )

    parser.add_argument(
        "--guidance_inner_steps",
        type=int,
        default=1,
        help="Inner guidance steps for conditional diffusion",
    )
    parser.add_argument(
        "--diffusion_n_stage_snapshots",
        type=int,
        default=4,
        help="Number of saved stage snapshots for 3D visualization",
    )
    parser.add_argument(
        "--diffusion_dc_loss_type",
        type=str,
        default="bce",
        choices=["bce", "l2"],
        help="Data-consistency loss for conditional diffusion.",
    )

    parser.add_argument(
        "--lambda_tv_cond",
        type=float,
        default=0.0,
        help="TV weight used in conditional diffusion guidance",
    )
    parser.add_argument(
        "--tv_mode",
        type=str,
        default="anisotropic",
        choices=["anisotropic", "isotropic"],
        help="TV mode used in conditional diffusion guidance",
    )
    parser.add_argument(
        "--tv_grad_normalize",
        type=int,
        default=1,
        help="Whether to normalize TV gradient",
    )

    parser.add_argument(
        "--pnp_init_mode",
        type=str,
        default="vh",
        choices=["vh", "visual_hull"],
    )

    parser.add_argument("--use_soft_masks", action="store_true")
    parser.add_argument("--soft_mask_sigma", type=float, default=1.0)

    parser.add_argument("--use_trained_segmentation", action="store_true")
    parser.add_argument("--soft_seg_ckpt", type=str, default=None)
    parser.add_argument("--soft_seg_device", type=str, default=None)
    parser.add_argument("--soft_seg_base_ch", type=int, default=16)
    parser.add_argument("--soft_seg_pred_thr", type=float, default=0.5)

    parser.add_argument("--xray_photon_count", type=float, default=50000)
    parser.add_argument("--xray_hu_scale", type=float, default=5e-3)
    parser.add_argument("--xray_hu_min", type=float, default=-1000.0)
    parser.add_argument("--xray_hu_max", type=float, default=3000.0)

    parser.add_argument("--use_soft_masks_for_recon", action="store_true")
    parser.add_argument("--raw_roots", type=str, nargs="+", default=None)
    parser.add_argument("--xray_delta_mu", type=float, default=0.008)

    parser.add_argument(
        "--use_diffusion_test_subset",
        action="store_true",
        help=(
            "Use a deterministic subset from the held-out diffusion-prior test split "
            "instead of selecting cases with --case_id."
        ),
    )
    parser.add_argument("--diffusion_test_subset_size", type=int, default=50)
    parser.add_argument(
        "--diffusion_test_subset_offset",
        type=int,
        default=0,
        help="Start offset within the held-out diffusion-prior test split.",
    )
    parser.add_argument("--diffusion_split_json", type=str, default=None,
                        help="Saved data_split.json from the diffusion checkpoint's training run; required for --use_diffusion_test_subset.")
    parser.add_argument("--diffusion_split_seed", type=int, default=0,
                        help="Legacy compatibility option; saved split membership is never regenerated.")
    parser.add_argument("--diffusion_train_frac", type=float, default=0.70)
    parser.add_argument("--diffusion_val_frac", type=float, default=0.15)
    parser.add_argument("--diffusion_test_frac", type=float, default=0.15)
    parser.add_argument("--diffusion_split_start_index", type=int, default=1)
    parser.add_argument("--diffusion_split_max_cases", type=int, default=2000)


    args = parser.parse_args()

    if not (0.0 <= float(args.cond_vh_soft_max) <= 1.0):
        parser.error("--cond_vh_soft_max must be in [0, 1]")
    if not (0.0 <= float(args.eval_fixed_thr) <= 1.0):
        parser.error("--eval_fixed_thr must be in [0, 1]")

    if args.use_diffusion_test_subset and args.case_id is not None:
        parser.error(
            "Do not combine --use_diffusion_test_subset with --case_id; "
            "the held-out test subset must be selected directly from the diffusion split."
        )

    ensure_dir(args.out_dir)

    imperfect_out = os.path.join(args.out_dir, "imperfect_masks")
    compare_out = os.path.join(args.out_dir, "comparison_plots")

    ensure_dir(imperfect_out)
    ensure_dir(compare_out)

    method_label = (
        "x0-guided conditional diffusion"
        if args.guidance_mode == "full"
        else "Prior-only diffusion"
    )

    tv_grad_normalize_bool = bool(int(args.tv_grad_normalize))

    print("\n" + "=" * 100)
    print("[run] IMPERFECT MASKS ONLY")
    print(f"[run] recon_mode={args.recon_mode}")
    print(f"[run] guidance_mode={args.guidance_mode}")
    if args.recon_mode == "cond_diffusion":
        conditioning_type = (
            "x0-guided measurement correction"
            if args.guidance_mode == "full"
            else "prior-only; no measurement update"
        )
        print(f"[run] conditioning_type={conditioning_type}")
    print(f"[run] diffusion_denoise_t={args.diffusion_denoise_t}")
    print(f"[run] cond_dc_step_size={args.cond_dc_step_size}")
    print(f"[run] cond_vh_soft_max={args.cond_vh_soft_max}")
    print(f"[run] alpha_mode={args.alpha_mode}")
    print(f"[run] diffusion_n_averages={args.diffusion_n_averages}")
    print(f"[run] diffusion_n_stage_snapshots={args.diffusion_n_stage_snapshots}")
    print(f"[run] diffusion_dc_loss_type={args.diffusion_dc_loss_type}")
    print(f"[run] guidance_inner_steps={args.guidance_inner_steps}")
    print(f"[run] lambda_tv_cond={args.lambda_tv_cond}")
    print(f"[run] tv_mode={args.tv_mode}")
    print(f"[run] tv_grad_normalize={tv_grad_normalize_bool}")
    print(f"[run] pnp_init_mode={args.pnp_init_mode}")
    print(f"[run] use_trained_segmentation={args.use_trained_segmentation}")
    print(f"[run] soft_seg_ckpt={args.soft_seg_ckpt}")
    print(f"[run] soft_seg_pred_thr={args.soft_seg_pred_thr}")
    print(f"[run] xray_photon_count={args.xray_photon_count}")
    print(f"[run] xray_hu_scale={args.xray_hu_scale}")
    print(f"[run] xray_delta_mu={args.xray_delta_mu}")
    print(f"[run] use_diffusion_test_subset={args.use_diffusion_test_subset}")
    if args.use_diffusion_test_subset:
        print(f"[run] diffusion_test_subset_size={args.diffusion_test_subset_size}")
        print(f"[run] diffusion_test_subset_offset={args.diffusion_test_subset_offset}")
        print(f"[run] diffusion_split_seed={args.diffusion_split_seed}")
        print(
            f"[run] diffusion_split_fracs="
            f"{args.diffusion_train_frac}/{args.diffusion_val_frac}/{args.diffusion_test_frac}"
        )
        print(f"[run] diffusion_split_start_index={args.diffusion_split_start_index}")
        print(f"[run] diffusion_split_max_cases={args.diffusion_split_max_cases}")
    print(f"[run] use_soft_masks_for_recon={args.use_soft_masks_for_recon}")
    print(f"[run] volume_grid_shape={args.volume_grid_shape}")
    print(
        f"[run] volume_mask_resample_mode="
        f"{args.volume_mask_resample_mode}"
    )
    print(
        f"[run] enforce_diffusion_grid_match="
        f"{bool(args.enforce_diffusion_grid_match)}"
    )
    print(f"[run] cond_init_method={args.cond_init_method}")
    print(f"[run] lowres_recon_det_shape={args.lowres_recon_det_shape}")
    print(f"[run] shared_obs_root={args.shared_obs_root}")
    print(f"[run] shared_obs_mode={args.shared_obs_mode}")
    print(f"[run] require_shared_obs_aux={args.require_shared_obs_aux}")
    print(f"[run] reconstruction_seed={args.reconstruction_seed}")
    print(f"[run] eval_threshold_mode={args.eval_threshold_mode}")
    print(f"[run] eval_fixed_thr={args.eval_fixed_thr}")


    run_angle_sweep(
        data_dir=args.data_dir,
        out_dir=imperfect_out,
        case_id=args.case_id,
        angle_list=args.angle_list,
        R_SO_MM=args.R_SO_MM,
        PLANE=args.PLANE,
        TILT_DEG=args.TILT_DEG,
        TILT_AXIS=args.TILT_AXIS,
        det_shape=(args.det_h, args.det_w),
        det_size_mm=(args.det_h_mm, args.det_w_mm),
        d_OD_mm=args.d_OD_mm,
        max_dim=args.max_dim,
        bbox_pad=args.bbox_pad,
        min_vox_component=args.min_vox_component,
        proj_eps=args.proj_eps,
        bp_eps=args.bp_eps,
        allow_missing_views=args.allow_missing_views,
        n_points=args.n_points,
        device=args.device,
        diffusion_prior_ckpt=args.diffusion_prior_ckpt,
        diffusion_prior_device=args.diffusion_prior_device,
        cond_dc_step_size=args.cond_dc_step_size,
        cond_vh_soft_max=args.cond_vh_soft_max,
        alpha_mode=args.alpha_mode,
        diffusion_denoise_t=args.diffusion_denoise_t,
        diffusion_n_averages=args.diffusion_n_averages,
        seg_corrupt_severity=args.imperfect_severity,
        seg_corrupt_seed=args.imperfect_seed,
        seg_corrupt_view_prob=args.seg_corrupt_view_prob,
        seg_corrupt_mode=args.seg_corrupt_mode,
        recon_mode=args.recon_mode,
        guidance_mode=args.guidance_mode,
        guidance_inner_steps=args.guidance_inner_steps,
        diffusion_n_stage_snapshots=args.diffusion_n_stage_snapshots,
        diffusion_dc_loss_type=args.diffusion_dc_loss_type,
        lambda_tv_cond=args.lambda_tv_cond,
        tv_mode=args.tv_mode,
        tv_grad_normalize=tv_grad_normalize_bool,
        pnp_init_mode=args.pnp_init_mode,
        use_soft_masks=args.use_soft_masks,
        soft_mask_sigma=args.soft_mask_sigma,
        use_trained_segmentation=args.use_trained_segmentation,
        soft_seg_ckpt=args.soft_seg_ckpt,
        soft_seg_device=args.soft_seg_device,
        soft_seg_base_ch=args.soft_seg_base_ch,
        soft_seg_pred_thr=args.soft_seg_pred_thr,
        xray_photon_count=args.xray_photon_count,
        xray_hu_scale=args.xray_hu_scale,
        xray_hu_min=args.xray_hu_min,
        xray_hu_max=args.xray_hu_max,
        use_soft_masks_for_recon=args.use_soft_masks_for_recon,
        raw_roots=args.raw_roots,
        xray_delta_mu=args.xray_delta_mu,
        cond_init_method=args.cond_init_method,
        lowres_recon_det_shape=args.lowres_recon_det_shape,
        shared_obs_root=args.shared_obs_root,
        shared_obs_mode=args.shared_obs_mode,
        exact_grid_shape_xyz=tuple(
            int(v) for v in args.volume_grid_shape
        ),
        exact_mask_resample_mode=args.volume_mask_resample_mode,
        mask_preprocessing=args.mask_preprocessing,
        enforce_diffusion_grid_match=bool(
            args.enforce_diffusion_grid_match
        ),
        use_diffusion_test_subset=bool(args.use_diffusion_test_subset),
        diffusion_test_subset_size=int(args.diffusion_test_subset_size),
        diffusion_test_subset_offset=int(args.diffusion_test_subset_offset),
        diffusion_split_seed=int(args.diffusion_split_seed),
        diffusion_split_json=args.diffusion_split_json,
        diffusion_train_frac=float(args.diffusion_train_frac),
        diffusion_val_frac=float(args.diffusion_val_frac),
        diffusion_test_frac=float(args.diffusion_test_frac),
        diffusion_split_start_index=int(args.diffusion_split_start_index),
        diffusion_split_max_cases=int(args.diffusion_split_max_cases),
        eval_threshold_mode=str(args.eval_threshold_mode),
        eval_fixed_thr=float(args.eval_fixed_thr),
        reconstruction_seed=args.reconstruction_seed,
        require_shared_obs_aux=bool(args.require_shared_obs_aux),
    )

    imperfect_cases = find_case_dirs_with_summary(imperfect_out)

    print("\n[info] imperfect cases with summary:")
    print(sorted(imperfect_cases.keys()))

    case_names = sorted(imperfect_cases.keys())
    case_names = [
        c for c in case_names
        if case_matches_requested(c, args.case_id)
    ]

    if len(case_names) == 0:
        print("\n[warn] no case folders with sweep_summary.json were found")
    else:
        n_ok = 0
        for cid in case_names:
            case_dir = imperfect_cases[cid]
            out_png = os.path.join(compare_out, f"{cid}_imperfect_compare.png")

            ok = plot_two_curve_case(
                case_dir=case_dir,
                out_png=out_png,
                method_label=method_label,
                title_suffix=(
                    f"imperfect masks | severity={args.imperfect_severity} | "
                    f"view_prob={args.seg_corrupt_view_prob} | mode={args.seg_corrupt_mode} | "
                    f"recon={args.recon_mode} | guidance={args.guidance_mode} | "
                    f"alpha={args.alpha_mode}"
                ),
            )
            if ok:
                print(f"[ok] saved comparison plot: {out_png}")
                n_ok += 1

        print(f"\n[info] per-case comparison plots saved: {n_ok}")

        mean_png = os.path.join(compare_out, "mean_imperfect_compare.png")
        mean_csv = os.path.join(compare_out, "mean_imperfect_compare.csv")

        ok_mean = plot_mean_two_curve(
            case_dirs=imperfect_cases,
            requested_case_ids=args.case_id,
            out_png=mean_png,
            out_csv=mean_csv,
            method_label=method_label,
            title_suffix=(
                f"imperfect masks | severity={args.imperfect_severity} | "
                f"view_prob={args.seg_corrupt_view_prob} | mode={args.seg_corrupt_mode} | "
                f"recon={args.recon_mode} | guidance={args.guidance_mode} | "
                f"alpha={args.alpha_mode}"
            ),
        )
        if ok_mean:
            print(f"[ok] saved mean comparison plot: {mean_png}")
            print(f"[ok] saved mean comparison csv : {mean_csv}")

    print("\n[done] imperfect-only comparison finished")
    print(f"imperfect results  : {imperfect_out}")
    print(f"comparison plots   : {compare_out}")


if __name__ == "__main__":
    main()
