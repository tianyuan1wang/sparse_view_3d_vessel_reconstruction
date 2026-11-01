import os

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from helpers import (
    _mid_slices_xyz,
    _draw_prob_panel,
    _draw_binary_panel,
    dice_iou,
    choose_view_indices,
)


# ============================================================
# Reconstruction visualization
# ============================================================


# ============================================================
# 3D probability visualization
# ============================================================

def _sample_probability_points(prob_xyz, prob_thr=0.05, max_points=25000, seed=0):
    prob_xyz = np.asarray(prob_xyz, dtype=np.float32)
    prob_xyz = np.clip(prob_xyz, 0.0, 1.0)

    keep = prob_xyz >= float(prob_thr)

    if np.any(keep):
        coords_xyz = np.argwhere(keep)
        vals = prob_xyz[keep]
    else:
        flat = prob_xyz.ravel()
        k = min(int(max_points), flat.size)
        if k <= 0:
            return np.zeros((0, 3), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        idx = np.argsort(flat)[-k:]
        coords_xyz = np.column_stack(np.unravel_index(idx, prob_xyz.shape))
        vals = flat[idx]

    if len(coords_xyz) > int(max_points):
        rng = np.random.default_rng(int(seed))
        w = vals.astype(np.float64)
        wsum = float(w.sum())

        if wsum > 0:
            w = w / wsum
            sel = rng.choice(len(coords_xyz), size=int(max_points), replace=False, p=w)
        else:
            sel = rng.choice(len(coords_xyz), size=int(max_points), replace=False)

        coords_xyz = coords_xyz[sel]
        vals = vals[sel]

    return coords_xyz.astype(np.int32), vals.astype(np.float32)


def _voxel_coords_xyz_to_world_mm(coords_xyz, shape_xyz, spacing_xyz):
    coords_xyz = np.asarray(coords_xyz, dtype=np.float32)
    shape_xyz = np.asarray(shape_xyz, dtype=np.float32)
    spacing_xyz = np.asarray(spacing_xyz, dtype=np.float32)

    xyz_mm = (coords_xyz + 0.5 - shape_xyz[None, :] / 2.0) * spacing_xyz[None, :]
    return xyz_mm.astype(np.float32)


def _set_3d_axes_equal_mm(ax, shape_xyz, spacing_xyz):
    shape_xyz = np.asarray(shape_xyz, dtype=np.float32)
    spacing_xyz = np.asarray(spacing_xyz, dtype=np.float32)

    ext = shape_xyz * spacing_xyz
    hx, hy, hz = 0.5 * ext[0], 0.5 * ext[1], 0.5 * ext[2]

    ax.set_xlim(-hx, hx)
    ax.set_ylim(-hy, hy)
    ax.set_zlim(-hz, hz)

    try:
        ax.set_box_aspect((float(ext[0]), float(ext[1]), float(ext[2])))
    except Exception:
        pass


def make_white_viridis():
    base = plt.get_cmap('viridis')
    base_colors = base(np.linspace(0.0, 1.0, 256))

    n_white = 45
    white = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    target = base_colors[n_white].copy()

    ramp = np.zeros((n_white, 4), dtype=np.float32)
    for i in range(n_white):
        t = i / max(n_white - 1, 1)
        ramp[i] = (1.0 - t) * white + t * target

    base_colors[:n_white] = ramp

    cmap = matplotlib.colors.ListedColormap(base_colors, name='white_viridis')
    return cmap


def plot_case_probability_evolution_3d(
    case_results,
    out_png,
    prob_kind='posterior',
    init_prior_p=0.5,
    prob_thr=0.05,
    max_points_per_panel=12000,
    seed=0,
    elev=25.0,
    azim=45.0,
    point_size=1.8,
    point_alpha=1.0,
    cbar_position='upper',
):
    case_results = sorted(case_results, key=lambda x: int(x['N_VIEWS']))
    if len(case_results) == 0:
        return

    first = case_results[0]
    case_id = first['case_id']
    shape_xyz = tuple(int(v) for v in first['shape_local_xyz'])
    spacing_xyz = np.asarray(first['spacing_eff_xyz_mm'], dtype=np.float32)

    if str(prob_kind).lower() == 'prior':
        nii_name = 'bayes_prior_map.nii.gz'
        title_root = '3D prior evolution'
    else:
        nii_name = 'bayes_prob_map.nii.gz'
        title_root = '3D posterior probability evolution'

    volumes = [np.full(shape_xyz, float(init_prior_p), dtype=np.float32)]
    titles = ['N=0\ninit prior']

    for r in case_results:
        p = os.path.join(r['output_dir'], nii_name)
        if not os.path.exists(p):
            continue
        vol = np.asarray(nib.load(p).get_fdata(), dtype=np.float32)
        volumes.append(vol)
        titles.append(f"N={int(r['N_VIEWS'])}")

    n_cols = len(volumes)
    if n_cols == 0:
        return

    fig = plt.figure(figsize=(4.2 * n_cols + 0.9, 4.8))
    fig.subplots_adjust(left=0.02, right=0.90, top=0.86, bottom=0.08, wspace=0.10)

    axes = []

    cmap = make_white_viridis()
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for i, (vol, ttl) in enumerate(zip(volumes, titles), start=1):
        ax = fig.add_subplot(1, n_cols, i, projection='3d')
        axes.append(ax)

        coords_xyz, vals = _sample_probability_points(
            prob_xyz=vol,
            prob_thr=float(prob_thr),
            max_points=int(max_points_per_panel),
            seed=int(seed + i),
        )

        if len(coords_xyz) > 0:
            pts_mm = _voxel_coords_xyz_to_world_mm(
                coords_xyz=coords_xyz,
                shape_xyz=vol.shape,
                spacing_xyz=spacing_xyz,
            )

            order = np.argsort(vals)
            pts_mm = pts_mm[order]
            vals = vals[order]

            ax.scatter(
                pts_mm[:, 0],
                pts_mm[:, 1],
                pts_mm[:, 2],
                c=vals,
                cmap=cmap,
                norm=norm,
                s=float(point_size),
                alpha=float(point_alpha),
                linewidths=0.0,
                edgecolors='none',
                depthshade=False,
            )
        else:
            ax.text2D(0.30, 0.5, 'No points', transform=ax.transAxes)

        _set_3d_axes_equal_mm(ax, vol.shape, spacing_xyz)

        ax.set_title(ttl)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=float(elev), azim=float(azim))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)

        try:
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        except Exception:
            pass

    fig.suptitle(f'{title_root}: {case_id}', fontsize=13)

    if cbar_position == 'upper':
        cax = fig.add_axes([0.92, 0.50, 0.012, 0.32])
    elif cbar_position == 'lower':
        cax = fig.add_axes([0.92, 0.14, 0.012, 0.32])
    else:
        cax = fig.add_axes([0.92, 0.26, 0.012, 0.40])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('occupancy probability')

    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close()


# ============================================================
# Reconstruction visualization with contours
# ============================================================

def save_volume_views_xyz_with_contours(
    vol_xyz,
    out_png,
    title='',
    contour_levels=(0.25, 0.50, 0.75, 0.90),
):
    sl = _mid_slices_xyz(vol_xyz)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im = None
    for ax, name in zip(axes, ['XY', 'XZ', 'YZ']):
        im = _draw_prob_panel(
            ax,
            sl[name],
            title=name,
            add_contours=True,
            contour_levels=contour_levels,
            cmap='viridis',
            vmin=0.0,
            vmax=1.0,
        )

    if title:
        fig.suptitle(title, fontsize=12)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label('occupancy probability')

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()


def save_bayes_comparison_xyz(
    gt_xyz,
    init_prior_xyz,
    final_prior_xyz,
    prob_xyz,
    bayes_fixed_xyz,
    bayes_oracle_xyz,
    out_png,
    title='',
    contour_levels=(0.25, 0.50, 0.75, 0.90),
):
    panels = [
        ('GT',           gt_xyz.astype(np.float32),           'binary'),
        ('Init prior',   init_prior_xyz.astype(np.float32),   'prob'),
        ('Final prior',  final_prior_xyz.astype(np.float32),  'prob'),
        ('Posterior q',  prob_xyz.astype(np.float32),         'prob'),
        ('Bayes fixed',  bayes_fixed_xyz.astype(np.float32),  'binary'),
        ('Bayes oracle', bayes_oracle_xyz.astype(np.float32), 'binary'),
    ]

    view_names = ['XY', 'XZ', 'YZ']
    fig, axes = plt.subplots(3, len(panels), figsize=(3.2 * len(panels), 9))
    last_im = None

    for c, (col_name, vol, kind) in enumerate(panels):
        sl = _mid_slices_xyz(vol)
        for r, view_name in enumerate(view_names):
            ax = axes[r, c]
            if kind == 'prob':
                last_im = _draw_prob_panel(
                    ax,
                    sl[view_name],
                    title=col_name if r == 0 else '',
                    add_contours=True,
                    contour_levels=contour_levels,
                    cmap='viridis',
                    vmin=0.0,
                    vmax=1.0,
                )
            else:
                _draw_binary_panel(
                    ax,
                    sl[view_name],
                    title=col_name if r == 0 else '',
                )

            if c == 0:
                ax.set_ylabel(view_name)

    if title:
        fig.suptitle(title, fontsize=13)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes[:, 1:4].ravel().tolist(), shrink=0.8)
        cbar.set_label('occupancy probability')

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()


def plot_case_prior_evolution(case_results, out_png, contour_level=0.50):
    case_results = sorted(case_results, key=lambda x: int(x['N_VIEWS']))
    if len(case_results) == 0:
        return

    priors = []
    angle_list = []

    for r in case_results:
        prior_path = os.path.join(r['output_dir'], 'bayes_prior_map.nii.gz')
        if not os.path.exists(prior_path):
            continue

        prior_xyz = np.asarray(nib.load(prior_path).get_fdata(), dtype=np.float32)
        priors.append(prior_xyz)
        angle_list.append(int(r['N_VIEWS']))

    if len(priors) == 0:
        return

    n_cols = len(priors)
    view_names = ['XY', 'XZ', 'YZ']

    fig, axes = plt.subplots(
        3, n_cols,
        figsize=(2.7 * n_cols, 8.0),
        squeeze=False,
    )

    last_im = None

    for c, (prior_xyz, n_views) in enumerate(zip(priors, angle_list)):
        sl = _mid_slices_xyz(prior_xyz)

        for r, view_name in enumerate(view_names):
            ax = axes[r, c]
            arr = np.asarray(sl[view_name], dtype=np.float32)

            im = ax.imshow(
                arr,
                origin='lower',
                cmap='viridis',
                vmin=0.0,
                vmax=1.0,
            )
            last_im = im

            lo = float(np.min(arr))
            hi = float(np.max(arr))
            if lo < float(contour_level) < hi:
                ax.contour(
                    arr,
                    levels=[float(contour_level)],
                    colors='white',
                    linewidths=0.8,
                )

            if r == 0:
                ax.set_title(f'N={n_views}')
            if c == 0:
                ax.set_ylabel(view_name)

            ax.axis('off')

    fig.suptitle(
        f"Final prior evolution vs number of angles: {case_results[0]['case_id']}",
        fontsize=13,
    )

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.85)
        cbar.set_label('prior probability')

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close()


def plot_diffusion_stage_evolution_3d(
    stage_prob_xyz_list,
    out_png,
    stage_titles=None,
    main_title="3D diffusion probability evolution",
    prob_thr=0.05,
    max_points_per_panel=12000,
    seed=0,
    elev=25.0,
    azim=45.0,
):
    """
    Plot multiple diffusion stages side-by-side in 3D.

    Parameters
    ----------
    stage_prob_xyz_list : list of np.ndarray
        Each item is a 3D probability map in XYZ order.
    out_png : str
        Output png path.
    stage_titles : list[str] or None
        Titles per stage. If None, default titles are used.
    """
    if stage_prob_xyz_list is None or len(stage_prob_xyz_list) == 0:
        return

    n_panels = len(stage_prob_xyz_list)
    rng = np.random.default_rng(seed)

    if stage_titles is None:
        stage_titles = [f"Stage {i}" for i in range(n_panels)]

    fig = plt.figure(figsize=(6 * n_panels, 6), constrained_layout=True)
    scatter_handle = None

    for i, prob_xyz in enumerate(stage_prob_xyz_list):
        ax = fig.add_subplot(1, n_panels, i + 1, projection="3d")

        prob_xyz = np.asarray(prob_xyz, dtype=np.float32)
        coords = np.argwhere(prob_xyz > prob_thr)

        if len(coords) == 0:
            ax.set_title(stage_titles[i])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=elev, azim=azim)
            continue

        vals = prob_xyz[coords[:, 0], coords[:, 1], coords[:, 2]]

        if len(coords) > max_points_per_panel:
            sel = rng.choice(len(coords), size=max_points_per_panel, replace=False)
            coords = coords[sel]
            vals = vals[sel]

        xs = coords[:, 0]
        ys = coords[:, 1]
        zs = coords[:, 2]

        scatter_handle = ax.scatter(
            xs, ys, zs,
            c=vals,
            cmap="viridis",
            s=3,
            vmin=0.0,
            vmax=1.0,
            linewidths=0,
            alpha=0.95,
        )

        ax.set_title(stage_titles[i], fontsize=18)
        ax.set_xlabel("X", fontsize=16)
        ax.set_ylabel("Y", fontsize=16)
        ax.set_zlabel("Z", fontsize=16)
        ax.view_init(elev=elev, azim=azim)
        ax.grid(False)

        # cleaner look similar to your current plot
        try:
            ax.set_box_aspect([
                max(1, prob_xyz.shape[0]),
                max(1, prob_xyz.shape[1]),
                max(1, prob_xyz.shape[2]),
            ])
        except Exception:
            pass

    fig.suptitle(main_title, fontsize=22)

    if scatter_handle is not None:
        cbar = fig.colorbar(scatter_handle, ax=fig.axes, shrink=0.65, pad=0.02)
        cbar.set_label("occupancy probability", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plotting / summary
# ============================================================

def plot_case_dice_curve(case_results, out_png):
    import numpy as np
    import matplotlib.pyplot as plt

    case_results = sorted(case_results, key=lambda x: int(x["N_VIEWS"]))
    if len(case_results) == 0:
        return

    xs = [int(r["N_VIEWS"]) for r in case_results]

    recon_mode = str(case_results[0].get("recon_mode", "")).lower()
    init_mode = str(case_results[0].get("pnp_init_mode", "")).lower()

    use_bayes_init = (
        recon_mode in {"pnp_diffusion", "cond_diffusion"}
        and init_mode == "bayes"
        and case_results[0].get("dice_bayes_baseline_fixed_vs_gt") is not None
    )

    use_pnp_init = (
        recon_mode == "pnp_diffusion"
        and case_results[0].get("pnp_init_dice_fixed_vs_gt") is not None
    )

    if use_bayes_init:
        ys_init = [
            r.get("dice_bayes_baseline_fixed_vs_gt", np.nan)
            for r in case_results
        ]
        init_label = "Bayes init"

    elif use_pnp_init:
        ys_init = [
            r.get("pnp_init_dice_fixed_vs_gt", np.nan)
            for r in case_results
        ]
        init_label = f"PnP init: {case_results[0].get('pnp_init_mode', 'init')}"

    else:
        ys_init = [
            r.get("dice_visual_hull_vs_gt", np.nan)
            for r in case_results
        ]
        init_label = "Visual hull"

    ys_final = [
        r.get("dice_final_method_fixed_vs_gt", np.nan)
        for r in case_results
    ]

    if recon_mode == "cond_diffusion":
        final_label = "Conditional diffusion"
    elif recon_mode == "pnp_diffusion":
        final_label = "PnP diffusion"
    elif recon_mode == "bayes_then_diffusion":
        final_label = "Bayes + diffusion"
    elif recon_mode == "bayes":
        final_label = "Bayes posterior"
    else:
        final_label = str(recon_mode)

    plt.figure(figsize=(7, 5))

    plt.plot(xs, ys_init, marker="o", linewidth=2, label=init_label)
    plt.plot(xs, ys_final, marker="o", linewidth=2, label=final_label)

    plt.xticks(xs)
    plt.xlabel("Number of angles")
    plt.ylabel("Dice score")
    plt.title("Dice vs number of angles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    


def plot_gt_vh_final_reconstruction_3d(
    gt_xyz,
    bayes_prob_xyz,
    bayes_fixed_xyz,
    final_prob_xyz,
    final_fixed_xyz,
    out_png,
    prob_thr=0.10,
    max_points_per_panel=30000,
    seed=0,
    elev=25.0,
    azim=45.0,
    final_method_name="Conditional diffusion",
):
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)

    def sample_points(arr, thr=0.5, max_points=30000):
        arr = np.asarray(arr)
        pts = np.argwhere(arr > thr)

        if pts.shape[0] == 0:
            return pts, None

        vals = arr[pts[:, 0], pts[:, 1], pts[:, 2]].astype(np.float32)

        if pts.shape[0] > max_points:
            idx = rng.choice(pts.shape[0], size=max_points, replace=False)
            pts = pts[idx]
            vals = vals[idx]

        return pts, vals

    def setup_ax(ax):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)

    def plot_binary(ax, mask, title, color="gold"):
        pts, _ = sample_points(mask, thr=0.5)

        if pts.shape[0] > 0:
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=1,
                c=color,
                alpha=0.9,
            )

        ax.set_title(title)
        setup_ax(ax)

    def plot_prob(ax, prob, title):
        pts, vals = sample_points(prob, thr=prob_thr)

        sc = None

        if pts.shape[0] > 0:
            sc = ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=1,
                c=vals,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                alpha=0.9,
            )

        ax.set_title(title)
        setup_ax(ax)

        return sc

    def plot_overlap(ax, pred_mask, gt_mask, title):
        pred = np.asarray(pred_mask) > 0
        gt = np.asarray(gt_mask) > 0

        tp = pred & gt
        fp = pred & (~gt)
        fn = (~pred) & gt

        pts_tp, _ = sample_points(tp, thr=0.5)
        pts_fp, _ = sample_points(fp, thr=0.5)
        pts_fn, _ = sample_points(fn, thr=0.5)

        if pts_tp.shape[0] > 0:
            ax.scatter(
                pts_tp[:, 0],
                pts_tp[:, 1],
                pts_tp[:, 2],
                s=1,
                c="green",
                alpha=0.9,
            )

        if pts_fp.shape[0] > 0:
            ax.scatter(
                pts_fp[:, 0],
                pts_fp[:, 1],
                pts_fp[:, 2],
                s=1,
                c="red",
                alpha=0.7,
            )

        if pts_fn.shape[0] > 0:
            ax.scatter(
                pts_fn[:, 0],
                pts_fn[:, 1],
                pts_fn[:, 2],
                s=1,
                c="blue",
                alpha=0.7,
            )

        ax.set_title(title)
        setup_ax(ax)

    gt_bin = np.asarray(gt_xyz) > 0

    bayes_prob = np.asarray(bayes_prob_xyz, dtype=np.float32)
    bayes_bin = np.asarray(bayes_fixed_xyz) > 0

    final_prob = np.asarray(final_prob_xyz, dtype=np.float32)
    final_bin = np.asarray(final_fixed_xyz) > 0

    fig = plt.figure(figsize=(25, 5))

    ax0 = fig.add_subplot(1, 5, 1, projection="3d")
    ax1 = fig.add_subplot(1, 5, 2, projection="3d")
    ax2 = fig.add_subplot(1, 5, 3, projection="3d")
    ax3 = fig.add_subplot(1, 5, 4, projection="3d")
    ax4 = fig.add_subplot(1, 5, 5, projection="3d")

    plot_binary(
        ax0,
        gt_bin,
        "Ground truth",
        color="gold",
    )

    sc1 = plot_prob(
        ax1,
        bayes_prob,
        "Bayes probability",
    )

    plot_overlap(
        ax2,
        pred_mask=bayes_bin,
        gt_mask=gt_bin,
        title="Bayes vs GT\ngreen=TP red=FP blue=FN",
    )

    sc2 = plot_prob(
        ax3,
        final_prob,
        f"{final_method_name} probability",
    )

    plot_overlap(
        ax4,
        pred_mask=final_bin,
        gt_mask=gt_bin,
        title=f"{final_method_name} vs GT\ngreen=TP red=FP blue=FN",
    )

    if sc2 is not None:
        cbar = fig.colorbar(sc2, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label("Occupancy probability")

    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gt_bayes_and_final_reconstruction_3d(
    gt_xyz,
    bayes_prob_xyz,
    bayes_fixed_xyz,
    final_prob_xyz,
    final_fixed_xyz,
    out_png,
    prob_thr=0.10,
    max_points_per_panel=30000,
    seed=0,
    elev=25.0,
    azim=45.0,
    final_method_name="Conditional diffusion",
    init_method_name="Init",
):
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)

    def sample_points(arr, thr=0.5):
        pts = np.argwhere(np.asarray(arr) > thr)
        if pts.shape[0] == 0:
            return pts, None
        vals = np.asarray(arr)[pts[:, 0], pts[:, 1], pts[:, 2]].astype(np.float32)
        if pts.shape[0] > max_points_per_panel:
            idx = rng.choice(pts.shape[0], size=max_points_per_panel, replace=False)
            pts = pts[idx]
            vals = vals[idx]
        return pts, vals

    def setup_ax(ax):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)

    def plot_binary(ax, mask, title, color="gold"):
        pts, _ = sample_points(mask, thr=0.5)
        if pts.shape[0] > 0:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=color, alpha=0.9)
        ax.set_title(title)
        setup_ax(ax)

    def plot_prob(ax, prob, title):
        pts, vals = sample_points(prob, thr=prob_thr)
        sc = None
        if pts.shape[0] > 0:
            sc = ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=1, c=vals, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.9,
            )
        ax.set_title(title)
        setup_ax(ax)
        return sc

    def plot_overlap(ax, pred_mask, gt_mask, title):
        pred = np.asarray(pred_mask) > 0
        gt = np.asarray(gt_mask) > 0

        tp = pred & gt
        fp = pred & (~gt)
        fn = (~pred) & gt

        for arr, color, alpha in [
            (tp, "green", 0.9),
            (fp, "red", 0.7),
            (fn, "blue", 0.7),
        ]:
            pts, _ = sample_points(arr, thr=0.5)
            if pts.shape[0] > 0:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=color, alpha=alpha)

        ax.set_title(title)
        setup_ax(ax)

    gt_bin = np.asarray(gt_xyz) > 0
    init_prob = np.asarray(bayes_prob_xyz, dtype=np.float32)
    init_bin = np.asarray(bayes_fixed_xyz) > 0
    final_prob = np.asarray(final_prob_xyz, dtype=np.float32)
    final_bin = np.asarray(final_fixed_xyz) > 0

    fig = plt.figure(figsize=(25, 5))

    ax0 = fig.add_subplot(1, 5, 1, projection="3d")
    ax1 = fig.add_subplot(1, 5, 2, projection="3d")
    ax2 = fig.add_subplot(1, 5, 3, projection="3d")
    ax3 = fig.add_subplot(1, 5, 4, projection="3d")
    ax4 = fig.add_subplot(1, 5, 5, projection="3d")

    plot_binary(ax0, gt_bin, "Ground truth")
    sc1 = plot_prob(ax1, init_prob, f"{init_method_name} probability")
    plot_overlap(ax2, init_bin, gt_bin, f"{init_method_name} vs GT\ngreen=TP red=FP blue=FN")
    sc2 = plot_prob(ax3, final_prob, f"{final_method_name} probability")
    plot_overlap(ax4, final_bin, gt_bin, f"{final_method_name} vs GT\ngreen=TP red=FP blue=FN")

    if sc2 is not None:
        cbar = fig.colorbar(sc2, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label("Occupancy probability")

    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)