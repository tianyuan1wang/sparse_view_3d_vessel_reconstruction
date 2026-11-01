import numpy as np
import torch


@torch.no_grad()
def reconstruct_visual_hull_from_masks_tomosipo(
    masks_2d,
    pg,
    vol_shape_zyx,
    spacing_zyx,
    allow_missing_views=0,
):
    n_views = len(masks_2d)
    if n_views == 0:
        raise ValueError('No 2D masks provided.')

    Nz, Ny, Nx = [int(x) for x in vol_shape_zyx]
    sz, sy, sx = [float(x) for x in spacing_zyx]

    src_all = np.asarray(pg.src_pos, dtype=np.float32)
    det_pos_all = np.asarray(pg.det_pos, dtype=np.float32)
    det_v_all = np.asarray(pg.det_v, dtype=np.float32)
    det_u_all = np.asarray(pg.det_u, dtype=np.float32)
    llc_all = np.asarray(pg.lower_left_corner, dtype=np.float32)

    if src_all.shape[0] != n_views:
        raise ValueError(f'pg has {src_all.shape[0]} views, but masks_2d has {n_views} views.')

    zs = (np.arange(Nz, dtype=np.float32) + 0.5 - Nz / 2.0) * sz
    ys = (np.arange(Ny, dtype=np.float32) + 0.5 - Ny / 2.0) * sy
    xs = (np.arange(Nx, dtype=np.float32) + 0.5 - Nx / 2.0) * sx

    YY, XX = np.meshgrid(ys, xs, indexing='ij')
    votes = np.zeros((Nz, Ny, Nx), dtype=np.uint16)

    for v in range(n_views):
        mask_hw = (np.asarray(masks_2d[v]) > 0)
        det_h, det_w = mask_hw.shape

        src = src_all[v].reshape(3)
        det_pos = det_pos_all[v].reshape(3)
        det_v = det_v_all[v].reshape(3)
        det_u = det_u_all[v].reshape(3)
        llc = llc_all[v].reshape(3)

        n = np.cross(det_v, det_u).astype(np.float32)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            raise ValueError(f'View {v}: detector basis vectors are degenerate.')
        n = n / n_norm

        plane_num = float(np.dot(det_pos - src, n))

        vv_denom = float(np.dot(det_v, det_v))
        uu_denom = float(np.dot(det_u, det_u))
        if vv_denom < 1e-12 or uu_denom < 1e-12:
            raise ValueError(f'View {v}: invalid det_v / det_u norms.')

        for iz, zw in enumerate(zs):
            Z = np.full_like(YY, zw, dtype=np.float32)
            Y = YY
            X = XX

            rz = Z - src[0]
            ry = Y - src[1]
            rx = X - src[2]

            denom = rz * n[0] + ry * n[1] + rx * n[2]

            valid = np.abs(denom) > 1e-12
            t = np.zeros_like(denom, dtype=np.float32)
            t[valid] = plane_num / denom[valid]
            valid &= (t > 0.0)

            Iz = src[0] + t * rz
            Iy = src[1] + t * ry
            Ix = src[2] + t * rx

            relz = Iz - llc[0]
            rely = Iy - llc[1]
            relx = Ix - llc[2]

            v_pix = (relz * det_v[0] + rely * det_v[1] + relx * det_v[2]) / vv_denom
            u_pix = (relz * det_u[0] + rely * det_u[1] + relx * det_u[2]) / uu_denom

            ii = np.rint(v_pix).astype(np.int32)
            jj = np.rint(u_pix).astype(np.int32)

            inside = (
                valid &
                (ii >= 0) & (ii < det_h) &
                (jj >= 0) & (jj < det_w)
            )

            support = np.zeros((Ny, Nx), dtype=np.uint8)
            if np.any(inside):
                support[inside] = mask_hw[ii[inside], jj[inside]].astype(np.uint8)

            votes[iz] += support

    required = max(int(n_views) - int(allow_missing_views), 1)
    vh_zyx = (votes >= required).astype(np.uint8)
    return vh_zyx, votes.astype(np.uint16)
