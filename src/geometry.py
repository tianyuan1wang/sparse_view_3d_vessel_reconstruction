import numpy as np

import tomosipo_compat
import tomosipo as ts


def _normalize_rows(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, float(eps))
    return (x / n).astype(np.float32)


def _xyz_to_zyx(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return arr[..., [2, 1, 0]].astype(np.float32)


def rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    axis = axis.lower()
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    raise ValueError("axis must be 'x', 'y', or 'z'")


def build_circular_trajectory_mm(
    n_views: int,
    R_SO_MM: float,
    plane: str = 'xy',
    center_mm=(0.0, 0.0, 0.0),
    start_angle_rad: float = 0.0,
    end_angle_rad: float = 2 * np.pi,
    endpoint: bool = False,
    tilt_deg: float = 0.0,
    tilt_axis: str = 'x',
):
    plane = plane.lower()
    center_mm = np.asarray(center_mm, dtype=np.float32).reshape(3)
    cx, cy, cz = [float(v) for v in center_mm]

    angles = np.linspace(
        start_angle_rad, end_angle_rad, n_views,
        endpoint=endpoint, dtype=np.float32
    )
    R = float(R_SO_MM)

    if plane == 'xy':
        S = np.stack([R * np.cos(angles), R * np.sin(angles), np.zeros_like(angles)], axis=1)
    elif plane == 'xz':
        S = np.stack([R * np.cos(angles), np.zeros_like(angles), R * np.sin(angles)], axis=1)
    elif plane == 'yz':
        S = np.stack([np.zeros_like(angles), R * np.cos(angles), R * np.sin(angles)], axis=1)
    else:
        raise ValueError("plane must be one of {'xy','xz','yz'}")

    if abs(float(tilt_deg)) > 1e-6:
        Rm = rotation_matrix(tilt_axis, np.deg2rad(float(tilt_deg)))
        S = (S @ Rm.T).astype(np.float32)

    S[:, 0] += cx
    S[:, 1] += cy
    S[:, 2] += cz
    return S.astype(np.float32), angles


def build_cone_vec_from_world_sources(
    S_mm,
    O_mm,
    d_OD_mm,
    det_shape=(512, 512),
    det_size=(300.0, 300.0),
    up_world=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    assume_tomosipo_zyx=True,
):
    """
    Self-contained replacement for util_recon.build_cone_vec_from_world_sources.

    Parameters
    ----------
    S_mm : ndarray, shape (V, 3)
        Source positions in world XYZ coordinates, in mm.
    O_mm : ndarray, shape (3,)
        Isocenter / object center in world XYZ coordinates, in mm.
    d_OD_mm : float
        Distance from isocenter to detector center along the central ray.
    det_shape : tuple(int, int)
        Detector shape as (height, width).
    det_size : tuple(float, float)
        Detector size in mm as (height_mm, width_mm).
    up_world : ndarray, shape (3,)
        Preferred detector up direction in world XYZ coordinates.
    assume_tomosipo_zyx : bool
        If True, convert XYZ coordinates to ZYX before calling tomosipo.

    Returns
    -------
    pg : tomosipo cone_vec geometry
    det_centers_mm : ndarray, shape (V, 3)
        Detector centers in world XYZ coordinates.
    """
    S_mm = np.asarray(S_mm, dtype=np.float32)
    O_mm = np.asarray(O_mm, dtype=np.float32).reshape(3)
    up_world = np.asarray(up_world, dtype=np.float32).reshape(3)

    rays = O_mm[None, :] - S_mm
    ray_dirs = _normalize_rows(rays)

    det_centers_mm = O_mm[None, :] + float(d_OD_mm) * ray_dirs

    up = np.repeat(up_world[None, :], len(S_mm), axis=0).astype(np.float32)
    det_u = np.cross(ray_dirs, up)
    bad = np.linalg.norm(det_u, axis=1) < 1e-8
    if np.any(bad):
        alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        up[bad] = alt_up[None, :]
        det_u = np.cross(ray_dirs, up)

    det_u = _normalize_rows(det_u)
    det_v = _normalize_rows(np.cross(det_u, ray_dirs))

    det_h, det_w = int(det_shape[0]), int(det_shape[1])
    size_h, size_w = float(det_size[0]), float(det_size[1])

    det_u = det_u * (size_w / det_w)
    det_v = det_v * (size_h / det_h)

    src_pos = S_mm
    det_pos = det_centers_mm

    if assume_tomosipo_zyx:
        src_pos_ts = _xyz_to_zyx(src_pos)
        det_pos_ts = _xyz_to_zyx(det_pos)
        det_u_ts = _xyz_to_zyx(det_u)
        det_v_ts = _xyz_to_zyx(det_v)
    else:
        src_pos_ts = src_pos.astype(np.float32)
        det_pos_ts = det_pos.astype(np.float32)
        det_u_ts = det_u.astype(np.float32)
        det_v_ts = det_v.astype(np.float32)

    pg = ts.cone_vec(
        shape=(det_h, det_w),
        src_pos=src_pos_ts,
        det_pos=det_pos_ts,
        det_v=det_v_ts,
        det_u=det_u_ts,
    )
    return pg, det_centers_mm


