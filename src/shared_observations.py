"""
Shared 2-D observation I/O for fair method comparison.

This module makes CondDiff, DIP, and NeCA consume exactly the same 2-D data:
  1) masks_2d.npz             binary masks used by visual hull
  2) recon_masks_2d.npz       float masks used by the reconstruction loss
  3) mask_confidence_maps.npz float confidence/weight maps

New bundles can additionally freeze the upstream evidence used to create those
inputs, without changing the legacy loader API:
  4) xrays_2d.npz                         noisy X-ray realization (float32)
  5) segmentation_probabilities_2d.npz    soft segmentation output (float32)
  6) gt_projection_masks_2d.npz           projection ground truth (uint8)

The shared observations are intended to be saved AFTER optional detector
resizing (for example --lowres_recon_det_shape 256 256), so all methods use
exactly the same final arrays and exactly the same projector detector size.

Directory layout:
  <shared_obs_root>/<case_id>/views_<N:03d>/
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


VALID_SHARED_OBS_MODES = {"none", "off", "false", "0", "save", "overwrite", "write", "save_if_missing", "create_if_missing", "load", "read"}


def normalize_shared_obs_mode(mode: Optional[str]) -> str:
    m = str(mode or "none").strip().lower()
    if m not in VALID_SHARED_OBS_MODES:
        raise ValueError(
            f"Unknown shared_obs_mode={mode!r}. Valid modes: none, save, overwrite, save_if_missing, load."
        )
    if m in {"off", "false", "0"}:
        return "none"
    if m == "write":
        return "overwrite"
    if m == "read":
        return "load"
    if m == "create_if_missing":
        return "save_if_missing"
    return m


def shared_obs_is_active(shared_obs_root: Optional[str], shared_obs_mode: Optional[str]) -> bool:
    return shared_obs_root not in [None, "", "None"] and normalize_shared_obs_mode(shared_obs_mode) != "none"


def shared_obs_is_load(shared_obs_root: Optional[str], shared_obs_mode: Optional[str]) -> bool:
    return shared_obs_root not in [None, "", "None"] and normalize_shared_obs_mode(shared_obs_mode) == "load"


def shared_obs_should_save(shared_obs_root: Optional[str], shared_obs_mode: Optional[str]) -> bool:
    if shared_obs_root in [None, "", "None"]:
        return False
    return normalize_shared_obs_mode(shared_obs_mode) in {"save", "overwrite", "save_if_missing"}


def get_shared_obs_dir(shared_obs_root: str, case_id: str, n_views: int) -> str:
    return os.path.join(str(shared_obs_root), str(case_id), f"views_{int(n_views):03d}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_npz_list(path: str, arrays: Sequence[np.ndarray], dtype) -> None:
    """Atomically replace one compressed array bundle.

    The temporary file lives beside the destination so ``os.replace`` stays on
    one filesystem.  ``metadata.json`` is written last by the public saver and
    therefore acts as the commit record for a complete observation bundle.
    """
    parent = os.path.dirname(os.path.abspath(path))
    _ensure_dir(parent)
    fd, tmp_path = tempfile.mkstemp(prefix=".shared_obs_", suffix=".npz", dir=parent)
    os.close(fd)
    try:
        np.savez_compressed(
            tmp_path,
            **{
                f"view_{i:03d}": np.asarray(a, dtype=dtype)
                for i, a in enumerate(arrays)
            },
        )
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _write_json_atomic(path: str, value: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    _ensure_dir(parent)
    fd, tmp_path = tempfile.mkstemp(prefix=".shared_obs_", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _load_npz_list(path: str) -> List[np.ndarray]:
    with np.load(path) as data:
        return [np.asarray(data[k]) for k in sorted(data.files)]


def _arrays_sha256(arrays: Sequence[np.ndarray]) -> str:
    h = hashlib.sha256()
    for a in arrays:
        arr = np.ascontiguousarray(np.asarray(a))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _validate_lists(
    masks_2d: Sequence[np.ndarray],
    recon_masks_2d: Sequence[np.ndarray],
    mask_weights_2d: Sequence[np.ndarray],
    n_views: int,
    obs_dir: str,
) -> None:
    if len(masks_2d) != int(n_views):
        raise RuntimeError(f"Shared observations in {obs_dir} have {len(masks_2d)} masks, expected {int(n_views)}")
    if len(recon_masks_2d) != int(n_views):
        raise RuntimeError(f"Shared observations in {obs_dir} have {len(recon_masks_2d)} recon masks, expected {int(n_views)}")
    if len(mask_weights_2d) != int(n_views):
        raise RuntimeError(f"Shared observations in {obs_dir} have {len(mask_weights_2d)} weights, expected {int(n_views)}")

    for i, (m, r, w) in enumerate(zip(masks_2d, recon_masks_2d, mask_weights_2d)):
        if np.asarray(m).ndim != 2 or np.asarray(r).ndim != 2 or np.asarray(w).ndim != 2:
            raise RuntimeError(f"Shared observation view {i} in {obs_dir} is not 2-D")
        if tuple(np.asarray(m).shape) != tuple(np.asarray(r).shape):
            raise RuntimeError(
                f"Shared observation view {i} shape mismatch in {obs_dir}: "
                f"mask={np.asarray(m).shape}, recon={np.asarray(r).shape}"
            )
        if tuple(np.asarray(r).shape) != tuple(np.asarray(w).shape):
            raise RuntimeError(
                f"Shared observation view {i} weight shape mismatch in {obs_dir}: "
                f"recon={np.asarray(r).shape}, weight={np.asarray(w).shape}"
            )


def _validate_optional_2d_list(
    arrays: Optional[Sequence[np.ndarray]],
    n_views: int,
    obs_dir: str,
    field_name: str,
) -> None:
    if arrays is None:
        return
    if len(arrays) != int(n_views):
        raise RuntimeError(
            f"Shared observations in {obs_dir} have {len(arrays)} {field_name} "
            f"arrays, expected {int(n_views)}"
        )

    expected_shape = None
    for i, array in enumerate(arrays):
        arr = np.asarray(array)
        if arr.ndim != 2:
            raise RuntimeError(
                f"Shared observation {field_name} view {i} in {obs_dir} is not 2-D"
            )
        if expected_shape is None:
            expected_shape = tuple(arr.shape)
        elif tuple(arr.shape) != expected_shape:
            raise RuntimeError(
                f"Shared observation {field_name} view {i} shape mismatch in "
                f"{obs_dir}: got {tuple(arr.shape)}, expected {expected_shape}"
            )


def _validate_auxiliary_shapes(
    xrays_2d: Optional[Sequence[np.ndarray]],
    segmentation_probabilities_2d: Optional[Sequence[np.ndarray]],
    gt_projection_masks_2d: Optional[Sequence[np.ndarray]],
    n_views: int,
    obs_dir: str,
) -> None:
    fields = {
        "X-ray": xrays_2d,
        "segmentation-probability": segmentation_probabilities_2d,
        "GT-projection-mask": gt_projection_masks_2d,
    }
    for name, arrays in fields.items():
        _validate_optional_2d_list(arrays, n_views, obs_dir, name)

    # These three arrays describe the same detector pixels and therefore must
    # share a shape when more than one auxiliary field is present.
    shapes = {
        name: tuple(np.asarray(arrays[0]).shape)
        for name, arrays in fields.items()
        if arrays is not None and len(arrays) > 0
    }
    if len(set(shapes.values())) > 1:
        raise RuntimeError(
            f"Shared auxiliary observation shape mismatch in {obs_dir}: {shapes}"
        )


def _validate_finite_arrays(
    arrays: Sequence[np.ndarray],
    obs_dir: str,
    field_name: str,
) -> None:
    for i, array in enumerate(arrays):
        if not np.all(np.isfinite(np.asarray(array))):
            raise RuntimeError(
                f"Shared observation {field_name} view {i} in {obs_dir} "
                "contains non-finite values"
            )


def save_shared_observations(
    shared_obs_root: str,
    case_id: str,
    n_views: int,
    masks_2d: Sequence[np.ndarray],
    recon_masks_2d: Sequence[np.ndarray],
    mask_weights_2d: Optional[Sequence[np.ndarray]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
    xrays_2d: Optional[Sequence[np.ndarray]] = None,
    segmentation_probabilities_2d: Optional[Sequence[np.ndarray]] = None,
    gt_projection_masks_2d: Optional[Sequence[np.ndarray]] = None,
) -> str:
    """Save final 2-D observations for one case and one view count.

    The three auxiliary inputs are optional for backward compatibility.  New
    experiments can supply them to freeze the exact noisy X-ray realization,
    continuous segmentation predictions, and projection ground truth together
    with the reconstruction inputs.
    """
    obs_dir = get_shared_obs_dir(shared_obs_root, case_id, n_views)
    _ensure_dir(obs_dir)

    masks_path = os.path.join(obs_dir, "masks_2d.npz")
    recon_path = os.path.join(obs_dir, "recon_masks_2d.npz")
    weights_path = os.path.join(obs_dir, "mask_confidence_maps.npz")
    xrays_path = os.path.join(obs_dir, "xrays_2d.npz")
    segmentation_probabilities_path = os.path.join(
        obs_dir, "segmentation_probabilities_2d.npz"
    )
    gt_projection_masks_path = os.path.join(obs_dir, "gt_projection_masks_2d.npz")

    if (not overwrite) and os.path.exists(masks_path) and os.path.exists(recon_path):
        print(f"  [shared obs] keep existing observations: {obs_dir}")
        return obs_dir

    if mask_weights_2d is None:
        mask_weights_2d = [np.ones_like(np.asarray(m, dtype=np.float32), dtype=np.float32) for m in recon_masks_2d]

    _validate_lists(masks_2d, recon_masks_2d, mask_weights_2d, int(n_views), obs_dir)
    _validate_auxiliary_shapes(
        xrays_2d=xrays_2d,
        segmentation_probabilities_2d=segmentation_probabilities_2d,
        gt_projection_masks_2d=gt_projection_masks_2d,
        n_views=int(n_views),
        obs_dir=obs_dir,
    )
    for field_name, arrays in (
        ("X-ray", xrays_2d),
        ("segmentation-probability", segmentation_probabilities_2d),
        ("GT-projection-mask", gt_projection_masks_2d),
    ):
        if arrays is None:
            continue
        for view_index, (array, mask) in enumerate(zip(arrays, masks_2d)):
            if tuple(np.asarray(array).shape) != tuple(np.asarray(mask).shape):
                raise RuntimeError(
                    f"Shared {field_name} view {view_index} in {obs_dir} has shape "
                    f"{np.asarray(array).shape}, but the reconstruction mask has "
                    f"shape {np.asarray(mask).shape}"
                )

    masks_u8 = [(np.asarray(m) > 0).astype(np.uint8) for m in masks_2d]
    recon_f32 = [np.clip(np.asarray(m, dtype=np.float32), 0.0, 1.0).astype(np.float32) for m in recon_masks_2d]
    weights_f32 = [np.asarray(w, dtype=np.float32).astype(np.float32) for w in mask_weights_2d]
    xrays_f32 = (
        None
        if xrays_2d is None
        else [np.asarray(x, dtype=np.float32).astype(np.float32) for x in xrays_2d]
    )
    segmentation_probabilities_f32 = (
        None
        if segmentation_probabilities_2d is None
        else [
            np.asarray(probability, dtype=np.float32).astype(np.float32)
            for probability in segmentation_probabilities_2d
        ]
    )
    gt_projection_masks_u8 = (
        None
        if gt_projection_masks_2d is None
        else [
            (np.asarray(mask) > 0).astype(np.uint8)
            for mask in gt_projection_masks_2d
        ]
    )

    _validate_finite_arrays(weights_f32, obs_dir, "confidence-map")
    if xrays_f32 is not None:
        _validate_finite_arrays(xrays_f32, obs_dir, "X-ray")
    if segmentation_probabilities_f32 is not None:
        _validate_finite_arrays(
            segmentation_probabilities_f32,
            obs_dir,
            "segmentation-probability",
        )
        for i, probability in enumerate(segmentation_probabilities_f32):
            if np.any(probability < 0.0) or np.any(probability > 1.0):
                raise RuntimeError(
                    f"Shared segmentation-probability view {i} in {obs_dir} "
                    "contains values outside [0, 1]"
                )

    _save_npz_list(masks_path, masks_u8, np.uint8)
    _save_npz_list(recon_path, recon_f32, np.float32)
    _save_npz_list(weights_path, weights_f32, np.float32)
    if xrays_f32 is not None:
        _save_npz_list(xrays_path, xrays_f32, np.float32)
    if segmentation_probabilities_f32 is not None:
        _save_npz_list(
            segmentation_probabilities_path,
            segmentation_probabilities_f32,
            np.float32,
        )
    if gt_projection_masks_u8 is not None:
        _save_npz_list(gt_projection_masks_path, gt_projection_masks_u8, np.uint8)

    meta = dict(metadata or {})
    meta.update(
        {
            "shared_observation_schema_version": 2,
            "case_id": str(case_id),
            "N_VIEWS": int(n_views),
            "num_views_saved": int(len(masks_u8)),
            "final_detector_shape_hw": list(np.asarray(masks_u8[0]).shape) if len(masks_u8) else None,
            "masks_2d_sha256": _arrays_sha256(masks_u8),
            "recon_masks_2d_sha256": _arrays_sha256(recon_f32),
            "mask_weights_2d_sha256": _arrays_sha256(weights_f32),
            "xrays_2d_sha256": (
                None if xrays_f32 is None else _arrays_sha256(xrays_f32)
            ),
            "segmentation_probabilities_2d_sha256": (
                None
                if segmentation_probabilities_f32 is None
                else _arrays_sha256(segmentation_probabilities_f32)
            ),
            "gt_projection_masks_2d_sha256": (
                None
                if gt_projection_masks_u8 is None
                else _arrays_sha256(gt_projection_masks_u8)
            ),
            "auxiliary_arrays_present": {
                "xrays_2d": xrays_f32 is not None,
                "segmentation_probabilities_2d": (
                    segmentation_probabilities_f32 is not None
                ),
                "gt_projection_masks_2d": gt_projection_masks_u8 is not None,
            },
            "auxiliary_detector_shape_hw": (
                None
                if xrays_f32 is None
                and segmentation_probabilities_f32 is None
                and gt_projection_masks_u8 is None
                else list(
                    np.asarray(
                        next(
                            arrays[0]
                            for arrays in (
                                xrays_f32,
                                segmentation_probabilities_f32,
                                gt_projection_masks_u8,
                            )
                            if arrays is not None
                        )
                    ).shape
                )
            ),
        }
    )
    _write_json_atomic(os.path.join(obs_dir, "metadata.json"), meta)

    print(f"  [shared obs] saved observations: {obs_dir}")
    return obs_dir


def load_shared_observations(
    shared_obs_root: str,
    case_id: str,
    n_views: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Dict[str, Any], str]:
    """Load final 2-D observations for one case and one view count."""
    obs_dir = get_shared_obs_dir(shared_obs_root, case_id, n_views)
    masks_path = os.path.join(obs_dir, "masks_2d.npz")
    recon_path = os.path.join(obs_dir, "recon_masks_2d.npz")
    weights_path = os.path.join(obs_dir, "mask_confidence_maps.npz")
    meta_path = os.path.join(obs_dir, "metadata.json")

    missing = [p for p in [masks_path, recon_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing shared 2-D observation file(s): "
            + ", ".join(missing)
            + ". Run one method first with --shared_obs_mode overwrite/save."
        )

    masks_2d = [(a > 0).astype(np.uint8) for a in _load_npz_list(masks_path)]
    recon_masks_2d = [np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0).astype(np.float32) for a in _load_npz_list(recon_path)]
    if os.path.exists(weights_path):
        mask_weights_2d = [np.asarray(a, dtype=np.float32).astype(np.float32) for a in _load_npz_list(weights_path)]
    else:
        mask_weights_2d = [np.ones_like(a, dtype=np.float32) for a in recon_masks_2d]

    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    _validate_lists(masks_2d, recon_masks_2d, mask_weights_2d, int(n_views), obs_dir)

    # Optional consistency check against stored checksums.
    if meta.get("masks_2d_sha256") and meta["masks_2d_sha256"] != _arrays_sha256(masks_2d):
        raise RuntimeError(f"Shared mask checksum mismatch in {obs_dir}")
    if meta.get("recon_masks_2d_sha256") and meta["recon_masks_2d_sha256"] != _arrays_sha256(recon_masks_2d):
        raise RuntimeError(f"Shared recon mask checksum mismatch in {obs_dir}")
    if meta.get("mask_weights_2d_sha256") and meta["mask_weights_2d_sha256"] != _arrays_sha256(mask_weights_2d):
        raise RuntimeError(f"Shared confidence-map checksum mismatch in {obs_dir}")

    print(f"  [shared obs] loaded observations: {obs_dir}")
    return masks_2d, recon_masks_2d, mask_weights_2d, meta, obs_dir


def load_shared_observation_auxiliaries(
    shared_obs_root: str,
    case_id: str,
    n_views: int,
    require_complete: bool = False,
) -> Dict[str, Any]:
    """Load optional X-ray/segmentation/GT-projection arrays.

    This is deliberately separate from :func:`load_shared_observations`, whose
    established five-value return API remains unchanged.  Legacy bundles return
    ``None`` for absent auxiliary fields unless ``require_complete=True``.
    The returned dictionary also includes ``metadata`` and ``obs_dir`` for
    provenance checks.
    """
    obs_dir = get_shared_obs_dir(shared_obs_root, case_id, n_views)
    paths = {
        "xrays_2d": os.path.join(obs_dir, "xrays_2d.npz"),
        "segmentation_probabilities_2d": os.path.join(
            obs_dir, "segmentation_probabilities_2d.npz"
        ),
        "gt_projection_masks_2d": os.path.join(
            obs_dir, "gt_projection_masks_2d.npz"
        ),
    }
    meta_path = os.path.join(obs_dir, "metadata.json")

    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if require_complete and missing:
        raise FileNotFoundError(
            f"Missing shared auxiliary observation file(s) in {obs_dir}: "
            + ", ".join(missing)
        )
    if (
        meta.get("final_detector_shape_hw") is not None
        and meta.get("auxiliary_detector_shape_hw") is not None
        and list(meta["final_detector_shape_hw"])
        != list(meta["auxiliary_detector_shape_hw"])
    ):
        raise RuntimeError(
            f"Shared base/auxiliary detector shapes disagree in {obs_dir}: "
            f"{meta['final_detector_shape_hw']} vs "
            f"{meta['auxiliary_detector_shape_hw']}"
        )

    checksum_keys = {
        "xrays_2d": "xrays_2d_sha256",
        "segmentation_probabilities_2d": "segmentation_probabilities_2d_sha256",
        "gt_projection_masks_2d": "gt_projection_masks_2d_sha256",
    }
    for name in missing:
        if meta.get(checksum_keys[name]):
            raise FileNotFoundError(
                f"Shared metadata declares {name}, but its file is missing in {obs_dir}"
            )

    xrays_2d = (
        None
        if not os.path.exists(paths["xrays_2d"])
        else [
            np.asarray(array, dtype=np.float32).astype(np.float32)
            for array in _load_npz_list(paths["xrays_2d"])
        ]
    )
    segmentation_probabilities_2d = (
        None
        if not os.path.exists(paths["segmentation_probabilities_2d"])
        else [
            np.asarray(array, dtype=np.float32).astype(np.float32)
            for array in _load_npz_list(paths["segmentation_probabilities_2d"])
        ]
    )
    gt_projection_masks_2d = (
        None
        if not os.path.exists(paths["gt_projection_masks_2d"])
        else [
            (np.asarray(array) > 0).astype(np.uint8)
            for array in _load_npz_list(paths["gt_projection_masks_2d"])
        ]
    )

    _validate_auxiliary_shapes(
        xrays_2d=xrays_2d,
        segmentation_probabilities_2d=segmentation_probabilities_2d,
        gt_projection_masks_2d=gt_projection_masks_2d,
        n_views=int(n_views),
        obs_dir=obs_dir,
    )
    if xrays_2d is not None:
        _validate_finite_arrays(xrays_2d, obs_dir, "X-ray")
    if segmentation_probabilities_2d is not None:
        _validate_finite_arrays(
            segmentation_probabilities_2d,
            obs_dir,
            "segmentation-probability",
        )

    loaded: Dict[str, Any] = {
        "xrays_2d": xrays_2d,
        "segmentation_probabilities_2d": segmentation_probabilities_2d,
        "gt_projection_masks_2d": gt_projection_masks_2d,
    }
    for name, arrays in loaded.items():
        expected = meta.get(checksum_keys[name])
        if arrays is not None and expected and expected != _arrays_sha256(arrays):
            raise RuntimeError(
                f"Shared {name.replace('_', ' ')} checksum mismatch in {obs_dir}"
            )

    print(f"  [shared obs] loaded auxiliary observations: {obs_dir}")
    loaded["metadata"] = meta
    loaded["obs_dir"] = obs_dir
    return loaded
