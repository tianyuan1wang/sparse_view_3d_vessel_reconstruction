"""Mask preparation used by the current DiffCond runs and older baselines."""
import hashlib

import numpy as np
from scipy import ndimage as ndi


def resize_training_mask(mask, out_shape=(128, 128, 128)):
    """Match the bundled diffusion trainer's float32, whole-volume mask loader."""
    mask = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
    zoom = [out_shape[i] / mask.shape[i] for i in range(3)]
    out = ndi.zoom(mask, zoom=zoom, order=0)
    if out.shape != tuple(out_shape):
        fixed = np.zeros(out_shape, dtype=np.float32)
        size = np.minimum(np.array(out.shape), np.array(out_shape))
        fixed[:size[0], :size[1], :size[2]] = out[:size[0], :size[1], :size[2]]
        out = fixed
    return (out > 0.5).astype(np.float32)


def validate_training_preparation(*, crop_bbox_xyz, bbox_pad, exact_grid_shape_xyz,
                                  exact_mask_resample_mode, exact_mask_eval_threshold,
                                  min_vox_component):
    if crop_bbox_xyz is not None or int(bbox_pad) != 0:
        raise ValueError("training_matched preprocessing requires the whole native volume and bbox_pad=0")
    if exact_grid_shape_xyz is None or tuple(exact_grid_shape_xyz) != (128, 128, 128):
        raise ValueError("training_matched preprocessing requires an exact 128 x 128 x 128 grid")
    if exact_mask_resample_mode != "nearest" or float(exact_mask_eval_threshold) != 0.5:
        raise ValueError("training_matched preprocessing requires nearest mask resizing and threshold 0.5")
    if int(min_vox_component) != 50:
        raise ValueError("The reconstruction-output component cutoff must remain 50 voxels")


def observation_target_metadata(prep):
    """Bind new observation caches to their target and preprocessing choice."""
    target = np.ascontiguousarray(prep['mask_xyz'], dtype=np.uint8)
    return {
        'mask_preprocessing': prep.get('mask_preprocessing', 'legacy'),
        'target_mask_sha256': hashlib.sha256(target.tobytes()).hexdigest(),
    }


def validate_observation_target(metadata, prep):
    expected = observation_target_metadata(prep)
    # Old baseline caches had no preprocessing tag. They are accepted only by
    # the legacy path; new DiffCond caches must state their target explicitly.
    if 'mask_preprocessing' not in metadata and expected['mask_preprocessing'] == 'legacy':
        return
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Shared observations have incompatible {key}; use a matching observation cache")
