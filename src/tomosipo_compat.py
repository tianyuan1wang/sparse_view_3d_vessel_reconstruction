import numpy as np
import tomosipo as ts

_orig_to_vec = ts.types.to_vec

def _patched_to_vec(vec, var_name="vector"):
    if np.isscalar(vec):
        val = float(vec)
        return np.array([[val, val, val]], dtype=np.float64)

    arr = np.asarray(vec, dtype=np.float64)

    if arr.shape == (3,):
        return arr.reshape(1, 3)

    if arr.ndim == 2 and arr.shape[1] == 3:
        return np.ascontiguousarray(arr, dtype=np.float64)

    return _orig_to_vec(vec, var_name)

ts.types.to_vec = _patched_to_vec
