import torch
import tomosipo_compat
import tomosipo as ts
import numpy as np


@torch.no_grad()
def project_label_from_mask(mask_zyx, vg, pg_k, device='cpu', proj_eps=1e-8):
    """
    mask_zyx must be in tomosipo order: (Z, Y, X)
    Returns a 2D detector mask [H, W].
    """
    mask_t = torch.from_numpy(np.asarray(mask_zyx, dtype=np.float32)).to(
        device=device, dtype=torch.float32
    )
    A_k = ts.operator(vg, pg_k)
    raw = A_k(mask_t)
    p = raw.squeeze()

    if p.ndim != 2:
        raise RuntimeError(
            f"Unexpected projector output shape: raw={tuple(raw.shape)}, squeezed={tuple(p.shape)}"
        )

    y = (p > float(proj_eps)).detach().cpu().numpy().astype(np.uint8)
    return y


@torch.no_grad()
def generate_multiview_masks_from_label(mask_zyx, vg, pg, device='cpu', proj_eps=1e-8):
    masks_2d = []
    for v in range(len(pg)):
        y = project_label_from_mask(mask_zyx, vg, pg[v], device=device, proj_eps=proj_eps)
        masks_2d.append(y)
    return masks_2d
