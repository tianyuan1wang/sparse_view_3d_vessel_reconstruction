import numpy as np
from skimage.measure import marching_cubes


def write_ply_mesh(path, verts, faces):
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(verts)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write(f'element face {len(faces)}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')

        for v in verts:
            f.write(f'{v[0]} {v[1]} {v[2]}\n')

        for tri in faces:
            f.write(f'3 {tri[0]} {tri[1]} {tri[2]}\n')


def write_ply_points(path, pts):
    pts = np.asarray(pts, dtype=np.float32)

    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(pts)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')

        for p in pts:
            f.write(f'{p[0]} {p[1]} {p[2]}\n')


def mesh_from_binary(mask3d, spacing_xyz):
    mask3d = np.asarray(mask3d, dtype=np.uint8)
    if mask3d.sum() == 0:
        return None, None

    verts, faces, normals, values = marching_cubes(
        volume=mask3d.astype(np.float32),
        level=0.5,
        spacing=tuple(float(x) for x in spacing_xyz),
    )
    return verts.astype(np.float32), faces.astype(np.int32)


def sample_points_from_mesh(verts, faces, n_points=20000, seed=0):
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    if len(faces) == 0:
        return verts.copy()

    tri = verts[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    good = areas > 1e-12
    if not np.any(good):
        return verts.copy()

    tri = tri[good]
    areas = areas[good]
    probs = areas / areas.sum()

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(tri), size=int(n_points), replace=True, p=probs)

    t = tri[idx]
    r1 = np.sqrt(rng.random((len(t), 1), dtype=np.float32))
    r2 = rng.random((len(t), 1), dtype=np.float32)

    pts = (1.0 - r1) * t[:, 0] + r1 * (1.0 - r2) * t[:, 1] + r1 * r2 * t[:, 2]
    return pts.astype(np.float32)
