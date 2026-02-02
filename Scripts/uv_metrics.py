#################################### Usage ####################################
# python uv_metrics.py --obj "path/to/obj"
#################################################################################

import argparse
import math
from collections import defaultdict
import numpy as np

EPS = 1e-12


# OBJ loader (minimal)
def load_obj_unified_vertices(path):
    verts = []
    uvs = []
    raw_faces = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v' and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vt' and len(parts) >= 3:
                u = float(parts[1]); v = float(parts[2])
                uvs.append([u, v])
            elif parts[0] == 'f':
                face = []
                for tok in parts[1:]:
                    comps = tok.split('/')
                    vi = int(comps[0]) if comps[0] != '' else None
                    vti = int(comps[1]) if len(comps) >= 2 and comps[1] != '' else None
                    vni = int(comps[2]) if len(comps) >= 3 and comps[2] != '' else None
                    face.append((vi, vti, vni))
                raw_faces.append(face)

    if len(uvs) == 0:
        raise RuntimeError("OBJ file contains no vt (UV) entries. This script requires UVs.")

    def fix_idx(idx, length):
        if idx is None:
            return None
        if idx > 0:
            return idx - 1
        else:
            return length + idx

    verts = np.asarray(verts, dtype=np.float64)
    uvs = np.asarray(uvs, dtype=np.float64)

    mapping = {}
    new_verts = []
    new_uvs = []
    new_faces = []

    for face in raw_faces:
        if len(face) < 3:
            continue
        face_fixed = []
        for (vi, vti, vni) in face:
            vi0 = fix_idx(vi, len(verts)) if vi is not None else None
            vti0 = fix_idx(vti, len(uvs)) if vti is not None else None
            face_fixed.append((vi0, vti0))
        v0 = face_fixed[0]
        for i in range(1, len(face_fixed)-1):
            tri = [v0, face_fixed[i], face_fixed[i+1]]
            tri_idx = []
            for (vi0, vti0) in tri:
                key = (vi0, vti0)
                if key not in mapping:
                    pos = verts[vi0].tolist()
                    uvcoord = uvs[vti0].tolist() if vti0 is not None else [0.0, 0.0]
                    mapping[key] = len(new_verts)
                    new_verts.append(pos)
                    new_uvs.append(uvcoord)
                tri_idx.append(mapping[key])
            new_faces.append(tri_idx)

    verts3d = np.asarray(new_verts, dtype=np.float64)
    uvs_arr = np.asarray(new_uvs, dtype=np.float64)
    faces_idx = np.asarray(new_faces, dtype=np.int64)

    return verts3d, uvs_arr, faces_idx


# Geometric helpers (numpy)
def tri_edges_from_faces(verts, faces):
    v0 = verts[faces[:,0]]
    v1 = verts[faces[:,1]]
    v2 = verts[faces[:,2]]
    return v0, v1, v2

def triangle_angles_3d(verts3d, faces):
    v0, v1, v2 = tri_edges_from_faces(verts3d, faces)
    a0 = v1 - v0
    b0 = v2 - v0
    a1 = v2 - v1
    b1 = v0 - v1
    a2 = v0 - v2
    b2 = v1 - v2

    def angle_between_3d(x, y):
        cross = np.cross(x, y)
        cross_norm = np.linalg.norm(cross, axis=1)
        dot = (x * y).sum(axis=1)
        angle = np.arctan2(cross_norm, np.clip(dot, -1e308, 1e308))
        return angle

    ang0 = angle_between_3d(a0, b0)
    ang1 = angle_between_3d(a1, b1)
    ang2 = angle_between_3d(a2, b2)
    return np.stack([ang0, ang1, ang2], axis=1)

def triangle_angles_2d(uvs, faces):
    u0 = uvs[faces[:,0]]
    u1 = uvs[faces[:,1]]
    u2 = uvs[faces[:,2]]
    a0 = u1 - u0
    b0 = u2 - u0
    a1 = u2 - u1
    b1 = u0 - u1
    a2 = u0 - u2
    b2 = u1 - u2

    def angle_between_2d(x, y):
        cross_z = x[:,0]*y[:,1] - x[:,1]*y[:,0]
        cross_abs = np.abs(cross_z)
        dot = (x * y).sum(axis=1)
        angle = np.arctan2(cross_abs, np.clip(dot, -1e308, 1e308))
        return angle

    ang0 = angle_between_2d(a0, b0)
    ang1 = angle_between_2d(a1, b1)
    ang2 = angle_between_2d(a2, b2)
    return np.stack([ang0, ang1, ang2], axis=1)

def triangle_3d_area(verts3d, faces):
    v0, v1, v2 = tri_edges_from_faces(verts3d, faces)
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    return area

def triangle_uv_signed_area(uvs, faces):
    u0 = uvs[faces[:,0]]
    u1 = uvs[faces[:,1]]
    u2 = uvs[faces[:,2]]
    a = u1 - u0
    b = u2 - u0
    cross_z = a[:,0]*b[:,1] - a[:,1]*b[:,0]
    return 0.5 * cross_z


# Metrics
def angle_distortion_stats(verts3d, uvs, faces, ignore_degenerate=True):
    angles3d = triangle_angles_3d(verts3d, faces)  # radians
    angles2d = triangle_angles_2d(uvs, faces)
    d = np.abs(angles3d - angles2d)  # (F,3)
    D = d.mean(axis=1)               # per-triangle (radians)

    areas = triangle_3d_area(verts3d, faces)
    degenerate = areas <= (EPS * 1e2)

    if ignore_degenerate:
        valid = ~degenerate
    else:
        valid = np.ones_like(D, dtype=bool)

    D_valid = D[valid]
    area_valid = areas[valid]

    stats = {}
    if D_valid.size > 0:
        stats['mean_rad'] = float(np.mean(D_valid))
        stats['median_rad'] = float(np.median(D_valid))
        stats['p90_rad'] = float(np.percentile(D_valid, 90.0))
        stats['max_rad'] = float(np.max(D_valid))
        stats['area_weighted_mean_rad'] = float((D_valid * area_valid).sum() / (area_valid.sum() + EPS))
    else:
        stats = {k: float('nan') for k in ['mean_rad','median_rad','p90_rad','max_rad','area_weighted_mean_rad']}

    return D, stats, degenerate

def normalized_area_distortion_stats(verts3d, uvs, faces, ignore_degenerate=True):
    area3d = triangle_3d_area(verts3d, faces)
    signed_uv_area = triangle_uv_signed_area(uvs, faces)
    area_uv_abs = np.abs(signed_uv_area)
    uv_flipped_mask = signed_uv_area <= 0.0
    degenerate = area3d <= (EPS * 1e2)

    if ignore_degenerate:
        mask_valid = ~degenerate
    else:
        mask_valid = np.ones_like(area3d, dtype=bool)

    total_uv = np.sum(area_uv_abs[mask_valid]) if np.any(mask_valid) else EPS
    total_3d = np.sum(area3d[mask_valid]) if np.any(mask_valid) else EPS

    total_uv = max(total_uv, EPS)
    total_3d = max(total_3d, EPS)

    frac_uv = area_uv_abs / total_uv
    frac_3d = area3d / total_3d

    frac_uv = np.clip(frac_uv, EPS, None)
    frac_3d = np.clip(frac_3d, EPS, None)

    r_norm = frac_uv / frac_3d
    D_norm = np.abs(np.log(r_norm))  # |ln r_norm|

    valid_idx = mask_valid
    D_valid = D_norm[valid_idx]
    area_valid = area3d[valid_idx]

    stats = {}
    if D_valid.size > 0:
        stats['mean'] = float(np.mean(D_valid))
        stats['median'] = float(np.median(D_valid))
        stats['p90'] = float(np.percentile(D_valid, 90.0))
        stats['max'] = float(np.max(D_valid))
        stats['area_weighted_mean'] = float((D_valid * area_valid).sum() / (area_valid.sum() + EPS))
    else:
        stats = {k: float('nan') for k in ['mean','median','p90','max','area_weighted_mean']}

    stats['flip_rate_percent'] = float(np.mean(uv_flipped_mask.astype(np.float64))) * 100.0

    return r_norm, D_norm, stats, {'degenerate_mask': degenerate, 'uv_flipped_mask': uv_flipped_mask}


# Main CLI
def main():
    parser = argparse.ArgumentParser(description="Compute angle (radians) and (normalized) area distortion for an OBJ with UVs.")
    parser.add_argument('--obj', required=True, help="Path to OBJ file")    
    parser.add_argument('--quiet', action='store_true', help="Less verbose")
    args = parser.parse_args()

    verts3d, uvs, faces = load_obj_unified_vertices(args.obj)
    if not args.quiet:
        print(f"Loaded OBJ: verts={verts3d.shape}, uvs={uvs.shape}, faces={faces.shape}")

    # Angle distortion (radians)
    D_angles, angle_stats, deg_mask = angle_distortion_stats(verts3d, uvs, faces, ignore_degenerate=True)
    print("\n=== Angle (conformal) distortion (RADIANS) ===")
    print(f"Per-triangle: {faces.shape[0]} triangles (degenerate masked: {int(np.sum(deg_mask))})")
    print(f"Mean angle error (rad): {D_angles.mean():.6f}")    


    r_norm, D_norm, area_stats, masks = normalized_area_distortion_stats(verts3d, uvs, faces, ignore_degenerate=True)
    print("\n=== Area distortion (NORMALIZED by totals) ===")
    print(f"Area-weighted mean |ln r| = {D_norm.mean():.6f}")

if __name__ == '__main__':
    main()
