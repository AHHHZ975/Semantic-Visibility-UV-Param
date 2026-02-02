#################################### Usage #######################################
# python seam_ao_from_obj.py --obj path/to/mesh.obj --outdir /tmp/out --hist
##################################################################################


# What this script does:
# 1) loads an .obj (vertices, texcoords, faces)
# 2) computes per-vertex AO (calls compute_vertex_ambient_occlusion(mesh_path))
# 3) extracts seam scores using find_uv_seam_by_eta_Jcut(...)
# 4) computes AO statistics for seam vertices and saves an optional histogram.

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import igl
from typing import List, Optional, Tuple
import igl.embree
import json
import trimesh


def build_vertex_neighbors_from_faces(faces, num_verts: Optional[int]=None, device='cpu') -> List[torch.LongTensor]:

    if isinstance(faces, torch.Tensor):
        faces_np = faces.cpu().numpy()
    else:
        faces_np = np.asarray(faces)
    if num_verts is None:
        num_verts = int(faces_np.max()) + 1
    neigh = [set() for _ in range(num_verts)]
    for f in faces_np:
        a,b,c = int(f[0]), int(f[1]), int(f[2])
        neigh[a].update([b,c]); neigh[b].update([a,c]); neigh[c].update([a,b])
    neighbors = []
    for s in neigh:
        if len(s) == 0:
            neighbors.append(torch.empty(0, dtype=torch.long, device=device))
        else:
            arr = np.array(sorted(list(s)), dtype=np.int64)
            neighbors.append(torch.from_numpy(arr).long().to(device))
    return neighbors


def compute_eta_with_Jcut(uv: torch.Tensor,
                          positions: torch.Tensor,
                          neighbors: List[torch.LongTensor],
                          J_cut: int = 3,
                          gamma: float = 50.0) -> torch.Tensor:
    """
    uv: Tensor[V,2] on device
    positions: Tensor[V,3] 3D positions on same device
    neighbors: list of LongTensor (per-vertex 1-ring)
    J_cut: number of nearest neighbors (in 3D) to consider per vertex
    gamma: sharpness for soft-max (higher -> closer to hard max)
    Returns:
      eta: Tensor[V] where eta[i] = softmax-approx max_{j in selected neighbors} || q_i - q_j ||_2
    """
    device = uv.device
    V = uv.shape[0]
    eta = torch.zeros(V, device=device, dtype=uv.dtype)

    for i in range(V):
        nbrs = neighbors[i]
        if nbrs.numel() == 0:
            eta[i] = 0.0
            continue

        # if more than J_cut neighbors, select closest J_cut in 3D
        if nbrs.numel() > J_cut:
            # compute 3D distances to all ring neighbors
            pi = positions[i].unsqueeze(0)        # [1,3]
            nbr_pos = positions[nbrs]             # [k,3]
            d3 = torch.norm(nbr_pos - pi, dim=1)  # [k]
            topk = torch.topk(-d3, k=J_cut).indices  # indices in 0..k-1 of smallest distances
            chosen = nbrs[topk]
        else:
            chosen = nbrs

        # compute uv distances to chosen neighbors
        diffs = uv[i].unsqueeze(0) - uv[chosen]     # [m,2]
        dists = torch.norm(diffs, dim=1)            # [m]

        # Differentiable soft-max approximation of max using log-sum-exp:
        # eta = (1/gamma) * log( sum_j exp(gamma * d_j) )
        # numerically stable: subtract max before exp
        if dists.numel() == 0:
            eta[i] = 0.0
        else:
            max_d, _ = torch.max(dists, dim=0)
            stabilized = dists - max_d
            sum_exp = torch.sum(torch.exp(gamma * stabilized))
            eta_soft = (1.0 / (gamma + 1e-12)) * (torch.log(sum_exp + 1e-12) + gamma * max_d)
            eta[i] = eta_soft

    return eta


def find_uv_seam_by_eta_Jcut(faces,
                            positions: torch.Tensor,
                            uv: torch.Tensor,
                            J_cut: int = 5,
                            tau: Optional[float] = None,                            
                            visualize: bool = True,
                            save_path: Optional[str] = None,
                            gamma: float = 50.0,
                            beta: float = 50.0,
                            tau_scale: float = 0.1
                           ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    faces: (F,3) numpy or LongTensor
    positions: Tensor[V,3] (3D positions)
    uv: Tensor[V,2] (UV coords)
    J_cut: number of 3D neighbors to consider (paper uses 3)
    tau: optional explicit threshold (if None use tau_scale * L(Q) by default)
    percentile: fallback percentile if tau is None and you prefer percentile approach (NOT used here)
    gamma: soft-max sharpness for eta (higher -> crisper)
    beta: sigmoid sharpness for soft membership
    tau_scale: multiplier for L(Q) (paper uses 0.02)
    visualize: whether to show UV plot
    Returns:
      seam_mask (Tensor[V], soft in [0,1]), eta (Tensor[V]), tau_used (float)
    """
    device = uv.device
    V = uv.shape[0]
    # 1- build neighbors (1-ring) on device
    neighbors = build_vertex_neighbors_from_faces(faces, num_verts=V, device=device)

    # 2- compute eta using up to J_cut nearest neighbors in 3D
    eta = compute_eta_with_Jcut(uv, positions, neighbors, J_cut=J_cut, gamma=gamma)  # [V]

    # compute L(Q): side length of UV bounding square
    umin, _ = torch.min(uv[:,0], dim=0)
    umax, _ = torch.max(uv[:,0], dim=0)
    vmin, _ = torch.min(uv[:,1], dim=0)
    vmax, _ = torch.max(uv[:,1], dim=0)
    Lq = torch.max(torch.stack([umax - umin, vmax - vmin])).item()

    # threshold tau = tau_scale * L(Q) as in the paper (or use provided tau)
    tau_used = tau_scale * Lq if tau is None else float(tau)

    # Differentiable soft seam membership: sigmoid around (eta - tau)
    s_v = torch.sigmoid(beta * (eta - tau_used))   # Tensor[V], values in (0,1)

    # For visualization we can threshold s_v > 0.5 (only for plotting)
    if visualize:
        uv_np = uv.detach().cpu().numpy()
        mask_np = (s_v.detach().cpu().numpy() > 0.5).astype(bool)
        plt.figure(figsize=(5,5))
        plt.scatter(uv_np[:,0], uv_np[:,1], s=4, c='lightgray')
        plt.scatter(uv_np[mask_np,0], uv_np[mask_np,1], s=16, c='red')
        plt.title(f"Seam (J_cut={J_cut}, tau={tau_used:.6f}, Lq={Lq:.6f})")
        plt.axis('off')
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            plt.close()
        else:
            plt.show()

    # Return the soft seam mask (differentiable), the eta (soft max), and tau_used
    return s_v, eta, tau_used



def compute_vertex_ambient_occlusion(mesh_load_path, visualize_ao=False):
    """
    Computes one AO value per vertex using libigl.embree.ambient_occlusion.

    Args:
        mesh_load_path: Path to the .obj file.

    Returns:
        AO_v: (V,) float array in [0,1], where 0 = fully exposed, 1 = fully occluded.
    """
    BINARY_VISUALIZATION = False

    # Load the mesh object's vertices, faces, and normals.
    v, f = igl.read_triangle_mesh(mesh_load_path)
    n   = igl.per_vertex_normals(v, f)

    # Compute AO in [0,1], then invert so white=occluded
    AO_v = 1.0 - igl.embree.ambient_occlusion(v, f, v, n, 400) # (0 = occluded, 1 = fully exposed)
    AO_v = AO_v.astype(np.float32)

    if visualize_ao:
        if BINARY_VISUALIZATION:
            # 1) Threshold into a binary mask
            threshold = 0.7
            AO_thresh = (AO_v > threshold).astype(float)
            #   AO_thresh[i] == 1 if AO_v[i] > 0.7 (deeply occluded)
            #             == 0 otherwise (well exposed)

            # 2) Turn that into RGBA colors: occluded = red, exposed = white
            colors = np.zeros((len(AO_thresh), 4), dtype=np.uint8)
            # exposed → white
            colors[AO_thresh == 0] = np.array([255,   0,   0, 255], dtype=np.uint8)
            # occluded → red
            colors[AO_thresh == 1] = np.array([255, 255, 255, 255], dtype=np.uint8)
        else:
            # 2) Choose a vivid colormap: e.g. 'plasma', 'viridis', or reversed 'inferno_r'
            #  'plasma' gives warm colors in the occluded zones and cooler ones in the lit parts
            cmap = plt.get_cmap("plasma")

            # 3) Map to RGBA 0–255
            #    cmap expects values in [0,1], returns (R,G,B,A) in [0,1]
            colors = (cmap(AO_v) * 255).astype(np.uint8)

        
        # 3) Build and show your mesh
        mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=colors)
        # Save the mesh to an OBJ file
        mesh.export('Mesh_with_AO.glb')
        mesh.show()

    # Convert back to float32
    return np.array(AO_v, dtype=np.float32)





# ---------------- OBJ parser ----------------
def parse_obj(path):
    verts = []
    texs = []
    faces_v = []
    faces_vt = []

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v' and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vt' and len(parts) >= 3:
                texs.append([float(parts[1]), float(parts[2])])
            elif parts[0] == 'f':
                tokens = parts[1:]
                if len(tokens) < 3:
                    continue
                # fan triangulate
                tri_tokens = []
                for i in range(1, len(tokens)-1):
                    tri_tokens.append((tokens[0], tokens[i], tokens[i+1]))
                for tri in tri_tokens:
                    v_idx_trip = []
                    vt_idx_trip = []
                    for tok in tri:
                        comps = tok.split('/')
                        v_i = int(comps[0])
                        vt_i = None
                        if len(comps) >= 2 and comps[1] != '':
                            try:
                                vt_i = int(comps[1])
                            except:
                                vt_i = None
                        v_idx_trip.append(v_i)
                        vt_idx_trip.append(vt_i if vt_i is not None else -1)
                    faces_v.append(v_idx_trip)
                    faces_vt.append(vt_idx_trip)

    verts = np.asarray(verts, dtype=np.float64)
    texs = np.asarray(texs, dtype=np.float64) if len(texs) > 0 else np.zeros((0,2), dtype=np.float64)
    faces_v = np.asarray(faces_v, dtype=np.int64) if len(faces_v) > 0 else np.zeros((0,3), dtype=np.int64)
    faces_vt = np.asarray(faces_vt, dtype=np.int64) if len(faces_vt) > 0 else np.zeros((0,3), dtype=np.int64)

    V = verts.shape[0]
    T = texs.shape[0]
    if faces_v.size > 0:
        faces_v = np.where(faces_v < 0, faces_v + V + 1, faces_v)
        faces_v = faces_v - 1
    if faces_vt.size > 0 and T > 0:
        faces_vt = np.where(faces_vt < 0, faces_vt + T + 1, faces_vt)
        faces_vt = np.where(faces_vt == -1, -1, faces_vt - 1)

    return verts, texs, faces_v, faces_vt


# ---------------- derive per-vertex UVs ----------------
def per_vertex_uv_from_obj(verts, texs, faces_v, faces_vt):
    V = verts.shape[0]
    if texs.shape[0] == V and texs.shape[0] > 0:
        return texs.copy(), "vt_count_eq_v"
    if faces_vt.size == 0:
        raise RuntimeError("No vt entries found in OBJ and texcoords empty.")
    uv_acc = np.zeros((V,2), dtype=np.float64)
    counts = np.zeros((V,), dtype=np.int64)
    F = faces_v.shape[0]
    for fi in range(F):
        for k in range(3):
            v_idx = int(faces_v[fi,k])
            vt_idx = int(faces_vt[fi,k])
            if vt_idx >= 0 and vt_idx < texs.shape[0]:
                uv_acc[v_idx] += texs[vt_idx]
                counts[v_idx] += 1
    zero_count = np.sum(counts == 0)
    if zero_count > 0:
        print(f"[warning] {zero_count}/{V} vertices have no referenced vt corners. Their UV set to (0,0).")
    nonzero = counts > 0
    uv = np.zeros((V,2), dtype=np.float64)
    uv[nonzero] = uv_acc[nonzero] / counts[nonzero][:,None]
    return uv, "averaged_face_corner_vt"


# ---------------- seam AO stats & histogram ----------------
def compute_seam_ao_stats_and_hist(ao_v, seam_mask, seam_threshold=0.5, out_hist_path=None, yaxis='density'):
    seam_mask_np = seam_mask.detach().cpu().numpy().ravel() if isinstance(seam_mask, torch.Tensor) else np.asarray(seam_mask).ravel()
    ao_v = np.asarray(ao_v).ravel()
    V = ao_v.shape[0]
    if seam_mask_np.shape[0] != V:
        raise ValueError("seam_mask and ao_v length mismatch")

    seam_bool = seam_mask_np > seam_threshold
    seam_idx = np.nonzero(seam_bool)[0]
    n_seam = int(seam_idx.size)

    global_mean = float(np.mean(ao_v))
    seam_mean = None
    seam_median = None
    seam_std = None
    occluded_fracs = {}
    seam_ao = None
    if n_seam > 0:
        seam_ao = ao_v[seam_idx]
        seam_mean = float(np.mean(seam_ao))
        seam_median = float(np.median(seam_ao))
        seam_std = float(np.std(seam_ao))
        thresholds = [0.25, 0.5, 0.75]
        occluded_fracs = {t: float(np.mean(seam_ao <= t)) for t in thresholds}

    print("=== AO Summary ===")
    print(f"Total vertices: {V}")
    print(f"Seam vertices (threshold {seam_threshold}): {n_seam} ({n_seam / max(V,1):.4f})")
    print(f"Global AO mean (0=occluded,1=exposed): {global_mean:.4f}")
    if n_seam > 0:
        print(f"Seam AO mean: {seam_mean:.4f}, median: {seam_median:.4f}, std: {seam_std:.4f}")
        for t,v in occluded_fracs.items():
            print(f"Frac seam vertices with AO <= {t}: {v:.3f}")
    else:
        print("No seam vertices detected (after thresholding).")

    if out_hist_path is not None:
        bins = np.linspace(0.0, 1.0, 51)
        plt.figure(figsize=(6,4))
        if n_seam > 0:
            if yaxis == 'count':
                plt.hist(ao_v, bins=bins, alpha=0.5, color='gray', label=f'All (N={V})', density=False)
                plt.hist(seam_ao, bins=bins, alpha=0.75, color='C1', label=f'Seam (N={n_seam})', density=False)
                plt.ylabel('Number of vertices')
            else:
                plt.hist(ao_v, bins=bins, alpha=0.6, color='black', label='All vertices', density=True)
                plt.hist(seam_ao, bins=bins, alpha=0.75, color='Red', label='Seam vertices', density=True)
                plt.ylabel('Vertices Density')
        else:
            if yaxis == 'count':
                plt.hist(ao_v, bins=bins, alpha=0.8, color='gray', label=f'All (N={V})', density=False)
                plt.ylabel('Number of vertices')
            else:
                plt.hist(ao_v, bins=bins, alpha=0.8, color='gray', label='All (density)', density=True)
                plt.ylabel('Density')

        plt.xlabel('Ambient Occlusion (0=occluded, 1=exposed)')
        plt.title(f'Seam Mean AO = {seam_mean:.4f}, Global Mean AO={global_mean:.4f}')
        plt.legend(loc='upper left', fontsize='medium')
        plt.tight_layout()
        plt.savefig(out_hist_path, dpi=600)
        plt.close()
        print(f"Saved AO histogram to: {out_hist_path}")

    return {
        'V': V,
        'n_seam': n_seam,
        'global_mean': global_mean,
        'seam_mean': seam_mean,
        'seam_median': seam_median,
        'seam_std': seam_std,
        'occluded_fracs': occluded_fracs
    }


# ---------------- Main script ----------------
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--obj', required=True, help='Input OBJ file (must contain geometry; vt optional)')
    parser.add_argument('--outdir', required=True, help='Output directory to save histogram and summaries')
    parser.add_argument('--seam-threshold', type=float, default=0.5, help='Threshold to binarize seam scores')
    parser.add_argument('--hist', action='store_true', help='Save histogram PNG')
    parser.add_argument('--hist-yaxis', choices=('count','density'), default='density', help='Histogram y-axis mode')
    parser.add_argument('--no-uv-scatter', action='store_true', help='Do not save UV scatter PNG')
    parser.add_argument('--device', choices=('cpu','cuda'), default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device for seam detector tensors')
    args = parser.parse_args()

    obj_path = args.obj
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print(f"Parsing OBJ: {obj_path}")
    verts, texs, faces_v, faces_vt = parse_obj(obj_path)
    print(f"Loaded V={verts.shape[0]} vertices, T={texs.shape[0]} texcoords, F={faces_v.shape[0]} faces")

    try:
        uv_per_vertex, uv_mode = per_vertex_uv_from_obj(verts, texs, faces_v, faces_vt)
        print(f"Derived per-vertex UVs using strategy: {uv_mode}")
    except Exception as e:
        print("Error deriving per-vertex UVs:", e)
        sys.exit(1)

    print("Computing AO per vertex (this may take a while depending on ray count)...")
    ao_v = compute_vertex_ambient_occlusion(obj_path, visualize_ao=False)
    print("AO computed (len=%d). sample: %s" % (len(ao_v), np.array2string(ao_v[:5], precision=3)))

    device = torch.device(args.device)
    print("Preparing tensors for seam detector on device:", device)
    positions_t = torch.tensor(verts, dtype=torch.float32, device=device)   # [V,3]
    uv_t = torch.tensor(uv_per_vertex, dtype=torch.float32, device=device)   # [V,2]
    faces_t = torch.from_numpy(faces_v).long().to(device)                    # [F,3]

    print("Running find_uv_seam_by_eta_Jcut(...)")
    try:
        seam_mask, eta, tau_used = find_uv_seam_by_eta_Jcut(
            faces=faces_t,
            positions=positions_t,
            uv=uv_t,
            J_cut=5,
            tau=None,
            visualize=False,
            save_path=None
        )
    except Exception as e:
        print("Error calling find_uv_seam_by_eta_Jcut:", e)
        raise

    hist_path = os.path.join(outdir, 'seam_ao_hist.png') if args.hist else None
    stats = compute_seam_ao_stats_and_hist(ao_v, seam_mask, seam_threshold=args.seam_threshold, out_hist_path=hist_path, yaxis=args.hist_yaxis)

    # --- UV scatter plot: grey = normal vertices, red = seam vertices (based on seam_threshold) ---
    if not args.no_uv_scatter:
        seam_mask_np = seam_mask.detach().cpu().numpy().ravel() if isinstance(seam_mask, torch.Tensor) else np.asarray(seam_mask).ravel()
        seam_bool = seam_mask_np > args.seam_threshold
        n_seam = int(np.sum(seam_bool))

        # bounding-box normalize UVs for visualization
        uvs = uv_per_vertex.copy().astype(np.float64)
        min_uv = uvs.min(axis=0)
        max_uv = uvs.max(axis=0)
        denom = (max_uv - min_uv)
        denom[denom == 0.0] = 1.0
        uvs_norm = (uvs - min_uv) / denom  # in [0,1] bbox

        uv_png_path = os.path.join(outdir, 'uv_seam_scatter.png')
        plt.figure(figsize=(6,6))
        plt.axis('off')
        # plot non-seam in grey
        non_seam_mask = ~seam_bool
        if np.any(non_seam_mask):
            plt.scatter(uvs_norm[non_seam_mask,0], uvs_norm[non_seam_mask,1], s=2, c='gray', marker='o', alpha=0.8)
        # plot seam verts in red (on top)
        if n_seam > 0:
            plt.scatter(uvs_norm[seam_bool,0], uvs_norm[seam_bool,1], s=8, c='red', marker='o', alpha=0.9)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f'UVs (red=seam, N_seam={n_seam})', fontsize=10)
        plt.tight_layout()
        plt.savefig(uv_png_path, dpi=400, bbox_inches='tight', pad_inches=0.01)
        plt.close()
        print(f"Saved UV scatter to: {uv_png_path}")
        stats['uv_scatter_path'] = uv_png_path

    # save stats to json
    summary_path = os.path.join(outdir, 'seam_ao_stats.json')
    with open(summary_path, 'w') as fh:
        json.dump(stats, fh, indent=2)
    print("Saved summary JSON to:", summary_path)

    # Also print where histogram was saved (if any)
    if hist_path is not None:
        print("Saved histogram PNG to:", hist_path)

    print("Done.")


if __name__ == '__main__':
    main()