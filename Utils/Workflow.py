from Utils.Funcs import *
from Utils.Network import FlexParaGlobal, FlexParaMultiChart, FlexParaGlobal_VisibilityAware
from typing import Tuple, Dict, Any, List, Optional
from collections import defaultdict

EPS = 1e-9


# Geometry utilities

def _ensure_vert_shape(verts):
    """
    Accepts verts as (V,3) or (3,V). Returns (V,3).
    """
    verts = verts.contiguous()
    if verts.dim() == 2 and verts.shape[0] == 3 and verts.shape[1] != 3:
        verts = verts.t()
    assert verts.dim() == 2 and verts.shape[1] == 3, "verts must be (V,3)"
    return verts

def _ensure_uv_shape(uv):
    """
    Accepts uv as (V,2) or (2,V). Returns (V,2).
    """    
    uv = uv.contiguous()
    if uv.dim() == 2 and uv.shape[0] == 2 and uv.shape[1] != 2:
        uv = uv.t()
    assert uv.dim() == 2 and uv.shape[1] == 2, "uv must be (V,2)"
    return uv

def triangle_angles_3d(verts3d, faces):
    """
    verts3d: (V,3)
    faces: (F,3) long
    returns angles: (F,3) in radians, angle_i is angle at face vertex i (corresponding to faces[:,i])
    """
    verts3d = _ensure_vert_shape(verts3d)
    v0 = verts3d[faces[:, 0]]
    v1 = verts3d[faces[:, 1]]
    v2 = verts3d[faces[:, 2]]
    # vectors pointing away from vertex
    a0 = v1 - v0   # for angle at v0: between a0 and a2rev
    b0 = v2 - v0
    a1 = v2 - v1   # angle at v1 between a1 and (v0-v1)
    b1 = v0 - v1
    a2 = v0 - v2   # angle at v2 between a2 and (v1-v2)
    b2 = v1 - v2

    def angle_between_3d(x, y):
        # angle = atan2(||x x y||, x·y)
        cross = torch.cross(x, y, dim=1)          # (F,3)
        cross_norm = torch.linalg.norm(cross, dim=1)
        dot = (x * y).sum(dim=1).clamp(min=-1e30, max=1e30)
        angle = torch.atan2(cross_norm, dot.clamp(min=EPS))
        return angle

    ang0 = angle_between_3d(a0, b0)
    ang1 = angle_between_3d(a1, b1)
    ang2 = angle_between_3d(a2, b2)
    angles = torch.stack([ang0, ang1, ang2], dim=1)  # (F,3)
    return angles

def triangle_angles_2d(uvs, faces):
    """
    uvs: (V,2)
    faces: (F,3)
    returns angles: (F,3) in radians
    """
    uvs = _ensure_uv_shape(uvs)
    u0 = uvs[faces[:, 0]]
    u1 = uvs[faces[:, 1]]
    u2 = uvs[faces[:, 2]]

    a0 = u1 - u0
    b0 = u2 - u0
    a1 = u2 - u1
    b1 = u0 - u1
    a2 = u0 - u2
    b2 = u1 - u2

    def angle_between_2d(x, y):
        # For 2D vectors, cross product magnitude is scalar x1*y2 - x2*y1
        # angle = atan2(abs(cross_z), dot)
        cross_z = x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]
        cross_abs = cross_z.abs()
        dot = (x * y).sum(dim=1)
        angle = torch.atan2(cross_abs, dot.clamp(min=EPS))
        return angle

    ang0 = angle_between_2d(a0, b0)
    ang1 = angle_between_2d(a1, b1)
    ang2 = angle_between_2d(a2, b2)
    angles = torch.stack([ang0, ang1, ang2], dim=1)
    return angles

def triangle_3d_area(verts3d, faces):
    verts3d = _ensure_vert_shape(verts3d)
    v0 = verts3d[faces[:, 0]]
    v1 = verts3d[faces[:, 1]]
    v2 = verts3d[faces[:, 2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=1)
    area = 0.5 * torch.linalg.norm(cross, dim=1)
    return area  # (F,)

def triangle_uv_signed_area(uvs, faces):
    """
    Returns signed area for UV triangles. Positive means consistent (non-flipped),
    negative means flipped (winding reversed).
    """
    uvs = _ensure_uv_shape(uvs)
    u0 = uvs[faces[:, 0]]
    u1 = uvs[faces[:, 1]]
    u2 = uvs[faces[:, 2]]
    # signed area = 0.5 * ((u1-u0) x (u2-u0))_z where cross_z = x1*y2 - x2*y1
    a = u1 - u0
    b = u2 - u0
    cross_z = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    signed_area = 0.5 * cross_z
    return signed_area  # shape (F,)

def angle_distortion_per_triangle(verts3d, uvs, faces, ignore_degenerate=True, deg=False):
    verts3d = _ensure_vert_shape(verts3d)
    uvs = _ensure_uv_shape(uvs)
    faces = faces.long()
    F = faces.shape[0]

    angles3d = triangle_angles_3d(verts3d, faces)  # (F,3)
    angles2d = triangle_angles_2d(uvs, faces)      # (F,3)

    # absolute angular differences (elementwise); handle wrap-around not needed because angles in [0, pi]
    d = torch.abs(angles3d - angles2d)             # (F,3)
    D = d.mean(dim=1)                              # (F,) per-triangle

    # mask degenerate triangles (very small 3D area) if requested
    areas = triangle_3d_area(verts3d, faces)       # (F,)
    valid = torch.ones_like(D, dtype=torch.bool)
    if ignore_degenerate:
        valid = areas > (EPS * 100)               # tune threshold if necessary
        D = D.clone()
        D[~valid] = float('nan')

    # stats
    valid_mask = ~torch.isnan(D)
    D_valid = D[valid_mask]
    stats = {}
    if D_valid.numel() > 0:
        stats['mean_rad'] = float(D_valid.mean().item())
        stats['median_rad'] = float(D_valid.median().item())
        stats['p90_rad'] = float(D_valid.kthvalue(int(0.9 * D_valid.numel())).values.item())
        stats['max_rad'] = float(D_valid.max().item())
        # area-weighted mean
        area_valid = areas[valid_mask]
        stats['area_weighted_mean_rad'] = float((D_valid * area_valid).sum().item() / (area_valid.sum().item() + EPS))
    else:
        stats['mean_rad'] = stats['median_rad'] = stats['p90_rad'] = stats['max_rad'] = float('nan')
        stats['area_weighted_mean_rad'] = float('nan')

    if deg:
        D_deg = torch.rad2deg(D)
        stats = {k.replace('_rad','_deg'): v if not isinstance(v, float) else (float(torch.rad2deg(torch.tensor(v)).item()) if 'rad' in k else v)
                 for k, v in stats.items()}
        return D, D_deg, stats
    return D, stats

def normalized_area_distortion(verts3d, uvs, faces, ignore_degenerate=True):
    # ensure shapes: verts3d (V,3), uvs (V,2), faces (F,3)
    area3d = triangle_3d_area(verts3d, faces)             # (F,)
    signed_uv_area = triangle_uv_signed_area(uvs, faces)  # (F,)
    area_uv_abs = signed_uv_area.abs()

    # Optionally mask tiny 3D triangles
    degenerate = area3d <= (EPS * 100)
    if ignore_degenerate:
        mask_valid = ~degenerate
    else:
        mask_valid = torch.ones_like(area3d, dtype=torch.bool)

    # normalize to fractions
    total_uv = area_uv_abs[mask_valid].sum().clamp(min=EPS)
    total_3d = area3d[mask_valid].sum().clamp(min=EPS)
    frac_uv = area_uv_abs / total_uv
    frac_3d = area3d / total_3d

    # Avoid division by zero for degenerate entries
    frac_uv = frac_uv.clamp(min=EPS)
    frac_3d = frac_3d.clamp(min=EPS)

    r_norm = frac_uv / frac_3d
    D_norm = torch.abs(torch.log(r_norm))

    # compute stats on valid triangles
    valid_idx = torch.nonzero(mask_valid).squeeze(1)
    D_valid = D_norm[mask_valid]
    stats = {}
    if D_valid.numel() > 0:
        stats['mean'] = float(D_valid.mean().item())
        stats['median'] = float(D_valid.median().item())
        stats['p90'] = float(D_valid.kthvalue(int(0.9 * D_valid.numel())).values.item())
        stats['max'] = float(D_valid.max().item())
        # area-weighted using 3D area
        area_valid = area3d[mask_valid]
        stats['area_weighted_mean'] = float((D_valid * area_valid).sum().item() / (area_valid.sum().item() + EPS))
    else:
        stats = {k: float('nan') for k in ['mean','median','p90','max','area_weighted_mean']}

    # flip rate (over all faces)
    uv_flipped = signed_uv_area <= 0.0
    stats['flip_rate_percent'] = float(uv_flipped.float().mean().item()) * 100.0

    # top offenders (indices)
    topk = min(10, D_valid.numel())
    if topk > 0:
        # we return global face indices of top distortions
        valid_vals = D_norm[mask_valid]
        _, rel_idx = torch.topk(valid_vals, topk, largest=True)
        top_faces = valid_idx[rel_idx].cpu().numpy().tolist()
    else:
        top_faces = []

    return r_norm, D_norm, stats, {
        'degenerate_mask': degenerate,
        'uv_flipped_mask': uv_flipped,
        'top_faces': top_faces
    }


# Compute AO statistics for seam vertices and optionally save a histogram.
def seam_ao_statistics(
    seam_mask,
    ao_v: Optional[np.ndarray] = None,
    mesh_path_for_ao: Optional[str] = None,
    seam_threshold: float = 0.5,
    compute_weighted_mean: bool = True,
    save_histogram_path: Optional[str] = None,
    yaxis: str = 'count'   # 'count' (number of vertices) or 'density'
) -> Dict[str, Any]:

    # convert seam_mask to numpy
    try:
        import torch
        if isinstance(seam_mask, (torch.Tensor,)):
            seam_mask_np = seam_mask.detach().cpu().numpy().ravel()
        else:
            seam_mask_np = np.asarray(seam_mask).ravel()
    except Exception:
        seam_mask_np = np.asarray(seam_mask).ravel()

    V = seam_mask_np.shape[0]

    # obtain AO values
    if ao_v is None:
        if mesh_path_for_ao is None:
            raise ValueError("Either ao_v or mesh_path_for_ao must be provided.")
        ao_v = compute_vertex_ambient_occlusion(mesh_path_for_ao, visualize_ao=False)
    ao_v = np.asarray(ao_v).ravel()
    if ao_v.shape[0] != V:
        raise ValueError(f"AO array length ({ao_v.shape[0]}) does not match seam_mask length ({V}).")

    # binarize seams
    seam_bool = seam_mask_np > seam_threshold
    seam_indices = np.nonzero(seam_bool)[0]
    n_seam = len(seam_indices)

    # basic stats
    global_mean = float(np.mean(ao_v))
    global_median = float(np.median(ao_v))

    stats = {
        'vertex_count': int(V),
        'seam_vertex_count': int(n_seam),
        'seam_vertex_fraction': float(n_seam / max(V, 1))
    }

    if n_seam == 0:
        stats.update({
            'seam_mean': None,
            'seam_weighted_mean': None,
            'global_mean': global_mean,
            'global_median': global_median,
            'occluded_fraction_thresholds': {}
        })
        return stats

    seam_ao = ao_v[seam_indices]
    seam_mean = float(np.mean(seam_ao))
    seam_median = float(np.median(seam_ao))
    seam_std = float(np.std(seam_ao))

    # weighted mean (if requested)
    seam_scores = seam_mask_np[seam_indices].astype(float)
    seam_scores_clamped = np.clip(seam_scores, 0.0, None)
    seam_weighted_mean = None
    if compute_weighted_mean and seam_scores_clamped.sum() > 0:
        seam_weighted_mean = float((seam_ao * seam_scores_clamped).sum() / seam_scores_clamped.sum())

    thresholds = [0.25, 0.5, 0.75]
    occluded_frac = {t: float(np.mean(seam_ao <= t)) for t in thresholds}  # note: 0=occluded in your convention

    stats.update({
        'seam_mean': seam_mean,
        'seam_median': seam_median,
        'seam_std': seam_std,
        'seam_weighted_mean': seam_weighted_mean,
        'global_mean': global_mean,
        'global_median': global_median,
        'occluded_fraction_thresholds': occluded_frac
    })

    # plotting: counts vs density
    if save_histogram_path is not None:
        plt.figure(figsize=(6,4))
        bins = np.linspace(0.0, 1.0, 51)

        if yaxis == 'count':
            # absolute counts: density=False
            n_all, bins_all, patches_all = plt.hist(ao_v, bins=bins, alpha=0.6, color='black',
                                                    label=f'All vertices (N={ao_v.size})', density=False)
            n_seam_hist, bins_seam, patches_seam = plt.hist(seam_ao, bins=bins, alpha=0.75, color='red',
                                                            label=f'Seam vertices (N={seam_ao.size})', density=False)
            plt.ylabel('Number of vertices')
        else:
            # density normalized
            n_all, bins_all, patches_all = plt.hist(ao_v, bins=bins, alpha=0.6, color='black',
                                                    label='All vertices', density=True)
            n_seam_hist, bins_seam, patches_seam = plt.hist(seam_ao, bins=bins, alpha=0.75, color='red',
                                                            label='Seam vertices', density=True)
            plt.ylabel('Vertices Density')

        # annotate means in title
        plt.xlabel('Ambient Occlusion (0=occluded, 1=exposed)')
        plt.title(f'Seam Mean AO = {seam_mean:.4f}, Global Mean AO={global_mean:.4f}')
        plt.legend(loc='upper left', fontsize='medium')
        plt.tight_layout()
        plt.savefig(save_histogram_path, dpi=600)
        plt.close()
        stats['histogram_path'] = save_histogram_path
        stats['histogram_yaxis'] = yaxis

    return stats


def train_visibility_aware_uv_param(
    vertices, normals, faces, 
    ambient_occlusion, num_iter, export_folder
):    

    # Model, optimizer, scheduler
    net = FlexParaGlobal().train().cuda()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-8, eps=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=1e-5)
    L1Loss = nn.L1Loss()

    opt_diff_itv = 5

    # Mesh vertices and faces tensors
    V = vertices.shape[0]
    F = faces.shape[0]
    # Per‐vertex PC + normals: [1, V, 6]
    PC_full = np.concatenate((vertices, normals), axis=-1)
    PC_full = torch.from_numpy(PC_full).unsqueeze(0).float().cuda()
    # Faces for distortion loss:
    faces_t = torch.from_numpy(faces).unsqueeze(0).long().cuda()

    # Ambient occlusion on GPU as torch.Tensor[V]
    AO_v_global = ambient_occlusion.to(dtype=torch.float32, device='cuda')

    # UV grid G over all vertices    
    M_side = int(np.ceil(np.sqrt(V)))
    G = torch.from_numpy(build_2d_grids(M_side, M_side).reshape(-1,2)).unsqueeze(0).float().cuda()  # [1, M, 2]

    # For logging
    total_loss = []
    boundary_ao_loss = []

    for epc_idx in tqdm(range(1, num_iter+1)):
        optimizer.zero_grad()

        # 1) Run the entire mesh through FlexParaGlobal_VisibilityAware
        P = PC_full[:, :, :3]    # [1, V, 3]
        Pn = PC_full[:, :, 3:6]  # [1, V, 3]
        (P_o, Q, P_c, P_cn,
         Q_hat, P_h, P_hn, P_ho, Q_hc) = net(G, P)

        # print(Q.shape)

        # 2) Compute all your original FlexPara losses
        # Normalize UVs
        Qn = uv_bounding_box_normalization(Q)
        Qhn = uv_bounding_box_normalization(Q_hat)
        Qhcn = uv_bounding_box_normalization(Q_hc)

        L_wrap   = chamfer_distance_cuda(P_h, P)
        rep_th   = (2/(M_side-1))*0.25
        L_unwrap = (compute_repulsion_loss(Qn,8,rep_th) +
                    compute_repulsion_loss(Qhn,8,rep_th) +
                    compute_repulsion_loss(Qhcn,8,rep_th))
        L_cc_p   = L1Loss(P, P_c) + L1Loss(Q_hat, Q_hc)
        L_cc_n   = compute_normal_cos_sim_loss(Pn, P_cn)

        # differential + triangle distortion every opt_diff_itv iters
        if epc_idx == 1 or epc_idx % opt_diff_itv == 0:
            _, e1, e2 = compute_differential_properties(P_c, Q)
            L_conf_diff = L1Loss(e1, e2)
            L_conf_tri  = angle_preserving_loss(Qn, P, faces_t)
            base_loss = (L_wrap + 0.01*L_unwrap + 0.01*L_cc_p +
                         0.005*L_cc_n + 0.01*L_conf_diff + 1e-5*L_conf_tri)
        else:
            base_loss = L_wrap + 0.01*L_unwrap + 0.01*L_cc_p + 0.005*L_cc_n


        # 3) Boundary‐AO loss on the full UV
        # a) Soft seam weights
        if epc_idx >= (0.0 * num_iter):
            Q_np = Qn.squeeze(0)                      # [V,2]
            P_pos = P.squeeze(0)           # [V,3] (your original 3D points)
            seam_mask, eta, tau_used = find_uv_seam(
                faces=faces_t[0],               # LongTensor[F,3] (or numpy array)
                positions=P_pos,                # Tensor [V,3] on same device as Q_uv
                uv=Q_np,                        # Tensor [V,2]
                J_cut=5,                        # as paper suggests
                tau=None,                       # will use 0.02*L(Q)
                visualize=False,
                save_path=None
            )

            # b) Boundary‐AO loss (differentiable)        
            L_bound = boundary_occlusion_loss(AO_v_global, seam_mask)
            boundary_ao_loss.append(L_bound.item())

            # 3) Combine into total loss
            loss = base_loss + (0.004 * L_bound)            
    
        else:
            loss = base_loss

        # print("\n Q_np: ", Q_np.requires_grad)
        # print("\n Q_np: ", Q_np.grad_fn)
        # print("\n seam_mask: ", seam_mask.requires_grad)
        # print("\n seam_mask: ", seam_mask.grad_fn)
        # print("\n L_bound: ", L_bound.requires_grad)
        # print("\n L_bound: ", L_bound.grad_fn)
        # print("\n loss: ", loss.requires_grad)
        # print("\n loss: ", loss.grad_fn)

        # 4) Total and backprop
        loss.backward()        
        optimizer.step()
        scheduler.step()

        total_loss.append(loss.item())        

        # Extract and visualize the cutting seam points for the last training iteration
        if epc_idx == num_iter:
            boundary_uv_points_dir = os.path.join(export_folder, "boundary_uv_points.png")
            P_pos = P.squeeze(0)           # [V,3] (your original 3D points)
            seam_mask, eta, tau_used = find_uv_seam(
                faces=faces_t[0],               # LongTensor[F,3] (or numpy array)
                positions=P_pos,                # Tensor [V,3] on same device as Q_uv
                uv=Q_np,                        # Tensor [V,2]
                J_cut=5,                        # as paper suggests
                tau=None,                       # will use 0.02*L(Q)
                visualize=True,
                save_path=boundary_uv_points_dir
            )

            boundary_3D_points_dir = os.path.join(export_folder, "mesh_with_seam.glb")
            save_glb_open3d(vertices, faces, seam_mask, out_path=boundary_3D_points_dir, threshold=0.5)



    # Save model + loss curves
    net.zero_grad()
    torch.cuda.empty_cache()
    torch.save(net.state_dict(), os.path.join(export_folder, "trained_model.pth"))
    save_1d_plot(total_loss, export_folder, "Total_Loss")
    save_1d_plot(boundary_ao_loss, export_folder, "Visibility_Loss")

    save_list_lines(total_loss, out_dir=export_folder, filename="total_loss.txt", mode="a")
    save_list_lines(boundary_ao_loss, out_dir=export_folder, filename="visibility_loss.txt", mode="a")


def train_semantic_aware_uv_param(
    vertices, normals, faces, 
    num_iter, export_folder
):    

    # Model, optimizer, scheduler
    net = FlexParaGlobal().train().cuda()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-8, eps=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=1e-5)
    L1Loss = nn.L1Loss()

    opt_diff_itv = 5

    # Mesh vertices and faces tensors
    V = vertices.shape[0]
    F = faces.shape[0]
    # Per‐vertex PC + normals: [1, V, 6]
    PC_full = np.concatenate((vertices, normals), axis=-1)
    PC_full = torch.from_numpy(PC_full).unsqueeze(0).float().cuda()
    # Faces for distortion loss:
    faces_t = torch.from_numpy(faces).unsqueeze(0).long().cuda()


    # UV grid G over all vertices    
    M_side = int(np.ceil(np.sqrt(V)))
    G = torch.from_numpy(build_2d_grids(M_side, M_side).reshape(-1,2)).unsqueeze(0).float().cuda()  # [1, M, 2]

    # For logging
    total_loss = []    

    for epc_idx in tqdm(range(1, num_iter+1)):
        optimizer.zero_grad()        

        # 1) Run the entire mesh through FlexParaGlobal_VisibilityAware
        P = PC_full[:, :, :3]    # [1, V, 3]
        Pn = PC_full[:, :, 3:6]  # [1, V, 3]
        (P_o, Q, P_c, P_cn,
         Q_hat, P_h, P_hn, P_ho, Q_hc) = net(G, P)
        
        # print(Q.shape)

        # 2) Compute all your original FlexPara losses
        # Normalize UVs
        Qn = uv_bounding_box_normalization(Q)
        Qhn = uv_bounding_box_normalization(Q_hat)
        Qhcn = uv_bounding_box_normalization(Q_hc)

        L_wrap   = chamfer_distance_cuda(P_h, P)
        rep_th   = (2/(M_side-1))*0.25
        L_unwrap = (compute_repulsion_loss(Qn,8,rep_th) +
                    compute_repulsion_loss(Qhn,8,rep_th) +
                    compute_repulsion_loss(Qhcn,8,rep_th))
        L_cc_p   = L1Loss(P, P_c) + L1Loss(Q_hat, Q_hc)
        L_cc_n   = compute_normal_cos_sim_loss(Pn, P_cn)

        # differential + triangle distortion every opt_diff_itv iters
        if epc_idx == 1 or epc_idx % opt_diff_itv == 0:
            _, e1, e2 = compute_differential_properties(P_c, Q)
            L_conf_diff = L1Loss(e1, e2)
            L_conf_tri  = angle_preserving_loss(Qn, P, faces_t)
            base_loss = (L_wrap + 0.01*L_unwrap + 0.01*L_cc_p +
                         0.005*L_cc_n + 0.01*L_conf_diff + 1e-5*L_conf_tri)
        else:
            base_loss = L_wrap + 0.01*L_unwrap + 0.01*L_cc_p + 0.005*L_cc_n


        # 4) Total and backprop
        loss = base_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss.append(loss.item())

 
    # Save model + loss curves
    net.zero_grad()
    torch.cuda.empty_cache()
    torch.save(net.state_dict(), os.path.join(export_folder, "trained_model.pth"))
    save_1d_plot(total_loss, export_folder, "Total_Loss")


def test_visibility_aware_uv_param(vertices, normals, faces, ambient_occlusion, load_ckpt_path, export_folder):
    net =  FlexParaGlobal().train().cuda()
    weight = torch.load(os.path.join(export_folder, load_ckpt_path))
    net.load_state_dict(weight)
    pre_samplings = np.concatenate((vertices,normals), axis=-1)
    pre_samplings = torch.tensor(pre_samplings).unsqueeze(0).float().cuda()
    P = pre_samplings[:, :, 0:3]
    P_N = pre_samplings[:, :, 3:6]

    N = pre_samplings.shape[1] 
    grid_height, grid_width = int(np.sqrt(N)), int(np.sqrt(N))
    G = torch.tensor(build_2d_grids(grid_height, grid_width).reshape(-1, 2)).unsqueeze(0).float().cuda() 

    with torch.no_grad():
        P_opened, Q, P_cycle, P_cycle_n, Q_hat, P_hat, P_hat_n, P_hat_opened, Q_hat_cycle = net(G, P)
    Q_eval_normalization = uv_bounding_box_normalization(Q)
    
    plt.figure(figsize=(16, 16))
    plt.axis('off')
    plt.scatter(ts2np(Q_eval_normalization.squeeze(0))[:, 0], ts2np(Q_eval_normalization.squeeze(0))[:, 1], s=5, c=((ts2np(P_N.squeeze(0)) + 1) / 2))
    plt.savefig(os.path.join(export_folder, "Q_eval_normalized.png"), dpi=400, bbox_inches="tight")
    


    Q_np = Q_eval_normalization.squeeze(0)                      # [V,2]
    P_pos = P.squeeze(0)           # [V,3] (your original 3D points)
    faces_t = torch.from_numpy(faces).unsqueeze(0).long().cuda()
    boundary_uv_points_dir = os.path.join(export_folder, "boundary_uv_points.png")
    seam_mask, eta, tau_used = find_uv_seam(
        faces=faces_t[0],               # LongTensor[F,3] (or numpy array)
        positions=P_pos,                # Tensor [V,3] on same device as Q_uv
        uv=Q_np,                        # Tensor [V,2]
        J_cut=5,                        # as paper suggests
        tau=None,                       # will use 0.02*L(Q)
        visualize=False,
        save_path=boundary_uv_points_dir
    )
    boundary_3D_points_dir = os.path.join(export_folder, "mesh_with_seam.glb")
    save_glb_open3d(vertices, faces, seam_mask, out_path=boundary_3D_points_dir, threshold=0.5)


    ################################ Computing AO values for 3D seam vertices ##########################    
    stats = seam_ao_statistics(
    seam_mask=seam_mask,          # tensor or numpy
    ao_v=ambient_occlusion,                    # or set mesh_path_for_ao instead
    seam_threshold=0.5,           # same as your GLB coloring threshold
    compute_weighted_mean=True,
    save_histogram_path=os.path.join(export_folder, 'seam_ao_hist.png'),
    yaxis='density'   # <- this makes y-axis show counts
    )
    print("Seam AO mean:", stats['seam_mean'])


    ################################# keep 3D seam vertices but remove their UVs ################################
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    seam_mask_np = to_numpy(seam_mask).ravel()
    seam_bool = seam_mask_np > 0.005   # same threshold you used for GLB coloring
    num_seam = int(np.sum(seam_bool))
    V = int(np.asarray(vertices).shape[0])
    print(f"Detected {num_seam}/{V} seam vertices (threshold 0.1).")

    verts_np = to_numpy(vertices)    # (V,3)
    uvs_np = to_numpy(Q_eval_normalization.squeeze(0))   # (V,2)
    norms_np = to_numpy(normals) if normals is not None else None
    faces_np = np.asarray(faces, dtype=np.int64)

    # fix faces if 1-based
    if faces_np.min() == 1 and faces_np.max() == verts_np.shape[0]:
        faces_np = faces_np - 1


 
    print("################################ Statistics before excluding seam-touching triangles ################################ ")
    ############################ Angle distortion ##################################
    D_conform_after, stats = angle_distortion_per_triangle(torch.tensor(verts_np).cuda(), torch.tensor(uvs_np).cuda(), torch.tensor(faces_np).cuda(), ignore_degenerate=True, deg=False)
    print("Angle distortion:", D_conform_after.mean().detach().cpu().numpy())
    ############################ Area distortion ##################################
    _, D_area_before, _, _ = normalized_area_distortion(torch.tensor(verts_np).cuda(), torch.tensor(uvs_np).cuda(), torch.tensor(faces_np).cuda())
    print("Area distortion:", D_area_before.mean().detach().cpu().numpy())    


    print("################################ Statistics after excluding seam-touching triangles ################################ ")
    # compute distortion AFTER excluding seam-touching triangles
    # build face mask: True for triangles KEEPed (none of their vertices are seam vertices)
    face_has_seam = np.any(seam_bool[faces_np], axis=1)
    keep_face_mask = ~face_has_seam
    if np.sum(keep_face_mask) == 0:
        D_area_after = None
        stats_after = None
        print("No non-seam triangles remain (after excluding seam-touching triangles).")
    else:
        faces_keep = faces_np[keep_face_mask]

        ############################ Angle distortion ##################################
        D_conform_after, stats = angle_distortion_per_triangle(torch.tensor(verts_np).cuda(), torch.tensor(uvs_np).cuda(), torch.tensor(faces_keep).cuda(), ignore_degenerate=True, deg=False)
        print("Angle distortion:", D_conform_after.mean().detach().cpu().numpy())
        ############################ Area distortion ##################################
        r, D_area_after, stats_after, masks = normalized_area_distortion(torch.tensor(verts_np).cuda(), torch.tensor(uvs_np).cuda(), torch.tensor(faces_keep).cuda())
        print("Area distortion:", D_area_after.mean().detach().cpu().numpy())
    print("#############################################################################################################################")


    # write two OBJs:
    # (1) full UV OBJ (unchanged geometry & UVs)
    obj_full = os.path.join(export_folder, "mesh_with_uv.obj")
    with open(obj_full, 'w', encoding='utf-8') as f:
        f.write('# OBJ exported with full UVs (all vertices have vt)\n')
        for i in range(verts_np.shape[0]):
            x,y,z = verts_np[i]
            f.write(f'v {float(x):.6f} {float(y):.6f} {float(z):.6f}\n')
        for i in range(uvs_np.shape[0]):
            u,v = uvs_np[i]
            f.write(f'vt {float(u):.6f} {float(v):.6f}\n')
        write_vn = False
        if norms_np is not None and norms_np.shape[0] == verts_np.shape[0] and norms_np.shape[1] == 3:
            write_vn = True
            for i in range(norms_np.shape[0]):
                nx,ny,nz = norms_np[i]
                f.write(f'vn {float(nx):.6f} {float(ny):.6f} {float(nz):.6f}\n')
        for face in faces_np:
            a,b,c = int(face[0])+1, int(face[1])+1, int(face[2])+1
            if write_vn:
                f.write(f'f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n')
            else:
                f.write(f'f {a}/{a} {b}/{b} {c}/{c}\n')

    # (2) OBJ with seam vertices having NO vt entries; faces touching seam vertices are written without vt
    obj_noseam_uvs = os.path.join(export_folder, "mesh_with_uv_noseam_uvs.obj")
    with open(obj_noseam_uvs, 'w', encoding='utf-8') as f:
        f.write('# OBJ exported with seam vertices having NO vt entries; faces touching seams have no vt.\n')
        # write vertices
        for i in range(verts_np.shape[0]):
            x,y,z = verts_np[i]
            f.write(f'v {float(x):.6f} {float(y):.6f} {float(z):.6f}\n')

        # map vertex -> vt index (1-based). seam vertices -> vt_index = 0 (no vt)
        vt_index = np.zeros(verts_np.shape[0], dtype=np.int32)
        vt_counter = 0
        for i in range(verts_np.shape[0]):
            if seam_bool[i]:
                vt_index[i] = 0
            else:
                vt_counter += 1
                vt_index[i] = vt_counter
                u,v = uvs_np[i]
                f.write(f'vt {float(u):.6f} {float(v):.6f}\n')

        # write normals (all)
        write_vn = False
        if norms_np is not None and norms_np.shape[0] == verts_np.shape[0] and norms_np.shape[1] == 3:
            write_vn = True
            for i in range(norms_np.shape[0]):
                nx,ny,nz = norms_np[i]
                f.write(f'vn {float(nx):.6f} {float(ny):.6f} {float(nz):.6f}\n')

        # write faces:
        for face in faces_np:
            a,b,c = int(face[0]), int(face[1]), int(face[2])
            av, bv, cv = a+1, b+1, c+1
            if vt_index[a] != 0 and vt_index[b] != 0 and vt_index[c] != 0:
                # all vertices have vt -> include vt indices
                if write_vn:
                    f.write(f'f {av}/{vt_index[a]}/{av} {bv}/{vt_index[b]}/{bv} {cv}/{vt_index[c]}/{cv}\n')
                else:
                    f.write(f'f {av}/{vt_index[a]} {bv}/{vt_index[b]} {cv}/{vt_index[c]}\n')
            else:
                # write face without vt indices (faces touching a seam vertex)
                if write_vn:
                    f.write(f'f {av}//{av} {bv}//{bv} {cv}//{cv}\n')
                else:
                    f.write(f'f {av} {bv} {cv}\n')

    print(f"Wrote full UV OBJ: {obj_full}")
    print(f"Wrote OBJ with seam UVs removed (faces touching seams untextured): {obj_noseam_uvs}")


def test_semantic_aware_uv_param(vertices, normals, faces, load_ckpt_path, export_folder):
    # 1) load model & compute Q
    net = FlexParaGlobal().train().cuda()
    weight = torch.load(os.path.join(export_folder, load_ckpt_path))
    net.load_state_dict(weight)

    pre_samplings = np.concatenate((vertices, normals), axis=-1)
    pre_samplings = torch.tensor(pre_samplings).unsqueeze(0).float().cuda()
    P   = pre_samplings[:, :, 0:3]
    P_N = pre_samplings[:, :, 3:6]

    N = pre_samplings.shape[1]
    h, w = int(np.sqrt(N)), int(np.sqrt(N))
    G = torch.tensor(build_2d_grids(h, w).reshape(-1, 2)).unsqueeze(0).float().cuda()

    with torch.no_grad():
        _, Q, *_ = net(G, P)

    # 2) normalize & save UVs
    Q_norm = uv_bounding_box_normalization(Q)      # [1, N, 2]
    uv_np   = Q_norm.squeeze(0).cpu().numpy()       # (N,2)
    np.save(os.path.join(export_folder, "uv_coords.npy"), uv_np)

    # 3) extract the semantic color from the folder name
    part_name = os.path.basename(export_folder)     # e.g. "part_192_67_251_1"
    _, rs, gs, bs, *_ = part_name.split("_")
    color_rgb = np.array([int(rs), int(gs), int(bs)]) / 255.0

    # 4) scatter-plot using that uniform semantic color
    plt.figure(figsize=(16, 16))
    plt.axis('off')
    plt.scatter(
        uv_np[:, 0], uv_np[:, 1],
        s=400,
        c=[color_rgb],    # uniform color for all points
        marker='.'
    )

    # 5) save the part_tex.png
    png_path = os.path.join(export_folder, "part_tex.png")
    plt.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close()


    # keep 3D seam vertices but remove their UVs
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)    

    verts_np = to_numpy(vertices)    # (V,3)
    uvs_np = to_numpy(Q_norm.squeeze(0))   # (V,2)
    norms_np = to_numpy(normals) if normals is not None else None
    faces_np = np.asarray(faces, dtype=np.int64)

    # fix faces if 1-based
    if faces_np.min() == 1 and faces_np.max() == verts_np.shape[0]:
        faces_np = faces_np - 1


 
    print("################################ Statistics ################################ ")
    ############################ Angle distortion ##################################
    D_conform_after, stats = angle_distortion_per_triangle(torch.tensor(verts_np).cuda(), torch.tensor(uvs_np).cuda(), torch.tensor(faces_np).cuda(), ignore_degenerate=True, deg=False)
    print("Angle distortion:", D_conform_after.mean().detach().cpu().numpy())
    ############################ Area distortion ##################################
    _, D_area_before, _, _ = normalized_area_distortion(torch.tensor(verts_np).cuda(), torch.tensor(uvs_np).cuda(), torch.tensor(faces_np).cuda())
    print("Area distortion:", D_area_before.mean().detach().cpu().numpy())
    print("#############################################################################################################################")


    # write results in a text file:
    with open("floats.txt", "a") as f:       # "a" appends
        f.write(f"{D_conform_after.mean().detach().cpu().numpy(), D_area_before.mean().detach().cpu().numpy()},\n")


    # write OBJ:    
    obj_full = os.path.join(export_folder, "mesh_with_uv.obj")
    with open(obj_full, 'w', encoding='utf-8') as f:
        f.write('# OBJ exported with full UVs (all vertices have vt)\n')
        for i in range(verts_np.shape[0]):
            x,y,z = verts_np[i]
            f.write(f'v {float(x):.6f} {float(y):.6f} {float(z):.6f}\n')
        for i in range(uvs_np.shape[0]):
            u,v = uvs_np[i]
            f.write(f'vt {float(u):.6f} {float(v):.6f}\n')
        write_vn = False
        if norms_np is not None and norms_np.shape[0] == verts_np.shape[0] and norms_np.shape[1] == 3:
            write_vn = True
            for i in range(norms_np.shape[0]):
                nx,ny,nz = norms_np[i]
                f.write(f'vn {float(nx):.6f} {float(ny):.6f} {float(nz):.6f}\n')
        for face in faces_np:
            a,b,c = int(face[0])+1, int(face[1])+1, int(face[2])+1
            if write_vn:
                f.write(f'f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n')
            else:
                f.write(f'f {a}/{a} {b}/{b} {c}/{c}\n')
    print(f"Wrote full UV OBJ: {obj_full}")    


def build_vertex_neighbors_from_faces(faces, num_verts: Optional[int]=None, device='cpu') -> List[torch.LongTensor]:
    """
    faces: (F,3) int array (global vertex indices)
    Returns:
      neighbors: list of length V, each entry is a sorted numpy array of neighboring vertex indices (1-ring).
    """
    # same as before: returns list[LongTensor] neighbors per vertex on device
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


def find_uv_seam(faces,
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


def save_glb_open3d(vertices, faces, seam_mask,
                    out_path="mesh_with_seam.glb",
                    seam_color=(1.0, 0.0, 0.0),    # RGB floats in [0,1]
                    base_color=(0.7, 0.7, 0.7),
                    threshold=0.5):
    """
    Save a GLB with vertex colors. Handles:
      - seam_mask bool array
      - seam_mask float array in [0,1] (soft membership)
      - seam_mask torch Tensor (cpu or cuda)
    If soft==False: perform hard thresholding: seam = seam_mask > threshold
    If soft==True: blend colors by seam strength (0..1)
    """

    # 1) convert seam_mask -> numpy float array in [0,1]
    if hasattr(seam_mask, "detach"):
        seam_mask = seam_mask.detach().cpu().numpy()
    seam_mask = np.asarray(seam_mask)

    # If boolean already, convert to float in [0,1]
    if seam_mask.dtype == np.bool_:
        seam_strength = seam_mask.astype(np.float32)
    else:
        # assume numeric; clamp to [0,1]
        seam_strength = np.clip(seam_mask.astype(np.float32), 0.0, 1.0)

    V = vertices.shape[0]
    if seam_strength.shape[0] != V:
        raise ValueError(f"seam_mask length {seam_strength.shape[0]} != num vertices {V}")

    # 2) build colors
    base_color = np.asarray(base_color, dtype=np.float32)
    seam_color = np.asarray(seam_color, dtype=np.float32)

    # hard threshold: only vertices with strength > threshold are seam
    seam_bool = seam_strength > float(threshold)
    colors = np.tile(base_color[None, :], (V, 1))
    colors[seam_bool] = seam_color

    # 3) create open3d mesh and save
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    mesh.compute_vertex_normals()

    ok = o3d.io.write_triangle_mesh(out_path, mesh, write_vertex_colors=True)
    if not ok:
        raise RuntimeError(f"Open3D failed to write {out_path}. Check Open3D version / support.")
    print(f"Saved GLB via Open3D: {out_path}")

