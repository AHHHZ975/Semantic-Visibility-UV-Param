import os, sys
import igl.embree
from pathlib import Path
from typing import Iterable, Any, Union
from PIL import Image
import math
import os

sys.path.append(os.path.abspath('..'))
from Utils.Packages import *
from Utils.Basic import *


def index_points(pc, idx):
    # pc: [B, N, C]
    # 1) idx: [B, S] -> pc_selected: [B, S, C]
    # 2) idx: [B, S, K] -> pc_selected: [B, S, K, C]
    device = pc.device
    B = pc.shape[0]
    view_shape = list(idx.shape) 
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B).to(device).view(view_shape).repeat(repeat_shape)
    pc_selected = pc[batch_indices, idx, :]
    return pc_selected



# In short, combine collects and re‑indexes sub‑mesh’s vertices 
# so everything can be treated as a standalone mesh of N vertices.
def combine(idx_faces):
    # idx_faces: Tensor of shape [B, Nf, 3], where
    #   B = batch size (here always 1),
    #   Nf = number of sampled faces,
    #   each entry is a global vertex ID.

    # 1- Step 1: Flatten & find all unique vertex IDs
    flattened_tensor = torch.flatten(idx_faces)  #  Turns [1, Nf, 3] to a 1D list of length 3*Nf of global IDs.
    unique_sorted_tensor = torch.unique(flattened_tensor) # Removes duplicates and sorts them, giving us a 1D tensor of
                                                          # length N containing every global vertex index that appears in
                                                          # our sampled faces.
    result_tensor = unique_sorted_tensor.long()
    result_tensor = result_tensor.unsqueeze(0)            # Re‑adds a leading batch dimension, so "result_tensor" ends up as
                                                          # shape [1, N]. This is going to be our "idx_combine": the full list
                                                          # of global IDs that make up our sub‑mesh, in sorted order.


    # Step 2: Build a map from global -> local
    mapping = {val.item(): idx for idx, val in enumerate(unique_sorted_tensor)} # This dict says “global ID val maps to local index idx”, where idx runs from 0 to N–1.
    
    # Step 3: Remap face array to use local IDs    
    # For each triangle (g0, g1, g2) of global IDs, it looks up mapping[g0], mapping[g1], mapping[g2], 
    # producing a new triangle (l0, l1, l2) of local IDs in [0..N–1].
    # It stacks these back into a tensor of shape [1, Nf, 3] and that’s our remapped triangles.
    new_tensor_idx = torch.tensor(
        [[mapping[val.item()] for val in row]
         for row in idx_faces.squeeze(0)])  # idx_faces.squeeze(0) removes the batch dimension, giving shape [Nf, 3]
    new_tensor_idx = new_tensor_idx.unsqueeze(0)

    # "result_tensor": shape [1, N], the sorted list of global vertex IDs in the sub‑mesh.
    # "new_tensor_idx": shape [1, Nf, 3], the same triangles but with local vertex IDs in [0..N–1].
    return result_tensor, new_tensor_idx



def load_mesh_model_vfn(mesh_load_path):
    mesh_v, mesh_f = pcu.load_mesh_vf(mesh_load_path)
    mesh_vn = rescale_normals(pcu.estimate_mesh_vertex_normals(mesh_v, mesh_f), scale=1.0)
    mesh_v, mesh_vn, mesh_f = mesh_v.astype(np.float32), mesh_vn.astype(np.float32), mesh_f.astype(np.int64)
    return mesh_v, mesh_vn, mesh_f


def build_edge_index(faces: np.ndarray) -> torch.LongTensor:
    """
    Given an (F,3) int array of triangle indices, return an (E,2) LongTensor
    of unique undirected edges.
    """
    # 1) Stack all directed edges (3 edges per triangle)
    f = faces
    e01 = f[:, [0, 1]]
    e12 = f[:, [1, 2]]
    e20 = f[:, [2, 0]]
    all_edges = np.vstack((e01, e12, e20))  # (3*F, 2)

    # 2) Sort each row so (i,j) and (j,i) become the same
    sorted_edges = np.sort(all_edges, axis=1)  # still shape (3*F, 2)

    # 3) Remove duplicated rows
    unique_edges = np.unique(sorted_edges, axis=0)  # (E, 2)

    # 4) Convert to PyTorch LongTensor
    edge_index = torch.from_numpy(unique_edges.astype(np.int64))
    return edge_index


def load_mesh_and_edges(mesh_load_path):
    mesh_v, mesh_vn, mesh_f = load_mesh_model_vfn(mesh_load_path)
    edge_index = build_edge_index(mesh_f)
    return mesh_v, mesh_vn, mesh_f, edge_index



def compute_vertex_ambient_occlusion(mesh_load_path, visualize_ao=False):
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
            colors = (cmap(AO_v) * 255).astype(np.uint8)

        
        # 3) Build and show the mesh
        mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=colors)
        # Save the mesh to an OBJ file
        mesh.export('Mesh_with_AO.glb')
        mesh.show()

    # Convert back to float32
    return np.array(AO_v, dtype=np.float32)



def save_pc_as_ply(save_path, points, colors=None, normals=None):
    assert save_path[-3:] == 'ply', 'not .ply file'
    if type(points) == torch.Tensor:
        points = np.asarray(points.detach().cpu())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        if type(colors) == torch.Tensor:
            colors = np.asarray(colors.detach().cpu())
        assert colors.min()>=0 and colors.max()<=1
        pcd.colors = o3d.utility.Vector3dVector(colors) # should be within the range of [0, 1]
    if normals is not None:
        if type(normals) == torch.Tensor:
            normals = np.asarray(normals.detach().cpu())
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=True) # should be saved as .ply file


def ts2np(x):
    y = np.asarray(x.detach().cpu())
    return y


def rescale_normals(normals, scale=1.0):
    # normals: (num_pts, 3)
    rescaled_normals = normals / np.linalg.norm(normals, ord=2, axis=-1, keepdims=True) * scale
    return rescaled_normals


####################################################################################################################
def build_2d_grids(H, W):
    h_p = np.linspace(-1, +1, H, dtype=np.float32) # np.linspace(-1.0, +1.0, H, dtype=np.float32)
    w_p = np.linspace(-1, +1, W, dtype=np.float32) # np.linspace(-1.0, +1.0, W, dtype=np.float32)
    grid_points = np.array(list(itertools.product(h_p, w_p))).reshape(H, W, 2) # (H, W, 2)
    return grid_points

def build_2d_grids_max_min(H, W,max,min):
    h_p = np.linspace(min, max, H, dtype=np.float32) # np.linspace(-1.0, +1.0, H, dtype=np.float32)
    w_p = np.linspace(min, max, W, dtype=np.float32) # np.linspace(-1.0, +1.0, W, dtype=np.float32)
    grid_points = np.array(list(itertools.product(h_p, w_p))).reshape(H, W, 2) # (H, W, 2)
    return grid_points

def build_2d_grids_max_min_centroid(H, W, Q):
    center_x = (Q[:,:,0].max() + Q[:,:,0].min())/2
    center_y = (Q[:,:,1].max() + Q[:,:,1].min())/2
    scale = max((Q[:,:,0].max()-Q[:,:,0].min()),(Q[:,:,1].max()-Q[:,:,1].min()))/2
    h_p = np.linspace(center_x-scale, center_x+scale, H, dtype=np.float32) # np.linspace(-1.0, +1.0, H, dtype=np.float32)
    w_p = np.linspace(center_y-scale, center_y+scale, W, dtype=np.float32) # np.linspace(-1.0, +1.0, W, dtype=np.float32)
    grid_points = np.array(list(itertools.product(h_p, w_p))).reshape(H, W, 2) # (H, W, 2)
    return grid_points

def build_2d_grids_r(H, W,r=1):
    h_p = np.linspace(-1*r, r, H, dtype=np.float32) # np.linspace(-1.0, +1.0, H, dtype=np.float32)
    w_p = np.linspace(-1*r, r, W, dtype=np.float32) # np.linspace(-1.0, +1.0, W, dtype=np.float32)
    grid_points = np.array(list(itertools.product(h_p, w_p))).reshape(H, W, 2) # (H, W, 2)
    return grid_points

def build_circular_grid(H, W, r=1.0):
    h_p = np.linspace(-1, +1, H, dtype=np.float32)
    w_p = np.linspace(-1, +1, W, dtype=np.float32)
    grid_points = np.array(list(itertools.product(h_p, w_p))).reshape(H, W, 2)
    mask = (grid_points[..., 0]**2 + grid_points[..., 1]**2) <= r**2
    circular_grid_points = grid_points[mask]
    
    return circular_grid_points

def build_2d_grids_charts(H, W):
    h_p = np.linspace(-1, 0, H, dtype=np.float32) # np.linspace(-1.0, +1.0, H, dtype=np.float32)
    w_p = np.linspace(-1, 0, W, dtype=np.float32) # np.linspace(-1.0, +1.0, W, dtype=np.float32)
    grid_points_1 = np.array(list(itertools.product(h_p, w_p))).reshape(H, W, 2) # (H, W, 2)

    h_p = np.linspace(0, +1, H, dtype=np.float32) # np.linspace(-1.0, +1.0, H, dtype=np.float32)
    w_p = np.linspace(0, +1, W, dtype=np.float32) # np.linspace(-1.0, +1.0, W, dtype=np.float32)
    grid_points_2 = np.array(list(itertools.product(h_p, w_p))).reshape(H, W, 2) # (H, W, 2)
    return grid_points_1,grid_points_2

def uv_bounding_box_normalization(uv_points):
    # uv_points: [B, N, 2]
    centroids = ((uv_points.min(dim=1)[0] + uv_points.max(dim=1)[0]) / 2).unsqueeze(1) # [B, 1, 2]
    uv_points = uv_points - centroids
    max_d = (uv_points**2).sum(dim=-1).sqrt().max(dim=-1)[0].view(-1, 1, 1) # [B, 1, 1]
    uv_points = uv_points / max_d
    return uv_points 

def compute_uv_grads(points_3D, points_2D):
    # points_3D: [B, N, 3]
    # points_2D: [B, N, 2]
    assert points_3D.size(1)==points_2D.size(1) and points_3D.size(2)==3 and points_2D.size(2)==2
    B, N, device = points_3D.size(0), points_3D.size(1), points_3D.device
    dx = torch.autograd.grad(points_3D[:, :, 0], points_2D, torch.ones_like(points_3D[:, :, 0]).float().to(device), create_graph=True)[0] # [B, N, 2]
    dy = torch.autograd.grad(points_3D[:, :, 1], points_2D, torch.ones_like(points_3D[:, :, 1]).float().to(device), create_graph=True)[0] # [B, N, 2]
    dz = torch.autograd.grad(points_3D[:, :, 2], points_2D, torch.ones_like(points_3D[:, :, 2]).float().to(device), create_graph=True)[0] # [B, N, 2]
    dxyz = torch.cat((dx.unsqueeze(2), dy.unsqueeze(2), dz.unsqueeze(2)), dim=2) # [B, N, 3, 2]
    grad_u = dxyz[:, :, :, 0] # [B, N, 3]
    grad_v = dxyz[:, :, :, 1] # [B, N, 3]
    return grad_u, grad_v


# This is inspired by the idea that the angle-preservation and area-preservation is 
# related to the Jacobian of the parameterization function.
# (Read notes from the Mesh parameterization book to understand this function.
# there is nice mathematic formulation behind this :D)
def compute_differential_properties(points_3D, points_2D):
    # points_3D: [B, N, 3]
    # points_2D: [B, N, 2]
    grad_u, grad_v = compute_uv_grads(points_3D, points_2D) # grad_u & grad_v: [B, N, 3]
    unit_normals = F.normalize(torch.cross(grad_u, grad_v, dim=-1), dim=-1) # [B, N, 3]
    Jf = torch.cat((grad_u.unsqueeze(-1), grad_v.unsqueeze(-1)), dim=-1) # [B, N, 3, 2]
    I = Jf.permute(0, 1, 3, 2).contiguous() @ Jf # [B, N, 2, 2]
    E = I[:, :, 0, 0] # [B, N]
    G = I[:, :, 1, 1] # [B, N]
    FF = I[:, :, 0, 1] # [B, N]
    item_1 = E + G # [B, N]
    item_2 = torch.sqrt(4*(FF**2) + (E-G)**2) # [B, N]
    lambda_1 = 0.5 * (item_1 + item_2) # [B, N]
    lambda_2 = 0.5 * (item_1 - item_2) # [B, N]
    # Note that we always have: "lambda_1 >= lambda_2"
    return unit_normals, lambda_1, lambda_2


def compute_repulsion_loss(points, K, minimum_distance):
    # points: [B, N, C]
    # K: number of neighbors
    # minimum_distance: Euclidean distance threshold
    knn_idx = knn_search(points, points, K+1)[:, :, 1:] # [B, N, K]
    knn_points = index_points(points, knn_idx) # [B, N, K, C]
    knn_distances = (((points.unsqueeze(2) - knn_points)**2).sum(dim=-1) + 1e-8).sqrt() # [B, N, K]
    loss = (F.relu(-knn_distances + minimum_distance)).mean()
    return loss


# To avoid overlapped UV coordinates, the unwrapping loss for each chart is formulated as:
# (Equation 19 in the paper)
def compute_repulsion_loss_charts(points, K, minimum_distance, charts_prob):
    # points: [B, N, C]
    # K: number of neighbors
    # minimum_distance: Euclidean distance threshold
    knn_idx = knn_search(points, points, K+1)[:, :, 1:] # [B, N, K]
    knn_points = index_points(points, knn_idx) # [B, N, K, C]
    knn_distances = (((points.unsqueeze(2) - knn_points)**2).sum(dim=-1) + 1e-8).sqrt() # [B, N, K]
    loss = ((F.relu(-knn_distances + minimum_distance)).mean(dim=-1)*charts_prob).mean()
    return loss

# Equation 20 in the paper - Cycle consistency loss within the 3D-2D-3D mapping process.
# This computes a L1 loss between the predicted vertices and the input vertices.
def L1_loss_charts(x, y, chart_prob):
    loss = (((x-y)*(x-y)).squeeze(0).sum(dim=1).sqrt()*chart_prob).mean()
    return loss

def compute_normal_cos_sim_loss(x1, x2):
    # x1: [B, N, 3]
    # x2: [B, N, 3]
    loss = (1 - F.cosine_similarity(x1.view(-1, 3), x2.view(-1, 3))).mean()
    return loss

# Equation 20 in the paper - Cycle consistency loss within the 3D-2D-3D mapping process.
# This computes a cosine similarity between the normal vectors of the input 3D mesh and
# normal vectors of the predicted mesh object.
def compute_normal_cos_sim_loss_charts(x1, x2,charts_prob):
    # x1: [B, N, 3]
    # x2: [B, N, 3]
    loss = ((1 - F.cosine_similarity(x1.view(-1, 3), x2.view(-1, 3)))*charts_prob).mean()
    return loss



# For the computation of triangle distortion loss TDL, we measure the 
# angle differences within each pair of 3D and 2D triangles. Given a certain
# triangle in the original mesh X with three angles,
# since each triangle vertex in P is mapped to its UV coordinate
# in Q, we can obtain the corresponding 2D triangle within
# the UV space with three corresponding angles.
# (Equation 14 in the paper) 
def angle_preserving_loss(uv_points, points_3d, triangles):    
    vertex_indices = triangles[0]
    a_uv = uv_points[0][vertex_indices[:, 0]]
    b_uv = uv_points[0][vertex_indices[:, 1]]
    c_uv = uv_points[0][vertex_indices[:, 2]]

    a_3d = points_3d[0][vertex_indices[:, 0]]
    b_3d = points_3d[0][vertex_indices[:, 1]]
    c_3d = points_3d[0][vertex_indices[:, 2]]

    angle_a_uv = ((b_uv - a_uv)*(c_uv - a_uv)).sum(dim=1)/(torch.norm(b_uv - a_uv,dim=1)*torch.norm(c_uv - a_uv,dim=1))
    angle_b_uv = ((a_uv - b_uv)*(c_uv - b_uv)).sum(dim=1)/(torch.norm(a_uv - b_uv,dim=1)*torch.norm(c_uv - b_uv,dim=1))
    angle_c_uv = ((b_uv - c_uv)*(a_uv - c_uv)).sum(dim=1)/(torch.norm(b_uv - c_uv,dim=1)*torch.norm(a_uv - c_uv,dim=1))


    angle_a_3d = ((b_3d - a_3d)*(c_3d - a_3d)).sum(dim=1)/(torch.norm(b_3d - a_3d,dim=1)*torch.norm(c_3d - a_3d,dim=1))
    angle_b_3d = ((a_3d - b_3d)*(c_3d - b_3d)).sum(dim=1)/(torch.norm(a_3d - b_3d,dim=1)*torch.norm(c_3d - b_3d,dim=1))
    angle_c_3d = ((b_3d - c_3d)*(a_3d - c_3d)).sum(dim=1)/(torch.norm(b_3d - c_3d,dim=1)*torch.norm(a_3d - c_3d,dim=1))

    loss_a = ((angle_a_uv-angle_a_3d).abs()).mean()
    loss_b = ((angle_b_uv-angle_b_3d).abs()).mean()
    loss_c = ((angle_c_uv-angle_c_3d).abs()).mean()
    loss = (loss_a+loss_b+loss_c)/3
    return loss



# Equation 23 in the paper
# The overall distortion constraint is computed by iterating all pairs of triangle edges.
# (Note that X is different in each iteration.)
# "triangles" has the shape of [1, 1600, 3] and contains 3D triangle face on the mesh.
# "uv_points" has the shape of [1, X, 3] and it is the actual unwrapped 2D UV coordinates. This would be our final UV parameterization after training.
# "point_3d" has the shape of [1, X, 3] and is the 3D point (vertex) coordinates. 
# "charts_prob" has the shape of [1, X] is a probability that is being assigned to each 3D vertex to determine which point belongs to which chart.
def isometric_loss_by_prob_l1_percent(uv_points, points_3d, triangles, charts_prob):

    vertex_indices = triangles[0]

    # Extracting the UV coordinates of each vertex in a triangle face.
    # In other words, these three points "a_uv", "b_uv", and "c_uv"
    # are the vertices of a 2D triangle in the UV space.
    a_uv = uv_points[0][vertex_indices[:, 0]]  # A [1600, 2]
    b_uv = uv_points[0][vertex_indices[:, 1]]  # B [1600, 2]
    c_uv = uv_points[0][vertex_indices[:, 2]]  # C [1600, 2]
    
    # Extracting the 3D vertices belonging to a triangle in 3D.
    a_3d = points_3d[0][vertex_indices[:, 0]]  # A [1600, 3]
    b_3d = points_3d[0][vertex_indices[:, 1]]  # B [1600, 3]
    c_3d = points_3d[0][vertex_indices[:, 2]]  # C [1600, 3]

    # Extract the chart probability of each 3D point
    a_prob = charts_prob[0][vertex_indices[:, 0]]  # A [1600]
    b_prob = charts_prob[0][vertex_indices[:, 1]]  # B [1600]
    c_prob = charts_prob[0][vertex_indices[:, 2]]  # C [1600]


    # Equation 21 in the paper
    # For an edge e belonging to a certain 3D triangle with two end
    # points pi and pj, whose chart assignment scores with respect
    # to the k-th chart are S(i, k) and S(j, k), we deduce a chart
    # assignment score for this edge by averaging the scores of the
    # two end points, as given by: s(k)(e) = (S(i, k) + S(j, k))/2.
    ab_prob = (a_prob + b_prob) / 2
    bc_prob = (b_prob + c_prob) / 2
    ac_prob = (a_prob + c_prob) / 2

    # The first part of the equation 22 in the paper - Computing the length of each edge in each 2D triangle.
    # The equation 22 says that we want the length of each edge in each triangle on the 3D space
    # will be the same as the length of its corresponding edge in the corresponding triangle in 2D.
    dis_ab_uv = ((a_uv - b_uv) * (a_uv - b_uv)).sum(dim=1).sqrt()
    dis_bc_uv = ((b_uv - c_uv) * (b_uv - c_uv)).sum(dim=1).sqrt()
    dis_ac_uv = ((a_uv - c_uv) * (a_uv - c_uv)).sum(dim=1).sqrt()

    # The second part of the equation 22 in the paper - Computing the length of each edge in each 3D triangle.
    dis_ab_3d = ((a_3d - b_3d) * (a_3d - b_3d)).sum(dim=1).sqrt()
    dis_bc_3d = ((b_3d - c_3d) * (b_3d - c_3d)).sum(dim=1).sqrt()
    dis_ac_3d = ((a_3d - c_3d) * (a_3d - c_3d)).sum(dim=1).sqrt()

    # Equation 23
    # Comuting the difference between the length of each edge in each triangle in 3D and the length of each edge in 2D.
    # This is how we enforce the "isometry" property to keep the original edge lengths.
    # TODO: Two ideas could be adding:
    # 1- Equiareal - Mainating original triangles or facets areas.
    # 2- Conformality - Preserving angles between edges.
    loss_ab = (((dis_ab_uv - dis_ab_3d).abs())*ab_prob).sum()
    loss_bc = (((dis_bc_uv - dis_bc_3d).abs())*bc_prob).sum()
    loss_ac = (((dis_ac_uv - dis_ac_3d).abs())*ac_prob).sum()

    loss = (loss_ab + loss_bc + loss_ac) / 3
    return loss



# This function simply computes a weighted average of our per‐vertex occlusion values, using the soft “seam strength” 
# "w_v" as weights. Concretely:
# AO_v[v] is how occluded vertex v is (0 = fully visible, 1 = fully occluded)
# w_v[v] is how strongly vertex v is predicted to lie on the seam (higher = more “seamy”)
def boundary_occlusion_loss(AO_v: torch.Tensor, w_v:   torch.Tensor) -> torch.Tensor:
    # We add a tiny epsilon so we never divide by zero if all "w_v" happened to be (nearly) zero.
    w_sum = w_v.sum() + 1e-6
    # Each vertex’s occlusion "AO_v" is multiplied by its seam strength "w_v". If the network puts
    # high seam‐weight on a highly occluded vertex, that contributes a large term to the sum; 
    # if it instead puts seam‐weight on a visible vertex (low AO_v), that contributes less.
    L = torch.sum(w_v * AO_v) / w_sum
    return L


def save_1d_plot(arr, save_dir, filename="plot.png", dpi=300):

    # Ensure the folder exists
    os.makedirs(save_dir, exist_ok=True)

    # Create the plot
    plt.figure()
    plt.plot(arr)
    plt.xlabel("Iteration")
    plt.ylabel(filename)
    plt.title(filename)

    # Save and clean up
    full_path = os.path.join(save_dir, filename + ".png")
    plt.savefig(full_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {full_path}")


def save_list_lines(
    items: Iterable[Any],
    out_dir: Union[str, Path],
    filename: str = "list.txt",
    mode: str = "w",
    encoding: str = "utf-8",
    strip_newlines: bool = True,
) -> Path:
    
    if mode not in ("w", "a"):
        raise ValueError("mode must be 'w' (overwrite) or 'a' (append)")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    with out_path.open(mode=mode, encoding=encoding, newline="") as f:
        for item in items:
            line = str(item)
            if strip_newlines:
                # remove stray newline characters from the item itself
                line = line.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
            f.write(line + "\n")

    return out_path


def next_power_of_two(x: int) -> int:
    """Return smallest power of two >= x."""
    return 1 if x == 0 else 2 ** math.ceil(math.log2(x))


# Compose per-chart images named Q_eval_normalized_{i}.png (i=0..charts_number-1)
# into a single atlas image of size atlas_size x atlas_size.
def make_uv_atlas_from_chart_images(
    export_folder: str,
    charts_number: int,
    out_name: str = "uv_atlas_4096.png",
    atlas_size: int = 4096,
    use_pow2_grid: bool = True,
    bg_color=(255,255,255),
    padding_px: int = 16
) -> str:

    # compute grid size
    base_side = math.ceil(math.sqrt(charts_number))
    if use_pow2_grid:
        grid_side = next_power_of_two(base_side)
    else:
        grid_side = base_side

    cells = grid_side * grid_side
    cell_size = atlas_size // grid_side  # integer cell width/height

    # Create blank atlas canvas
    atlas = Image.new("RGB", (atlas_size, atlas_size), color=bg_color)

    # iterate cells row-major and paste images if available
    for i in range(cells):
        r = i // grid_side
        c = i % grid_side
        x0 = c * cell_size
        y0 = r * cell_size

        # compute inner region (respect padding)
        inner_w = cell_size - 2 * padding_px
        inner_h = cell_size - 2 * padding_px
        if inner_w <= 0 or inner_h <= 0:
            raise ValueError("padding_px is too large for the chosen atlas_size/grid configuration")

        if i < charts_number:
            chart_path = os.path.join(export_folder, f"Q_eval_normalized_{i}.png")
            if os.path.exists(chart_path):
                try:
                    img = Image.open(chart_path).convert("RGBA")
                except Exception as e:
                    print(f"[Warning] failed to open {chart_path}: {e}")
                    img = None
            else:
                img = None

            if img is None:
                # create small blank placeholder with faint border
                placeholder = Image.new("RGBA", (inner_w, inner_h), (250,250,250,255))
                atlas.paste(placeholder, (x0 + padding_px, y0 + padding_px))
                continue

            # Resize img to fit inside inner box preserving aspect ratio
            img_w, img_h = img.size
            scale = min(inner_w / img_w, inner_h / img_h, 1.0)  # do not upscale; change to >1 to upscale
            target_w = int(round(img_w * scale))
            target_h = int(round(img_h * scale))
            resized = img.resize((target_w, target_h), resample=Image.LANCZOS)

            # paste centered in the cell (convert to RGB background if needed)
            paste_x = x0 + padding_px + (inner_w - target_w) // 2
            paste_y = y0 + padding_px + (inner_h - target_h) // 2
            # If original has alpha, composite on white background to avoid transparency
            if resized.mode == "RGBA":
                bg = Image.new("RGBA", (target_w, target_h), bg_color + (255,))
                bg.paste(resized, (0,0), mask=resized.split()[3])
                resized_for_paste = bg.convert("RGB")
            else:
                resized_for_paste = resized.convert("RGB")
            atlas.paste(resized_for_paste, (paste_x, paste_y))
        else:
            # cell is unused/empty: leave blank (bg_color)
            continue

    out_path = os.path.join(export_folder, out_name)
    atlas.save(out_path, format="PNG")
    print(f"[Saved] UV atlas: {out_path} (grid {grid_side}x{grid_side}, cell size {cell_size}px)")
    return out_path
