import glob
import json
import os
import copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import pymeshlab
import trimesh
import networkx as nx
import igraph
from numpy.random import RandomState
from trimesh.base import Trimesh, Scene
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from omegaconf import OmegaConf

from torchtyping import TensorType


NumpyTensor = TensorType
TorchTensor = TensorType





EPSILON = 1e-20
SCALE = 1e6



def _ensure_face_colors(face_colors, n_faces):
    """
    Convert various color formats to an (n_faces, 4) uint8 RGBA array.
    Accepts:
      - (n_faces, 3) or (n_faces, 4) arrays, float [0..1] or uint8
      - single color (3,) or (4,)
    """
    fc = np.asarray(face_colors)

    # Handle single color
    if fc.ndim == 1:
        fc = np.tile(fc, (n_faces, 1))

    # If provided per-vertex for faces (F,3,4) -> reduce to per-face by taking first vertex color
    if fc.ndim == 3 and fc.shape[1] == 3:
        # expecting shape (F, 3, 3 or 4)
        fc = fc[:, 0, :]

    # Now fc should be (F,3) or (F,4)
    if fc.shape[1] == 3:
        # add alpha channel
        alpha_col = np.full((fc.shape[0], 1), 255, dtype=np.uint8)
        fc = np.concatenate([fc, alpha_col], axis=1)

    # If floats in [0,1], convert to 0-255
    if np.issubdtype(fc.dtype, np.floating):
        if fc.max() <= 1.0 + 1e-12:
            fc = (fc * 255.0).round().astype(np.uint8)
        else:
            # floats >1, cast to uint8 safely
            fc = np.clip(fc, 0, 255).round().astype(np.uint8)
    else:
        fc = fc.astype(np.uint8)

    # Ensure correct length
    if fc.shape[0] != n_faces:
        # if someone passed a palette with fewer entries, tile or repeat as fallback
        fc = np.tile(fc.reshape(-1, fc.shape[1])[:1], (n_faces, 1))

    return fc

def duplicate_verts(mesh: Trimesh) -> Trimesh:
    """
    Duplicate vertices so colors can be assigned per-face without interpolation.
    Works even when mesh.visual is textured (TextureVisuals) or colored.

    Returns a new Trimesh with per-face colors stored in face_colors.
    """
    # Make sure faces are available
    faces = mesh.faces
    # Build duplicated-vertex array (each face gets its own 3 unique verts)
    verts = mesh.vertices[faces.reshape(-1), :]

    new_faces = np.arange(0, verts.shape[0]).reshape(-1, 3)

    # Try to pull per-face colors from different visual representations
    face_colors = None

    # 1) If mesh has explicit face_colors attribute (ColorVisuals)
    if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
        face_colors = mesh.visual.face_colors

    # 2) If vertex colors exist, derive a per-face color (use first vertex color of each face)
    elif hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        vc = np.asarray(mesh.visual.vertex_colors)
        # vc might be shape (V,3) or (V,4). Build per-face (F,3,4) then pick first column -> (F,4)
        try:
            per_face_vc = vc[faces]          # (F,3,3/4)
            face_colors = per_face_vc[:, 0, :]  # use color of first vertex of face
        except Exception:
            # fallback to uniform gray
            face_colors = np.tile(np.array([200, 200, 200, 255], dtype=np.uint8), (faces.shape[0], 1))

    # 3) Textured meshes (TextureVisuals) — we don't rasterize textures here; default to gray
    else:
        # safe default: opaque gray
        face_colors = np.tile(np.array([200, 200, 200, 255], dtype=np.uint8), (faces.shape[0], 1))

    face_colors = _ensure_face_colors(face_colors, faces.shape[0])

    # Because we duplicated vertices, need to expand face_colors to per-face-vertex ordering:
    # Each face is duplicated into 3 vertices in `verts`, so we need to repeat each face color 3 times.
    face_colors_expanded = np.repeat(face_colors, 3, axis=0)

    return Trimesh(vertices=verts, faces=new_faces, face_colors=face_colors_expanded, process=False)



def handle_pose(pose: NumpyTensor['4 4']) -> NumpyTensor['4 4']:
    """
    Handles common case that results in numerical instability in rendering faceids:

        ...
        pose, _ = scene.graph[name]
        pose = handle_pose(pose)
        ...
    """
    identity = np.eye(4)
    if np.allclose(pose, identity, atol=1e-6):
        return identity
    return pose


def transform(pose: NumpyTensor['4 4'], vertices: NumpyTensor['nv 3']) -> NumpyTensor['nv 3']:
    """
    """
    homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
    return (pose @ homogeneous.T).T[:, :3]


def scene2mesh(scene: Scene, process=True) -> Trimesh:
    """
    """
    if len(scene.geometry) == 0:
        mesh = None  # empty scene
    else:
        data = []
        for name, geom in scene.geometry.items():
            if name in scene.graph:
                pose, _ = scene.graph[name]
                pose = handle_pose(pose)
                vertices = transform(pose, geom.vertices)
            else:
                vertices = geom.vertices
            # process=True removes duplicate vertices (needed for correct face graph), affecting face indices but not faces.shape
            data.append(Trimesh(vertices=vertices, faces=geom.faces, visual=geom.visual, process=process))
        
        mesh = trimesh.util.concatenate(data)
        mesh = Trimesh(vertices=mesh.vertices, faces=mesh.faces, visual=mesh.visual, process=process)
    return mesh



def bounding_box(vertices: NumpyTensor['n 3']) -> NumpyTensor['2 3']:
    """
    Compute bounding box from vertices.
    """
    return np.array([vertices.min(axis=0), vertices.max(axis=0)])


def bounding_box_centroid(vertices: NumpyTensor['n 3']) -> NumpyTensor['3']:
    """
    Compute bounding box centroid from vertices.
    """
    return bounding_box(vertices).mean(axis=0)



def norm_mesh(mesh: Trimesh) -> Trimesh:
    """
    Normalize mesh vertices to bounding box [-1, 1]. 
    
    NOTE:: In place operation that consumes mesh.
    """
    centroid = bounding_box_centroid(mesh.vertices)
    mesh.vertices -= centroid
    mesh.vertices /= np.abs(mesh.vertices).max()
    mesh.vertices *= (1 - 1e-3)
    return mesh

def read_mesh(filename: Path, norm=False, process=True) -> Trimesh | None:
    """
    Read/convert a possible scene to mesh. 
    
    If conversion occurs, the returned mesh has only vertex and face data i.e. no texture information.

    NOTE: sometimes process=True does unexpected actions, such as cause face color misalignment with faces
    """    
    source = trimesh.load(filename)

    if isinstance(source, trimesh.Scene):
        mesh = scene2mesh(source, process=process)
    else:
        assert(isinstance(source, trimesh.Trimesh))
        mesh = source
    if norm:
        mesh = norm_mesh(mesh)
    return mesh




def partition_cost(
    mesh           : Trimesh,
    partition      : NumpyTensor['f'],
    cost_data      : NumpyTensor['f num_components'],
    cost_smoothness: NumpyTensor['e']
) -> float:
    """
    """
    cost = 0
    for f in range(len(partition)):
        cost += cost_data[f, partition[f]]
    for i, edge in enumerate(mesh.face_adjacency):
        f1, f2 = int(edge[0]), int(edge[1])
        if partition[f1] != partition[f2]:
            cost += cost_smoothness[i]
    return cost


def construct_expansion_graph(
    label          : int,
    mesh           : Trimesh,
    partition      : NumpyTensor['f'],
    cost_data      : NumpyTensor['f num_components'],
    cost_smoothness: NumpyTensor['e']
) -> nx.Graph:
    """
    """
    G = nx.Graph() # undirected graph
    A = 'alpha'
    B = 'alpha_complement'

    node2index = {}
    G.add_node(A)
    G.add_node(B)
    node2index[A] = 0
    node2index[B] = 1
    for i in range(len(mesh.faces)):
        G.add_node(i)
        node2index[i] = 2 + i

    aux_count = 0
    for i, edge in enumerate(mesh.face_adjacency): # auxillary nodes
        f1, f2 = int(edge[0]), int(edge[1])
        if partition[f1] != partition[f2]:
            a = (f1, f2)
            if a in node2index: # duplicate edge
                continue
            G.add_node(a)
            node2index[a] = len(mesh.faces) + 2 + aux_count
            aux_count += 1

    for f in range(len(mesh.faces)):
        G.add_edge(A, f, capacity=cost_data[f, label])
        G.add_edge(B, f, capacity=float('inf') if partition[f] == label else cost_data[f, partition[f]])

    for i, edge in enumerate(mesh.face_adjacency):
        f1, f2 = int(edge[0]), int(edge[1])
        a = (f1, f2)
        if partition[f1] == partition[f2]:
            if partition[f1] != label:
                G.add_edge(f1, f2, capacity=cost_smoothness[i])
        else:
            G.add_edge(a, B, capacity=cost_smoothness[i])
            if partition[f1] != label:
                G.add_edge(f1, a, capacity=cost_smoothness[i])
            if partition[f2] != label:
                G.add_edge(a, f2, capacity=cost_smoothness[i])
    
    return G, node2index


def repartition(
    mesh: trimesh.Trimesh,
    partition      : NumpyTensor['f'],
    cost_data      : NumpyTensor['f num_components'],
    cost_smoothness: NumpyTensor['e'],
    smoothing_iterations: int,
    _lambda=1.0,
):
    A = 'alpha'
    B = 'alpha_complement'
    labels = np.unique(partition)

    cost_smoothness = cost_smoothness * _lambda

    # networkx broken for float capacities
    #cost_data       = np.round(cost_data       * SCALE).astype(int)
    #cost_smoothness = np.round(cost_smoothness * SCALE).astype(int)

    cost_min = partition_cost(mesh, partition, cost_data, cost_smoothness)

    for i in range(smoothing_iterations):

        #print('Repartition iteration ', i)
        
        for label in tqdm(labels):
            G, node2index = construct_expansion_graph(label, mesh, partition, cost_data, cost_smoothness)
            index2node = {v: k for k, v in node2index.items()}

            '''
            _, (S, T) = nx.minimum_cut(G, A, B)
            assert A in S and B in T
            S = np.array([v for v in S if isinstance(v, int)]).astype(int)
            T = np.array([v for v in T if isinstance(v, int)]).astype(int)
            '''

            G = igraph.Graph.from_networkx(G)
            outputs = G.st_mincut(source=node2index[A], target=node2index[B], capacity='capacity')
            S = outputs.partition[0]
            T = outputs.partition[1]
            assert node2index[A] in S and node2index[B] in T
            S = np.array([index2node[v] for v in S if isinstance(index2node[v], int)]).astype(int)
            T = np.array([index2node[v] for v in T if isinstance(index2node[v], int)]).astype(int)

            assert (partition[S] == label).sum() == 0 # T consists of those assigned 'alpha' and S 'alpha_complement' (see paper)
            partition[T] = label

            cost = partition_cost(mesh, partition, cost_data, cost_smoothness)
            if cost > cost_min:
                raise ValueError('Cost increased. This should not happen because the graph cut is optimal.')
            cost_min = cost
    
    return partition


def prep_mesh_shape_diameter_function(source: Trimesh | Scene) -> Trimesh:
    """
    """
    if isinstance(source, trimesh.Scene):
        source = scene2mesh(source)
    source.merge_vertices(merge_tex=True, merge_norm=True)
    return source


def colormap_shape_diameter_function(mesh: Trimesh, sdf_values: NumpyTensor['f']) -> Trimesh:
    """
    """
    assert len(mesh.faces) == len(sdf_values)
    mesh = duplicate_verts(mesh) # needed to prevent face color interpolation
    mesh.visual.face_colors = trimesh.visual.interpolate(sdf_values, color_map='jet')
    return mesh


def colormap_partition(mesh: Trimesh, partition: NumpyTensor['f']) -> Trimesh:
    """
    """
    assert len(mesh.faces) == len(partition)
    palette = RandomState(0).randint(0, 255, (np.max(partition) + 1, 3)) # must init every time to get same colors
    mesh = duplicate_verts(mesh) # needed to prevent face color interpolation
    mesh.visual.face_colors = palette[partition]
    return mesh


# Recommended defaults (used in our experiments): rays=64, cone_amplitude=120, alpha=5. Increasing rays improves robustness at cost of runtime.
def shape_diameter_function(mesh: Trimesh, norm=True, alpha=5, rays=64, cone_amplitude=120) -> NumpyTensor['f']:
    """
    """
    # This function uses pymeshlab's "compute_scalar_by_shape_diameter_function_per_vertex" to estimate a thickness scalar per face    
    mesh = pymeshlab.Mesh(mesh.vertices, mesh.faces)
    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(mesh)
    meshset.compute_scalar_by_shape_diameter_function_per_vertex(rays=rays, cone_amplitude=cone_amplitude)

    # The code queries face_scalar_array() after calling the per-vertex SDF operator.
    sdf_values = meshset.current_mesh().face_scalar_array()

    # NaN values are clamped to zero
    sdf_values[np.isnan(sdf_values)] = 0

    # If norm=True the raw SDF is normalized to [0,1] and a monotonic re-scaling is applied (parameter alpha)
    # to compress the dynamic range and make clusters easier to separate.
    if norm:
        # normalize and smooth shape diameter function values
        min = sdf_values.min()
        max = sdf_values.max()
        sdf_values = (sdf_values - min) / (max - min)
        sdf_values = np.log(sdf_values * alpha + 1) / np.log(alpha + 1)
    return sdf_values


def partition_faces(mesh: Trimesh, num_components: int, _lambda: float, smooth=True, smoothing_iterations=1, **kwargs) -> NumpyTensor['f']:
    """
    """

    # This function uses pymeshlab's "compute_scalar_by_shape_diameter_function_per_vertex" to estimate a thickness scalar per face
    # (the code queries face_scalar_array() after calling the per-vertex SDF operator).
    sdf_values = shape_diameter_function(mesh, norm=True).reshape(-1, 1)

    # Then, it reshapes ShDF to a column vector and fits GaussianMixture(num_components) (scikit-learn) to the 1D SDF samples.

    # Fit a 1D Gaussian Mixture Model (GMM) to ShDF values to obtain data likelihoods
    gmm = GaussianMixture(num_components)
    gmm.fit(sdf_values)

    # Compute soft class probabilities
    probs = gmm.predict_proba(sdf_values)
    
    # If smooth=False the function simply returns the maximum-probability class per face (argmax).
    if not smooth:
        return np.argmax(probs, axis=1)

    # data and smoothness terms
    # Compute data cost per face (-log(probs+epsilon)). The posterior probabilities encode the likelihood of each face belonging to each semantic component.
    cost_data       = -np.log(probs + EPSILON)
    
    # Construct a smoothness cost between adjacent faces
    # Intuition: small dihedral angles (faces nearly coplanar) yield smaller penalty for keeping the same label; 
    # large dihedral angles (sharp edges) encourage boundaries.
    cost_smoothness = -np.log(mesh.face_adjacency_angles / np.pi + EPSILON)

    # The smoothness cost is scaled by the user _lambda (repartition lambda) to balance data fidelity vs. boundary length.
    cost_smoothness = _lambda * cost_smoothness

    # Initial partition via minimum data cost (per-face argmin).
    # This assigns each face to the GMM component with minimal negative log-likelihood (maximum posterior).
    partition = np.argmin(cost_data, axis=1)

    # Refine partition with iterative alpha-expansion style graph cuts
    partition = repartition(mesh, partition, cost_data, cost_smoothness, smoothing_iterations=smoothing_iterations)
    return partition


def partition2label(mesh: Trimesh, partition: NumpyTensor['f']) -> NumpyTensor['f']:
    """
    """
    edges = trimesh.graph.face_adjacency(mesh=mesh)
    graph = defaultdict(set)
    for face1, face2 in edges:
        graph[face1].add(face2)
        graph[face2].add(face1)
    labels = set(list(np.unique(partition)))
    
    components = []
    visited = set()

    def dfs(source: int):
        stack = [source]
        components.append({source})
        visited.add(source)
        
        while stack:
            node = stack.pop()
            for adj in graph[node]:
                if adj not in visited and partition[adj] == partition[node]:
                    stack.append(adj)
                    components[-1].add(adj)
                    visited.add(adj)

    for face in range(len(mesh.faces)):
        if face not in visited:
            dfs(face)

    partition_output = np.zeros_like(partition)
    label_total = 0
    for component in components:
        for face in component:
            partition_output[face] = label_total
        label_total += 1
    return partition_output


def segment_mesh_sdf(filename: Path | str, config: OmegaConf, extension='glb') -> Trimesh:
    """
    """
    print('Segmenting mesh with Shape Diameter Funciont: ', filename)
    filename = Path(filename)
    config = copy.deepcopy(config)
    config.output = Path(config.output) / filename.stem

    # This loads a file with trimesh.load. If the file is a trimesh.Scene (multiple geometry nodes),
    # scene2mesh() composes node geometries into one mesh while applying node poses (via handle_pose
    # and transform) so the returned object is a single trimesh.Trimesh.
    # If norm=True, norm_mesh() recenters the mesh at its bounding-box centroid and scales vertices
    # so the maximum absolute coordinate is ~1. Normalization improves numerical stability
    # of subsequent geometric operations (pymesh processing and ray-based sampling)
    mesh = read_mesh(filename, norm=True)

    # This converts the trimesh into a pymeshlab.Mesh and ensures a consistent vertex/face topology
    # (merge_vertices()) so pymeshlab's ShDF operators behave robustly.
    mesh = prep_mesh_shape_diameter_function(mesh)

    partition              = partition_faces(mesh, config.num_components, config.repartition_lambda, config.repartition_iterations)
    partition_disconnected = partition2label(mesh, partition)
    faces2label = {int(i): int(partition_disconnected[i]) for i in range(len(partition_disconnected))}

    os.makedirs(config.output, exist_ok=True)
    mesh_colored = colormap_partition(mesh, partition_disconnected)
    mesh_colored.export        (f'{config.output}/{filename.stem}_segmented.{extension}')
    json.dump(faces2label, open(f'{config.output}/{filename.stem}_face2label.json', 'w'))
    return mesh_colored


if __name__ == '__main__':
    import glob
    from natsort import natsorted

    def read_filenames(pattern: str):
        """
        """
        filenames = glob.glob(pattern)
        filenames = map(Path, filenames)
        filenames = natsorted(list(set(filenames)))
        print('Segmenting ', len(filenames), ' meshes')
        return filenames

    # filenames = read_filenames('./assets/*.glb')
    filenames = [Path('./assets/Bullet.glb')]
    config = OmegaConf.load('./mesh_segmentation_shape_diameter_function.yaml')
    for i, filename in enumerate(filenames):
        segment_mesh_sdf(filename, config)