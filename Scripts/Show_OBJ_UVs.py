########################### USAGE ####################################
# python Show_OBJ_UVs.py ../data/Blender/Squidward_House/squidward_house.obj ../data/Blender/Squidward_House
######################################################################

# What this script does:
# - Save ONE UV image that shows only colored UV points (no fills/edges), with white background.
# - Color corresponding 3D vertices with the same island colors and open a trimesh viewer.

import os
import sys
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import colorsys, random
import trimesh


DOT_SIZE = 120       # change this if you want bigger/smaller dots (points)
DPI = 400
FIGSIZE = (16, 16)

def parse_obj_with_vt(path):
    verts = []
    uvs = []
    faces = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v':
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == 'vt':
                u = float(parts[1]); v = float(parts[2]) if len(parts) > 2 else 0.0
                uvs.append((u, v))
            elif parts[0] == 'f':
                v_idx = []; vt_idx = []; has_vt = False
                for tok in parts[1:]:
                    comps = tok.split('/')
                    v_idx.append(int(comps[0]) - 1 if comps[0] else None)
                    if len(comps) >= 2 and comps[1] != '':
                        vt_idx.append(int(comps[1]) - 1)
                        has_vt = True
                    else:
                        vt_idx.append(None)
                faces.append({'v': v_idx, 'vt': vt_idx if has_vt else None})
    return np.array(verts, dtype=float), np.array(uvs, dtype=float), faces

def triangulate_faces(faces):
    tri_v = []; tri_vt = []
    for face in faces:
        v = face['v']; vt = face['vt']
        if len(v) < 3:
            continue
        for i in range(1, len(v)-1):
            tri_v.append((v[0], v[i], v[i+1]))
            tri_vt.append((vt[0], vt[i], vt[i+1]) if vt is not None else None)
    return tri_v, tri_vt

class DSU:
    def __init__(self,n): self.p=list(range(n))
    def find(self,a):
        while self.p[a]!=a:
            self.p[a]=self.p[self.p[a]]; a=self.p[a]
        return a
    def union(self,a,b):
        ra=self.find(a); rb=self.find(b)
        if ra!=rb: self.p[rb]=ra

def gen_island_colors(n, seed=42):
    random.seed(seed)
    hex_list=[]; rgba=[]
    for i in range(n):
        h=(i*0.618033988749895)%1.0
        s=0.6+0.4*random.random(); v=0.7+0.3*random.random()
        r,g,b = colorsys.hsv_to_rgb(h,s,v)
        ri,gi,bi = int(r*255), int(g*255), int(b*255)
        hex_list.append('#{:02x}{:02x}{:02x}'.format(ri,gi,bi))
        rgba.append((ri,gi,bi,255))
    return hex_list, np.array(rgba, dtype=np.uint8)

def run(obj_path, export_folder):
    os.makedirs(export_folder, exist_ok=True)

    verts, uvs, faces = parse_obj_with_vt(obj_path)
    if uvs.size == 0:
        raise RuntimeError("OBJ has no vt entries (no UVs).")
    tri_v, tri_vt = triangulate_faces(faces)
    if len(tri_v) == 0:
        raise RuntimeError("No triangles found after triangulation.")

    # map vt -> triangles
    vt_to_tris = defaultdict(list)
    for t_idx, vt_trip in enumerate(tri_vt):
        if vt_trip is None: continue
        for vt in vt_trip:
            if vt is not None: vt_to_tris[vt].append(t_idx)

    # group triangles into islands
    n_tri = len(tri_v)
    dsu = DSU(n_tri)
    for vt, tri_list in vt_to_tris.items():
        if len(tri_list) <= 1: continue
        first = tri_list[0]
        for other in tri_list[1:]:
            dsu.union(first, other)

    # island id per triangle
    comp_map = {}; island_ids = []
    for t in range(n_tri):
        root = dsu.find(t)
        if root not in comp_map: comp_map[root]=len(comp_map)
        island_ids.append(comp_map[root])
    island_ids = np.array(island_ids, dtype=int)
    n_islands = int(island_ids.max())+1 if len(island_ids)>0 else 0

    # vt indices per island
    island_to_vts = [set() for _ in range(max(1,n_islands))]
    for t_idx, vt_trip in enumerate(tri_vt):
        if vt_trip is None: continue
        isl = island_ids[t_idx]
        for vt in vt_trip:
            if vt is not None: island_to_vts[isl].add(vt)

    # vt -> vertex mapping
    vt_to_verts = defaultdict(list)
    for t_idx, vt_trip in enumerate(tri_vt):
        if vt_trip is None: continue
        v_trip = tri_v[t_idx]
        for j in range(3):
            vt = vt_trip[j]; v = v_trip[j]
            if vt is not None and v is not None:
                vt_to_verts[vt].append(v)

    # generate colors
    hex_colors, rgba_colors = gen_island_colors(max(1,n_islands), seed=42)
    default_hex = '#ffffff'   # white for any missing vt (ensures gaps stay white)
    default_grey_rgba = np.array([180,180,180,255], dtype=np.uint8)

    # assign vertex colors by majority island (for seams)
    vert_island_lists = [[] for _ in range(len(verts))]
    for isl, vt_set in enumerate(island_to_vts):
        for vt in vt_set:
            for v in vt_to_verts.get(vt, []):
                vert_island_lists[v].append(isl)
    vertex_colors = np.zeros((len(verts),4), dtype=np.uint8)
    for vi, lst in enumerate(vert_island_lists):
        if not lst:
            vertex_colors[vi] = default_grey_rgba
        else:
            most = Counter(lst).most_common(1)[0][0]
            vertex_colors[vi] = rgba_colors[most]

    # build trimesh with vertex colors
    tri_faces = np.array(tri_v, dtype=int)
    mesh = trimesh.Trimesh(vertices=verts, faces=tri_faces, process=False)
    mesh.visual.vertex_colors = vertex_colors

    # save as .glb
    glb_path = os.path.join(export_folder, "colorized_mesh.glb")
    try:
        mesh.export(glb_path)
        print("Saved colorized mesh to:", glb_path)
    except Exception as e:
        print("Failed to save GLB:", e)

    # create per-vt hex color list; vt not in any island -> white
    vt_hex = [default_hex] * len(uvs)
    for isl, vt_set in enumerate(island_to_vts):
        col = hex_colors[isl]
        for vt in vt_set:
            vt_hex[vt] = col

    # --------- Save UV points image (white background, only dots) ----------
    out_path = os.path.join(export_folder, "UV_points_only.png")

    # ensure white figure/axes background
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')

    # scatter: small dot marker, no edge, opaque
    ax.scatter(uvs[:,0], uvs[:,1], s=DOT_SIZE, c=vt_hex, marker='.', edgecolors='none', alpha=1.0)

    # constrain to 0..1 UV box (change if your UVs are outside)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # save with explicit white facecolor and no padding so background stays white
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close(fig)
    print("Saved UV points image (white gaps) to:", out_path)

    # --------- Show 3D viewer with vertex colors ----------
    print("Opening 3D viewer with vertex colors. Close window to finish.")
    try:
        mesh.show()
    except Exception as e:
        print("mesh.show() failed:", e)
        # fallback: try to save preview PNG
        try:
            scene = mesh.scene()
            png = scene.save_image(resolution=(1024,768))
            if png is not None:
                from PIL import Image
                import io
                im = Image.open(io.BytesIO(png))
                preview_path = os.path.join(export_folder, "3d_colored_preview.png")
                im.save(preview_path)
                print("Saved 3D preview to:", preview_path)
        except Exception as ee:
            print("Fallback render failed:", ee)

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python uv_points_white_background_and_color_3d.py /path/to/your.obj /path/to/export_folder")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
