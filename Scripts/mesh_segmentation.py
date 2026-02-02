import os
import argparse
import trimesh
import numpy as np

# ------------------------------- PARSE ARGUMENTS -------------------------------
parser = argparse.ArgumentParser(
    description="Segment a colored GLB into per-color OBJ parts."
)
parser.add_argument(
    "--glb_path", type=str, required=True,
    help="Path to the input colored GLB file"
)
parser.add_argument(
    "--output_dir", type=str, required=True,
    help="Directory to write part_R_G_B.obj files"
)
args = parser.parse_args()

GLB_PATH = args.glb_path
OUT_DIR  = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------- LOAD AND FLATTEN -------------------------------
print(f"[SEGMENTATION] Loading {GLB_PATH} and flattening into single mesh...")
mesh = trimesh.load(
    GLB_PATH,
    process=False,
    force='mesh'
)

# ------------------------------- EXTRACT FACE COLORS -------------------------------
face_colors = mesh.visual.face_colors[:, :3]  # drop alpha
unique_colors, inv_idx = np.unique(face_colors, axis=0, return_inverse=True)

# ------------------------------- CARVE AND EXPORT ---------------------------------
for idx, color in enumerate(unique_colors):
    mask = (inv_idx == idx)
    face_indices = np.nonzero(mask)[0]
    if len(face_indices) <= 2:
        continue
    submesh = mesh.submesh([face_indices], append=True, repair=False)
    r, g, b = color.tolist()
    # Generate .obj files
    out_name = f"part_{r}_{g}_{b}.obj"
    out_path = os.path.join(OUT_DIR, out_name)
    submesh.export(out_path)
    # Generate .glb files
    out_name = f"part_{r}_{g}_{b}.glb"
    out_path = os.path.join(OUT_DIR, out_name)
    submesh.export(out_path)

    print(f"[SEGMENTATION] Wrote {out_path}")
