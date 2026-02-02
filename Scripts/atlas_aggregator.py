
############################## USAGE ######################################
# python make_atlas.py --part_root "../checkpoints/Rabbit" --atlas_size 1024
##########################################################################

import os
import json
import argparse
import numpy as np
from glob import glob
from PIL import Image

# ------------------------------- PARSE ARGUMENTS -------------------------------
parser = argparse.ArgumentParser(
    description="Pack per-part textures & UVs into a single atlas."
)
parser.add_argument(
    "--part_root", type=str, required=True,
    help="Root directory containing each part's export folder"
)
parser.add_argument(
    "--atlas_size", type=int, default=1024,
    help="Size (width and height) of the resulting atlas image"
)
args = parser.parse_args()

# ------------------------------- CONFIG -------------------------------
PART_ROOT  = args.part_root  # e.g. '../checkpoints/Rabbit'
ATLAS_SIZE = args.atlas_size

# find all part export dirs named like “part_R_G_B_…_1”
pattern = os.path.join(PART_ROOT, "part_*_*_*_*" )
part_dirs = sorted(glob(pattern))
if not part_dirs:
    raise RuntimeError(f"No part directories found with pattern: {pattern}")

# compute grid layout
total = len(part_dirs)
grid_count = int(np.ceil(np.sqrt(total)))
tile = ATLAS_SIZE // grid_count

# prepare empty atlas and output data
atlas = np.zeros((ATLAS_SIZE, ATLAS_SIZE, 3), dtype=np.uint8)
final_data = {}

# loop & pack
for idx, pdir in enumerate(part_dirs):
    name = os.path.basename(pdir)
    # parse RGB from folder name: 'part_R_G_B_<...>'
    parts = name.split("_")
    r, g, b = map(int, parts[1:4])
    color = [r, g, b]

    # load uv coords and texture image
    uv_path  = os.path.join(pdir, "uv_coords.npy")
    tex_path = os.path.join(pdir, "part_tex.png")
    uv       = np.load(uv_path)                     # shape (N,2)
    tex_img  = Image.open(tex_path).convert("RGB").resize((tile, tile), Image.LANCZOS)

    # pack into atlas
    row, col = divmod(idx, grid_count)
    y0, x0 = row * tile, col * tile
    atlas[y0:y0+tile, x0:x0+tile] = np.array(tex_img)

    # remap uv coords
    offset = np.array([x0, y0]) / ATLAS_SIZE
    scale  = np.array([tile, tile]) / ATLAS_SIZE
    remapped = uv * scale + offset

    final_data[name] = {
        "color": color,
        "uv": remapped.tolist()
    }

# save atlas and JSON
atlas_path = os.path.join(PART_ROOT, f"final_atlas_{ATLAS_SIZE}.png")
json_path  = os.path.join(PART_ROOT, "final_uvs.json")
Image.fromarray(atlas).save(atlas_path)
print(f"Wrote atlas image to {atlas_path}")
with open(json_path, "w") as f:
    json.dump(final_data, f, indent=2)
print(f"Wrote UV data to {json_path}")