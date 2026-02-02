import os
import subprocess
from glob import glob
import argparse

# --------------------------------------------- Parse arguments ------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Segment a GLB, then train and test UV parameterization estimation on each part, and build the final atlas."
)
parser.add_argument(
    "--seg_out_dir", type=str, required=True,
    help="Directory where mesh_segmentation.py will output part_*.obj files"
)
parser.add_argument(
    "--glb_dir", type=str, required=True,
    help="Path to the input colored 3D GLB file"
)
args = parser.parse_args()

# --------------------------------------------- Config ------------------------------------------------------
SEG_OUT_DIR    = args.seg_out_dir           # Directory where "mesh_segmentation.py" will output part_*.obj files
GLB_PATH       = args.glb_dir               # Path to the input colored 3D GLB file
CKPT_DIR       = "../checkpoints"           # Directory where the final results including the final UV parameterization will be saved.
TRAIN_SCRIPT   = "train.py"            # A script to train our UV parameterization estimator model
TEST_SCRIPT    = "test.py"             # A script to run our trained UV parameterization estimator model at inference stage
MAKE_ATLAS_SCRIPT = "atlas_aggregator.py"   # A script to aggregate all estimated UV coordinates (correponding to each 3D segmented part of the original GLB file) into one unified texture image.
NUM_ITER       = 2000                       # An arg to train_ours.py -> It is the number of training iteration for training of UV MLP models.
VIS_AWARE      = "False"                    # Enable/disable our visibility-awareness feature for UV parameterization
SEM_AWARE      = "True"                     # Enable/disable our semantic-awareness feature for UV parameterization
ATLAS_SIZE     = 4096                       # The size of the final texture image that would contain the final UV parameterization: ATLAS_SIZE * ATLAS_SIZE

# ------------------------------------------------------ 3D Segmentation ------------------------------------------------------
# First, run our 3D segmentation script that create and save separate 3D parts of an given 3D .glb file.
# This separates the parts based on their color. Specifically, the input .glb file is a 3D object obtained
# from the SAMesh model (a 3D segmentation method) which colorizes each part in a 3D object based on its semantic.
# (e.g. in the case of bunny, ear would be pink, the head is purple, etc.). Then, we run the following script,
# to extract and save each 3D part into the "--output_dir". Once we have all segmented parts, we then perform
# the training of our UV estimator model.
print(f"[SEGMENTATION] running mesh_segmentation.py → parts in {SEG_OUT_DIR}...")
subprocess.run(
    [
        "python", "mesh_segmentation.py",
        "--output_dir", SEG_OUT_DIR,
        "--glb_path", GLB_PATH
    ],
    check=True
)
print("[SEGMENTATION] done.")

# ------------------------------------------------------ Train and Test ------------------------------------------------------
# Perform the training of our UV parameterization estimator model on each 3D part that
# are extracted and segmented from the previous step.
os.makedirs(CKPT_DIR, exist_ok=True)
part_paths = sorted(glob(os.path.join(SEG_OUT_DIR, "part_*.obj")))
if not part_paths:
    raise RuntimeError(f"No parts found in {SEG_OUT_DIR}")

# track export folders for atlas
export_root = os.path.join(CKPT_DIR, os.path.basename(SEG_OUT_DIR)) # This "basename" gives the name of the most inner folder in the directory
export_folders = []

# Perform the UV learning step for each segmented part in a directory
for obj_path in part_paths:
    name = os.path.splitext(os.path.basename(obj_path))[0]
    export_folder = os.path.join(export_root, f"{name}_1")
    export_folders.append(export_folder)

    # Train
    train_args = [
        "python", TRAIN_SCRIPT,
        obj_path, CKPT_DIR,
        str(NUM_ITER),
        VIS_AWARE, SEM_AWARE
    ]
    print("[TRAIN]", " ".join(train_args))
    subprocess.run(train_args, check=True) # Call external train script ("train_ours.py")

    # Test
    os.makedirs(export_folder, exist_ok=True)
    test_args = [
        "python", TEST_SCRIPT,
        obj_path, export_folder,
        VIS_AWARE, SEM_AWARE
    ]
    print("[TEST]", " ".join(test_args))
    subprocess.run(test_args, check=True) # Call external train script ("test.py")

    print(f"[DONE] {name}: outputs in {export_folder}\n")

# ------------------------------------------------------ Build final atlas ------------------------------------------------------
# Once we learned and stored the UV parameterization for each 3D segmented part, 
# we aggregate all those small UV parameterization (Atlases) into one big unified
# atlast image.
print(f"[ATLAS] calling {MAKE_ATLAS_SCRIPT}...")

# Call external atlas script ("make_atlas.py") 
subprocess.run(
    [
        "python", MAKE_ATLAS_SCRIPT,
        "--part_root", export_root,
        "--atlas_size", str(ATLAS_SIZE)
    ],
    check=True
)
print("[ATLAS] completed.")