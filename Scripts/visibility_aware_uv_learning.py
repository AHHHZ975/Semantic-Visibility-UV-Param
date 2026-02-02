import os
import subprocess
from glob import glob
import argparse

# --------------------------------------------- Parse arguments ------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Segment a GLB, then train and test UV parameterization estimation on each part, and build the final atlas."
)
parser.add_argument(
    "--export_dir", type=str, required=True,
    help="Directory where mesh_segmentation.py will output part_*.obj files"
)
parser.add_argument(
    "--obj_dir", type=str, required=True,
    help="Path to the input colored 3D GLB file"
)
args = parser.parse_args()

# --------------------------------------------- Config ------------------------------------------------------
OBJ_PATH       = args.obj_dir               # Path to the input colored 3D GLB file
EXPORT_DIR     = args.export_dir            # Directory where the final results including the final UV parameterization will be saved.
TRAIN_SCRIPT   = "train.py"            # A script to train our UV parameterization estimator model
TEST_SCRIPT    = "test.py"             # A script to run our trained UV parameterization estimator model at inference stage
NUM_ITER       = 2000                      # An arg to train_ours.py -> It is the number of training iteration for training of UV MLP models.
VIS_AWARE      = "True"                     # Enable/disable our visibility-awareness feature for UV parameterization
SEM_AWARE      = "False"                    # Enable/disable our semantic-awareness feature for UV parameterization

# ------------------------------------------------------ Train and Test ------------------------------------------------------
# Train
train_args = [
    "python", TRAIN_SCRIPT,
    OBJ_PATH, EXPORT_DIR,
    str(NUM_ITER),
    VIS_AWARE, SEM_AWARE
]
print("[TRAIN]", " ".join(train_args))
subprocess.run(train_args, check=True) # Call external train script ("train_ours.py")


# Test
obj_name = os.path.splitext(os.path.basename(OBJ_PATH))[0]
EXPORT_DIR = os.path.join(EXPORT_DIR, f"{obj_name}_1")
test_args = [
    "python", TEST_SCRIPT,
    OBJ_PATH, EXPORT_DIR,
    VIS_AWARE, SEM_AWARE
]
print("[TEST]", " ".join(test_args))
subprocess.run(test_args, check=True) # Call external train script ("test.py")
