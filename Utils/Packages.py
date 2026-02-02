import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import sys
import time
import h5py
import glob
import scipy
import shutil
import igl
import pickle
import sklearn
import trimesh
import argparse
import itertools
import matplotlib
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import point_cloud_utils as pcu
from scipy.spatial.transform import Rotation as scipy_R

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import IPython
IPython.display.clear_output()


