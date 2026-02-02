The code for the paper:
# Unsupervised Representation Learning for 3D Mesh Parameterization with Semantic and Visibility Objectives

* ### See the project webpage [[here]](https://ahhhz975.github.io/Automatic3DMeshParameterization/)
* ### See the paper [[here]](https://arxiv.org/pdf/2509.25094)

<img width="2919" height="1125" alt="ICLR_Teaser_Figure_v1 2" src="https://github.com/user-attachments/assets/1d3d455a-57d1-4d8a-80ae-8fe453b7f11e" />


Recent 3D generative models produce high-quality textures for 3D mesh objects. However, they commonly rely on the heavy assumption that input 3D meshes are accompanied by manual mesh parameterization (UV mapping), a manual task that requires both technical precision and artistic judgment. Industry surveys show that this process often accounts for a significant share of asset creation, creating a major bottleneck for 3D content creators. Moreover, existing automatic methods often ignore two perceptually important criteria: (1) semantic awareness (UV charts should align semantically similar 3D parts across shapes) and (2) visibility awareness (cutting seams should lie in regions unlikely to be seen). To overcome these shortcomings and to automate the mesh parameterization process, we present an unsupervised differentiable framework that augments standard geometry-preserving UV learning with semantic- and visibility-aware objectives. For semantic-awareness, our pipeline (i) segments the mesh into semantic 3D parts, (ii) applies an unsupervised learned per-part UV-parameterization backbone, and (iii) aggregates per-part charts into a unified UV atlas. For visibility-awareness, we use ambient occlusion (AO) as an exposure proxy and back-propagate a soft differentiable AO-weighted seam objective to steer cutting seams toward occluded regions. By conducting qualitative and quantitative evaluations against state-of-the-art methods, we show that the proposed method produces UV atlases that better support texture generation and reduce perceptible seam artifacts compared to recent baselines.

# Qualitative Results of Semantic-Aware UV Parameterization

![Results_SemanticAware_Full](https://github.com/user-attachments/assets/90f1b7bc-fdca-46b0-973d-cdf55f528744)

# Qualitative Results of Visibility-Aware UV Parameterization

<img width="3649" height="1611" alt="ICLR_AO_UV_Seams_Figure_v1 1" src="https://github.com/user-attachments/assets/ae03395f-a0e1-4f46-b48a-f58bb6ac3ebd" />

# Qualitative Results of Checkerboard Texturing

![Results_VisibilityAware_CheckerboardDistortion_Full](https://github.com/user-attachments/assets/97b8921f-d583-4833-9a64-3d415f3b20a0)



## Installation

0- Before starting the installation, make sure that you have NVIDIA GPU Computing Toolkit (e.g. CUDA) installed on your OS and make sure the version of your OS CUDA is the same as the one you are using when installing Pytorch. This is important because some packages we use (e.g. ChamferDistance or EMD) will be installed through the command ```python setup.py install``` which uses the OS CUDA to install those packages. Then, if there is any mismatch between the version of the OS CUDA and Pytorch CUDA, the installation will pause. To download and install the proper version of CUDA toolkit, use this official link from NVIDIA: https://developer.nvidia.com/cuda-downloads


1- We recommend using a virtual environment or a conda environment.
```bash
conda create -n meshparam python=3.10
```

2- Install the proper version of Pytorch library depending on your machine. For more information see the [Pytorch webpage](https://pytorch.org).
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

3- Install other dependencies as follows:
```bash
pip install -r requirements.txt
pip install scikit-learn
python -m pip install -U matplotlib
python -m pip install libigl           # To compute the ambient occlusion (AO) map on 3D meshes.
pip install trimesh
```

4- Install the Chamfer Distance (CD) package:
```bash
cd cdbs/CD
python setup.py install
```

5- Install the Earth-Mover's Distance (EMD) package:
```bash
cd cdbs/EMD
python setup.py install
```

## Installing the SAMesh project (3D Partitioning)

1- You need to use a virtual environment or a conda environment different than the one used to install the FlexPara as the SAMesh requires Python >= 3.12
```bash
conda create -n samesh python=3.12
```

2- Clone the repository - SAMesh relies on one or more git submodules (e.g. the Segment Anything v2 code). Make sure you pull them in:
```bash
git clone https://github.com/gtangg12/samesh.git
cd samesh
```

3- Initialize and update submodules:
```bash
git submodule update --init --recursive
```

4- Install the main package in editable mode - This will install SAMesh and make it easy to pick up any local changes:
```bash
pip install -e .
```

5- Install each submodule in editable mode - After step 3 you should have submodule folders under your repo (for example third_party/SAM2 or similar). For each one, do:
```bash
cd path/to/submodule
pip install -e .
cd ../../
```

6- The code should be ready to run but in case you face any errors related to EGL, install ```mesalib``` first to use ```osmesa``` instead of ```egl``` ([Reference](https://github.com/mcfletch/pyopengl/issues/10#issuecomment-1722334253)):
```bash
conda install -c conda-forge mesalib
```
Then, if required, chanfe the environment variable to osmesa:
```bash
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"
```


6- Troubleshooting common build errors:

- If building SAM2 fails, try installing with no build isolation:
```bash
pip install --no-build-isolation -e .
```

- If you see errors from PyOpenGL/pyrender, pin your version:
```bash
pip install PyOpenGL==3.1.7
```

7- Download a SAM2 (not SAM 2.1!) checkpoint from the [official SAM2 repo](https://github.com/facebookresearch/sam2?tab=readme-ov-file#sam-2-checkpoints) to grab a pretrained model (you’ll need this for the 2D mask stage). Place it somewhere you can point to it in your configs.



## Runnig Our Semantic-Aware UV Parameterization Pipeline
The following command does the steps below:

1- Semantically segment parts of a 3D object

2- Perform the training stage of our UV learning pipeline

3- Perform the inference stage of our UV learning pipeline

4- Aggregate all the UV parameterization for each segmented 3D part and merge them into one big unified atlas image

For the SAMesh as the base 3D segmentation method, run the following command:
```bash
python semantic_aware_uv_learning.py --seg_out_dir "../checkpoints/Rabbit" --glb_dir "../Data/Segmented_Meshes/SAMesh/Rabbit/Rabbit_segmented.glb"
```
For the Shape Diameter Function (ShDF) as the base 3D segmentation method, run the following command:
```bash
python semantic_aware_uv_learning.py --seg_out_dir "../checkpoints/Rabbit" --glb_dir "../Data/Segmented_Meshes/ShDF/Rabbit/Rabbit_segmented.glb"
```



## Runnig Our Visibility-Aware UV Parameterization Pipeline
The following command does the steps below:

1- Perform the training stage of our UV learning pipeline

2- Perform the inference stage of our UV learning pipeline


```bash
python visibility_aware_uv_learning.py --obj_dir "../../Data/Rabbit.obj" --export_dir "../expt"
```


### Training Alone
```bash
python train.py ../Data/Rabbit.obj ../expt 10000 True False
```

### Testing Alone
```bash
python test.py ../Data/Rabbit.obj ../expt/Rabbit_1 True False
```

## Tips for any potential issues during or after the installation process

1- When installing the CD or EMD packages, you might face an error saying something like "mismatch between the version of OS CUDA and Pytorch CUDA". This is because of the fact that the command ```python setup.py install``` uses the OS CUDA and if its version is not the same as the Pytorch CUDA version, it returns such an error. To resolve this, you need to remove and re-install either your OS CUDA or Pytorch to have the consistent and same version for both CUDAs. We recommend to keep the OS CUDA and re-install Pytorch accordingly. Also, after the installing the OS CUDA don't forget to add the address to the NVIDIA GPU Computing Toolkit as the ```CUDA_PATH``` to your Environment Variables. For Windows 10 or 11 users, the way you do is to search for Environment Variable in the taskbar, and then under the System Variables window, you need to add variable ```CUDA_PATH``` with the value ```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8``` (or something like this).

2- For users on Windows 10 or 11, you should install Microsoft Visual Studio 2022 (with the option "Desktop development with C++" checked) to provide a C++ compiler for the different packages (e.g. CD and EMD) to be properly installed. This is because these packages run the compiler existing in a directory of Microsoft Visual Studio on Windows.


3- If you get this error ```OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.```, it occurs when a Python program or library (like PyTorch) tries to access CUDA for GPU computation but cannot find the CUDA installation because the ```CUDA_HOME``` environment variable is not set. You need to manually set the ```CUDA_HOME``` variable to point to the root directory of your CUDA installation. Here is how to resolve it:

a) Typically, CUDA is installed in ```/usr/local/cuda``` on Linux or ```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X``` on Windows (where vX.X is your version).

For example: On Linux: ``` /usr/local/cuda ``` and on Windows:  ``` C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8 ```

b) Set Environment Variable: 

Linux/macOS: Open your terminal and add the following lines to your ``` ~/.bashrc ```:
```
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
```
Then run:
```
source ~/.bashrc
```

Windows: First install the CUDA toolkit from [here](https://developer.nvidia.com/cuda-downloads?target_os=Windows). Then, open the Start menu, search for "Environment Variables."
In "System Properties," click "Environment Variables." Under "System variables," click "New," and add:
    Variable Name: ```CUDA_HOME```
    Variable Value: ```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8``` (adjust based on your version).

c) Restart your Terminal or IDE: After setting the environment variable, restart the terminal or the IDE you're using.



# Citation
We appreciate your interest in our research. If you find this study useful in your work, we kindly ask that you cite it using the following format.
```

@article{zamani2025unsupervised,    
  title={Unsupervised Representation Learning for 3D Mesh Parameterization with Semantic and Visibility Objectives},
  author={Zamani, AmirHossein and Roy, Bruno and Rampini, Arianna},
  journal={arXiv preprint arXiv:2509.25094},
  year={2025},
  url={https://ahhhz975.github.io/Automatic3DMeshParameterization/}
}
```
