This repo is the derived implementation of InstantMesh. For original repo, visit https://github.com/TencentARC/InstantMesh
## Dependencies and Installation
We recommend using Python>=3.10, PyTorch>=2.1.0, and CUDA>=12.1.

conda create --name instantmesh python=3.10
conda activate instantmesh
pip install -U pip

# Ensure Ninja is installed
conda install Ninja

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7

# Install Triton 
pip install triton

# Install other requirements
pip install -r requirements.txt