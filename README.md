# MAPS
Multi-Agent Lidar 3D Point Cloud Segmentation

## Setup

1. Create an environment:
```bash
conda env create -f environment.yaml
```
Or
```bash
conda env create -f cpu_environment.yaml
```
2. Compile bitsandbytes from source (If GPU):
```bash
git clone https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install -e .
```
3. Compule dgl from source (IF GPU):
```bash
# If you have installed dgl-cudaXX.X package, please uninstall it first.
conda install -c dglteam/label/th24_cu118 dgl
```
