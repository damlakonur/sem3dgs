# sem3dgs

Semantic 3D Gaussian Splatting project with video rendering capabilities.

## Environment Setup

### 1. Clone with submodules
```bash
git clone https://github.com/damlakonur/sem3dgs.git --recursive
cd sem3dgs
```

If already cloned without `--recursive`:
```bash
git submodule update --init --recursive
```

### 2. Create conda environment
```bash
conda env create -f environment.yml
conda activate sem3dgs
```

### 3. Install Gaussian Splatting CUDA extensions
```bash
conda run -n sem3dgs bash -c "export CUDA_HOME=\$CONDA_PREFIX && pip install gaussian-splatting/submodules/diff-gaussian-rasterization gaussian-splatting/submodules/simple-knn gaussian-splatting/submodules/fused-ssim"
```

## Usage

### Training
```bash
conda activate sem3dgs
cd gaussian-splatting

# Basic training
python train.py -s ../data/bicycle -m output/bicycle


### Render Video
```bash
python src/scripts/render_video.py \
  -m gaussian-splatting/output/<model_name> \
  -o videos/output.mp4 \
  --duration 10 \
  --fps 30 \
  --radius 3.0
```

## Requirements
- CUDA 11.6
- Python 3.7
- PyTorch 1.12.1
