# 1. Create a new conda environment
```
conda create -y -n mvgamba python=3.10
conda activate mvgamba
```

# 2. Upgrade base tools
```
pip install --upgrade pip wheel setuptools
```

# 3. Install PyTorch/cu121
```
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

# 4. Install xFormers matching torch 2.5.1+cu121
```
pip install xformers==0.0.28
```

# 5. Other dependencies
```
pip install --no-build-isolation --no-cache-dir \
  "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.2.0.post2"
```

# Install mamba-ssm with no-build-isolation and no-deps
```
pip install --no-build-isolation --no-deps mamba-ssm==1.2.2
```

# 6. Clone the repository (use HTTPS to avoid SSH key issues)
```
git clone --recursive https://github.com/SkyworkAI/MVGamba.git
cd MVGamba
git submodule sync --recursive
git submodule update --init --recursive
```

# 7. Build rendering submodules
# 3D Gaussian splatting (+ depth, alpha rendering)
```
pip install ./submodules/diff-gaussian-rasterization
```

# 2D Gaussian surfel
```
pip install ./submodules/diff-surfel-rasterization
```

# 8. Install nvdiffrast (for mesh extraction)
```
pip install git+https://github.com/NVlabs/nvdiffrast
```


# 9. Install remaining dependencies
# (skip deps here to avoid overwriting locked versions above)
```
pip install -r requirements.txt --no-deps
```

# 10. Run with accelerate
```
accelerate launch --config_file acc_configs/gpu1.yaml main.py mvgamba --workspace results/mvgamba_debug
```
