
## Multi-View Gamba Model for Unified 3D Generation

This is the official implementation of *MVGamba: Unify 3D Content Generation as State Space Sequence Modeling* (NeurIPS 2024).


https://github.com/user-attachments/assets/f71f0349-0e11-4c31-94a7-b7a57275ccb3


### [Arxiv](https://arxiv.org/abs/2406.06367) | [Weights]


- [x] Release MVGamba-3DGS training and inference code.
- [x] Release MVGamba-2DGS training and inference code, please refer to the surfel branch.
- [ ] Release pretrained checkpoints.

### Install

```bash
# xformers is required! please refer to https://github.com/facebookresearch/xformers for details.
# for example, we use torch 2.1.0 + cuda 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install causal-conv1d==1.2.0 mamba-ssm
git clone --recursive git@github.com:SkyworkAI/MVGamba.git
# a modified 3D gaussian splatting (+ depth, alpha rendering)
pip install ./submodules/diff-gaussian-rasterization
# 2D gaussian surfel 
pip install ./submodules/diff-surfel-rasterization

# for mesh extraction
pip install git+https://github.com/NVlabs/nvdiffrast

# other dependencies
pip install -r requirements.txt
```

### Training

**NOTE**: Due to company property restrictions, we are unable to provide the full training data used for training MVGamba. Here, we alternatively follow the practice by @kiui(https://github.com/ashawkey) and provide the ~80K subset of **[Objaverse](https://objaverse.allenai.org/objaverse-1.0)** and in **[objaverse_filter](https://github.com/ashawkey/objaverse_filter)**. please check and modify the [dataset](./core/provider_ikun.py) implementation!

```bash
# training (use single node training)
accelerate launch --config_file acc_configs/gpu8.yaml main.py mvgamba --workspace /root/Results/workspace_mvgamba

# training (use multi-node training)
bash ./scripts/mvgamba_dist.sh
```

### Inference

 For [MVDream](https://github.com/bytedance/MVDream) and [ImageDream](https://github.com/bytedance/ImageDream), we use a [diffusers implementation](https://github.com/ashawkey/mvdream_diffusers). The required model weights are downloaded automatically.

```bash 
bash ./scripts/mvgamba_infer.sh
```
### Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [LGM](https://github.com/3DTopia/LGM)
- [OpenLRM](https://github.com/3DTopia/OpenLRM)
- [Gamba](https://github.com/SkyworkAI/Gamba)
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)
- [tyro](https://github.com/brentyi/tyro)


### Citation

```bibtex

@article{yi2024mvgamba,
  title={MVGamba: Unify 3D Content Generation as State Space Sequence Modeling},
  author={Yi, Xuanyu and Wu, Zike and Shen, Qiuhong and Xu, Qingshan and Zhou, Pan and Lim, Joo-Hwee and Yan, Shuicheng and Wang, Xinchao and Zhang, Hanwang},
  journal={arXiv preprint arXiv:2406.06367},
  year={2024}
}

```
Please also check our another project for ultra fast single view 3D generation [Gamba](https://arxiv.org/abs/2403.18795). The code and pretrained weights have been released in https://github.com/SkyworkAI/Gamba.


```bibtex

@article{shen2024gamba,
  title={Gamba: Marry gaussian splatting with mamba for single view 3d reconstruction},
  author={Shen, Qiuhong and Wu, Zike and Yi, Xuanyu and Zhou, Pan and Zhang, Hanwang and Yan, Shuicheng and Wang, Xinchao},
  journal={arXiv preprint arXiv:2403.18795},
  year={2024}
}

```
