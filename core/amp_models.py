import math

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import pdb

from core.typings import *
from core.options import Options
from kiui.lpips import LPIPS

class LPIPS_Loss(nn.Module):
    def __init__(self, opt: Options):
        super(LPIPS_Loss, self).__init__()
        self.opt = opt
        self.lpips_loss = LPIPS(net='vgg')
        self.lpips_loss.requires_grad_(False)
    
    def forward(self):
        pass
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward_with_amp(self, gt_images, pred_images):
        loss_lpips = self.lpips_loss(
            # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
            # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
            # downsampled to at most 256 to reduce memory cost
            F.interpolate(gt_images.to(torch.bfloat16).view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
            F.interpolate(pred_images.to(torch.bfloat16).view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
        ).mean()
        return loss_lpips

class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise ValueError("Fp32LayerNorm is deprecated for debugging, use torch.nn.LayerNorm instead")

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        assert output.dtype == torch.float32
        return output.type_as(input)