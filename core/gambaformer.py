import math 

import torch
import torch.nn as nn

from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from timm.models.layers import DropPath, to_2tuple
# from typings import *
from torch.utils.checkpoint import checkpoint

# Basic types
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from torch import Tensor

class ReorderTokensFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, index_order):
        """
        Reorders the token_num dimension of the input tensor according to the index_order.

        Args:
        - input_tensor: A tensor of shape (batch_size, token_num, feature_dim).
        - index_order: A tensor of shape (token_num,) indicating the desired order.

        Returns:
        - A tensor with the same shape as input_tensor but with the token_num dimension reordered.
        """
        
        reordered = torch.index_select(input_tensor, 1, index_order)
        inverse_index = torch.argsort(index_order)
        ctx.save_for_backward(index_order, inverse_index)
        return reordered, inverse_index

    @staticmethod
    def backward(ctx, grad_output, grad_inv):
        """
        Propagates gradients back to the input tensor before reordering.

        Args:
        - grad_output: Gradients tensor with the same shape as the output of the forward method.

        Returns:
        - Gradients with respect to the input_tensor.
        - None for index_order, as it does not require gradients.
        """
        _, inverse_index = ctx.saved_tensors        
        grad_input = torch.index_select(grad_output, 1, inverse_index)
        
        return grad_input, None

def reorder_tokens(input_tensor, index_order):
    return ReorderTokensFunction.apply(input_tensor, index_order)


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        
        # drop path 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class ConditionModulationBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition and another modulation vector applied to sub-blocks.
    """
    def __init__(self, inner_dim: int, 
                 drop_path_rate: float = 0., layer_idx=0,
                 residual_in_fp32=False,
                 rms_norm=False,
                 fused_add_norm=False,
                 grad_checkpointing=False):
        super().__init__()
        self.grad_checkpointing = grad_checkpointing
        
        self.mamba_block = create_block(d_model=inner_dim, 
                                        ssm_cfg=None,
                                        norm_epsilon=1e-5,
                                        rms_norm=rms_norm,
                                        residual_in_fp32=residual_in_fp32,
                                        fused_add_norm=fused_add_norm,
                                        layer_idx=layer_idx,
                                        drop_path=drop_path_rate,
                                        )
    
    def forward(self, hidden_states, residual, inference_params=None):
        if self.grad_checkpointing and hidden_states.requires_grad:
            hidden_states, residual = checkpoint(self.mamba_block, hidden_states, residual, inference_params)
        else:
            hidden_states, residual = self.mamba_block(hidden_states, residual, inference_params)
        return hidden_states, residual

def rotate_blocks_pytorch_for_loop(original_tensor, block_size):
    num_blocks = original_tensor.size(0) // block_size
    configurations = []
    for rotation in range(num_blocks):
        start_index = rotation * block_size % original_tensor.size(0)
        rotated_tensor = torch.cat((original_tensor[start_index:], original_tensor[:start_index]), dim=0)
        configurations.append(rotated_tensor)
    return torch.stack(configurations, dim=0)

class GambaFormer(nn.Module):
    def __init__(self, 
                 inner_dim: int, num_layers: int, 
                 gs_num:int, 
                 drop_path_rate: float = 0.1,
                 fused_add_norm=True,
                 rms_norm=True,
                 norm_epsilon=1e-5,
                 residual_in_fp32=True,
                 initializer_cfg=None,
                 grad_checkpointing=False,
                 use_pos_embed=True):
        super().__init__()        
        self.use_pos_embed = use_pos_embed
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(gs_num, inner_dim) * (1. / inner_dim) ** 0.5)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList([
            ConditionModulationBlock(
                inner_dim=inner_dim, drop_path_rate=inter_dpr[i],
                layer_idx=i,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                rms_norm=rms_norm,
                grad_checkpointing=grad_checkpointing,
            )
            for i in range(num_layers)
        ])

        factory_kwargs = {"device": None, "dtype": None}
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            inner_dim, eps=norm_epsilon, **factory_kwargs
        )

        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32

        self.apply(
            partial(
                _init_weights,
                n_layer=num_layers,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
    
    def forward(self, img_cond,inference_params=None):
        N, L, _ = img_cond.shape
        if self.use_pos_embed:
            hidden_states, residual = img_cond + self.pos_embed.repeat(N, 1, 1), None
        else:
            hidden_states, residual = img_cond, None
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(hidden_states, residual, inference_params)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )    
        
        return hidden_states
        
if __name__ == "__main__":
    model = GambaFormer(inner_dim=512, 
                  image_feat_dim=768, 
                  mod_embed_dim=128, 
                  num_layers=8, 
                  gs_num=16384, 
                  drop_path_rate=0.1,
                  rorder=False).cuda().train()
    import pdb 
    img_cond = torch.randn(1, 1024, 768).cuda()
    mod = torch.randn(1, 128).cuda()
    output = model(img_cond, mod)
    print(output)
