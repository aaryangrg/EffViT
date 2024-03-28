import torch
import torch.nn as nn

from efficientvit.models.nn import (
    IdentityLayer,
    OpSequential,
    ResidualBlock,
)

from efficientvit.models.nn.flexible_ops import (
    FlexibleEfficientViTBlock,
    FlexibleConvLayer,
    FlexibleDSConv,
    FlexibleMBConv,
)
from efficientvit.models.utils import build_kwargs_from_config
from timm.models.layers import to_2tuple


__all__ = [
    "FlexibleGDINOBackbone",
    "FlexibleGDINOBackboneRectified",
    "flexible_efficientvit_backbone_swin_t_224_1k",
    "flexible_efficientvit_backbone_swin_b_384_22k",
    "flexible_efficientvit_backbone_swin_t_224_1k_rectified"
]

# From Grounding Dino / Open Grounding Dino Implementation
class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=3, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = nn.functional.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nn.functional.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x # B x C x H x W

class FlexibleGDINOBackbone(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
    ) -> None:
        super().__init__()

        self.width_list = []

        # No normalization and fixed internal convolution kernel size
        # Why is patch size = 2?
        self.patch_embed = PatchEmbed(patch_size = 2, in_chans=3, embed_dim=3)
        # input stem 
        self.input_stem = [
            FlexibleConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=1,
                norm=norm,
                act_func=act_func,
                flex = [False, True]
            )
        ]
        for _ in range(depth_list[0]):
            if _ == depth_list[0]-1:
                flex_vals = [True, False]
                residual_func = None
            else :
                flex_vals = [True, True]
                residual_func = IdentityLayer()
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
                flex = flex_vals
            )
            self.input_stem.append(ResidualBlock(block, residual_func))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # Possible index = 0,1
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                if i == 0 :
                    flex_vals = [False, True]
                    residual_func = None
                elif i == d-1 :
                    flex_vals = [True, False]
                    residual_func = None
                else :
                    flex_vals = [True, True]
                    residual_func = IdentityLayer()
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                    flex = flex_vals
                )
                stage.append(ResidualBlock(block, residual_func))
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        #  Possible Index 2,3
        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
                flex = [False, True]
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w
            for _ in range(d):
                if _ == d-1 :
                    flex_out = False
                else :
                    flex_out = True
                stage.append(
                    FlexibleEfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                        flex_out = flex_out,
                        disable_residual= not flex_out
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
        flex = [True, True]
    ) -> nn.Module:
        if expand_ratio == 1:
            block = FlexibleDSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
                flex = flex
            )
        else:
            block = FlexibleMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
                flex = flex
            )
        return block

    # Return outputs for stage 0-3
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        x = self.patch_embed(x)
        x = self.input_stem(x)
        for stage_id, stage in enumerate(self.stages, 0):
            x = stage(x)
            outs.append(x)
        return outs

    # def forward_raw(self, x: gdino_util_misc.NestedTensor):
    #     images = x.tensors
    #     masks = x.mask
    #     outs = []
    #     images = self.patch_embed(images)
    #     images = self.input_stem(images)
    #     for stage_id, stage in enumerate(self.stages, 1):
    #         images = stage(images)
    #         outs.append(images)
    #     # Images processed
            
    #     # Reshaping masks
    #     outs_dict = {}
    #     for idx, out_i in enumerate(outs):
    #         m = masks
    #         assert m is not None
    #         mask = torch.nn.functional.interpolate(m[None].float(), size=out_i.shape[-2:]).to(torch.bool)[0]
    #         outs_dict[idx] = gdino_util_misc.NestedTensor(out_i, mask)

    #     return outs_dict

def flexible_efficientvit_backbone_swin_t_224_1k(**kwargs) -> FlexibleGDINOBackbone:
    backbone = FlexibleGDINOBackbone(
        width_list = [32, 64, 192, 384, 768],
        depth_list = [1, 2, 3, 4, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, FlexibleGDINOBackbone),
    )
    return backbone


# Modify according to actual channel dimensions
def flexible_efficientvit_backbone_swin_b_384_22k(**kwargs) -> FlexibleGDINOBackbone:
    backbone = FlexibleGDINOBackbone(
        width_list = [32, 64, 192, 384, 768],
        depth_list = [1, 2, 3, 4, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, FlexibleGDINOBackbone),
    )
    return backbone

class FlexibleGDINOBackboneRectified(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
    ) -> None:
        # in:
        #   torch.Size([2, 3, 1024, 1024])
        # outs:
        #  [torch.Size([2, 96, 256, 256]), 
        #  torch.Size([2, 192, 128, 128]),
        #  torch.Size([2, 384, 64, 64]), 
        #  torch.Size([2, 768, 32, 32])]
        super().__init__()

        self.width_list = []
        
        # Input : 2 x 3 x 1024 x 1024 --> Out : 2 x 96 x 256 x 256
        self.patch_embed = PatchEmbed(patch_size = 4, in_chans=3, embed_dim=width_list[0], norm_layer=nn.LayerNorm)

        # Input : 2 x 96 x 256 x 256 --> Out : 2 x 96 x 256 x 256
        self.input_stem = [ 
            FlexibleConvLayer(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                norm=norm,
                act_func=act_func,
                flex = [False, True]
            )
        ]

        # Input : 2 x 96 x 256 x 256 --> Out : 2 x 96 x 256 x 256
        for _ in range(depth_list[0]):
            if _ == depth_list[0]-1:
                flex_vals = [True, False]
                residual_func = None
            else :
                flex_vals = [True, True]
                residual_func = IdentityLayer()
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
                flex = flex_vals
            )
            self.input_stem.append(ResidualBlock(block, residual_func))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)
        
        # Input : 2 x 96 x 256 x 256 --> Out : 2 x 96 x 256 x 256 --> 0
        # Input : 2 x 96 x 256 x 256 --> Out : 2 x 192 x 128 x 128 --> 1
        self.stages = []
        first_block = True
        for w, d in zip(width_list[1:3], depth_list[1:3]): 
            stage = []
            for i in range(d):
                if i == 0 :
                    flex_vals = [False, True]
                    residual_func = None
                elif i == d-1 :
                    flex_vals = [True, False]
                    residual_func = None
                else :
                    flex_vals = [True, True]
                    residual_func = IdentityLayer()
                stride = 2 if i == 0 else 1
                if first_block : 
                    stride = 1
                    first_block = False
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                    flex = flex_vals
                )
                stage.append(ResidualBlock(block, residual_func))
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        # Input : 2 x 192 x 128 x 128 --> Out : 2 x 384 x 64 x 64 --> 2
        # Input : 2 x 384 x 64 x 64 --> Out : 2 x 768 x 32 x 32 --> 3
        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
                flex = [False, True]
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w
            for _ in range(d):
                if _ == d-1 :
                    flex_out = False
                else :
                    flex_out = True
                stage.append(
                    FlexibleEfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                        flex_out = flex_out,
                        disable_residual= not flex_out
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

        output_layer_norms = []
        # Initalizing separate layer norms for each output
        for i_layer in self.width_list[2:]: # For Last 3 outputs (outputs which are actually used) 
            layer = nn.LayerNorm(i_layer)
            output_layer_norms.append(layer)
        self.output_layer_norms = nn.ModuleList(output_layer_norms)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
        flex = [True, True]
    ) -> nn.Module:
        if expand_ratio == 1:
            block = FlexibleDSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
                flex = flex
            )
        else:
            block = FlexibleMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
                flex = flex
            )
        return block

    # Return outputs for stage 0-3
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        x = self.patch_embed(x)
        x = self.input_stem(x)
        for stage_id, stage in enumerate(self.stages, 0):
            x = stage(x)
            if stage_id > 0 :
                # Make B x (H * W) X C, apply layer normalization, convert back to original shape (no changes made directly to x)
                B, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
                y = self.output_layer_norms[stage_id-1](x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)).view(B, H, W, C).permute(0, 3, 1, 2)
                outs.append(y)
            else :
                outs.append(x) # Not used currently (dimension 96)
        return outs

# def flexible_efficientvit_backbone_swin_t_224_1k_rectified(**kwargs) -> FlexibleGDINOBackbone:
#     backbone = FlexibleGDINOBackboneRectified(
#         width_list = [96, 96, 192, 384, 768],
#         depth_list = [1, 2, 2, 3, 3],
#         dim=16,
#         **build_kwargs_from_config(kwargs, FlexibleGDINOBackbone),
#     )
#     return backbone

def flexible_efficientvit_backbone_swin_t_224_1k_rectified(**kwargs) -> FlexibleGDINOBackbone:
    backbone = FlexibleGDINOBackboneRectified(
        width_list = [96, 96, 192, 384, 768],
        depth_list = [1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, FlexibleGDINOBackbone),
    )
    return backbone