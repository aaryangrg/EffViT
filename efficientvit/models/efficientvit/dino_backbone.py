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
from timm.model.layers import to_2tuple

__all__ = [
    "FlexibleGDINOBackbone",
    "flexible_efficientvit_backbone_swin_t_224_1k"
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
        self.patch_embed = PatchEmbed(patch_size = 4, in_chans=3, embed_dim=3)
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
            else :
                flex_vals = [True, True]
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
                flex = flex_vals
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
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
                elif i == d-1 :
                    flex_vals = [True, False]
                else :
                    flex_vals = [True, True]
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
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
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
                        flex_out = flex_out
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
        for stage_id, stage in enumerate(self.stages, 1):
            x = stage(x)
            outs.append(x)
        return outs



def flexible_efficientvit_backbone_swin_t_224_1k(**kwargs) -> FlexibleGDINOBackbone:
    # in:
    #   torch.Size([2, 3, 1024, 1024])
    # outs:
    #  [torch.Size([2, 96, 256, 256]), 
    #  torch.Size([2, 192, 128, 128]),
    #  torch.Size([2, 384, 64, 64]), 
    #  torch.Size([2, 768, 32, 32])]
    backbone = FlexibleGDINOBackbone(
        width_list=[32, 96, 192, 384, 768],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, FlexibleGDINOBackbone),
    )
    return backbone