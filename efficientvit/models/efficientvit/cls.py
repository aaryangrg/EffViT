# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

from efficientvit.models.efficientvit.flexible_backbone import FlexibleEfficientViTBackbone
from efficientvit.models.nn.flexible_ops import FlexibleConvLayer
import torch
import torch.nn as nn

from efficientvit.models.efficientvit.backbone import EfficientViTBackbone, EfficientViTLargeBackbone
from efficientvit.models.nn import ConvLayer, LinearLayer, OpSequential
from efficientvit.models.utils import build_kwargs_from_config

__all__ = [
    "EfficientViTCls",
    ######################
    "efficientvit_cls_b0",
    "efficientvit_cls_b1",
    "efficientvit_cls_b2",
    "efficientvit_cls_b3",
    ######################
    "efficientvit_cls_l1",
    "efficientvit_cls_l2",
    "efficientvit_cls_l3",
    ###################### - Width Adjusted Models
    "efficientvit_width_adjusted_cls_b0",
    "efficientvit_width_adjusted_cls_b1",
    "efficientvit_width_adjusted_cls_b3",
    "efficientvit_width_adjusted_cls_l3",
    ###################### - Flexible Models
    "flexible_efficientvit_cls_b1",
    "flexible_efficientvit_cls_b3"
]


class ClsHead(OpSequential):
    def __init__(
        self,
        in_channels: int,
        width_list: list[int],
        n_classes=1000,
        dropout=0.0,
        norm="bn2d",
        act_func="hswish",
        fid="stage_final",
    ):
        ops = [
            ConvLayer(in_channels, width_list[0], 1, norm=norm, act_func=act_func),
            nn.AdaptiveAvgPool2d(output_size=1),
            LinearLayer(width_list[0], width_list[1], False, norm="ln", act_func=act_func),
            LinearLayer(width_list[1], n_classes, True, dropout, None, None),
        ]
        super().__init__(ops)

        self.fid = fid

    def forward(self, feed_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        x = feed_dict[self.fid]
        return OpSequential.forward(self, x)


class FlexibleClsHead(OpSequential):
    def __init__(
        self,
        in_channels: int,
        width_list: list[int],
        n_classes=1000,
        dropout=0.0,
        norm="bn2d",
        act_func="hswish",
        fid="stage_final",
    ):
        # Could set to True, False (no changes to the linear layers)
        ops = [
            FlexibleConvLayer(in_channels, width_list[0], 1, norm=norm, act_func=act_func, flex = [True, False]),
            nn.AdaptiveAvgPool2d(output_size=1),
            LinearLayer(width_list[0], width_list[1], False, norm="ln", act_func=act_func),
            LinearLayer(width_list[1], n_classes, True, dropout, None, None),
        ]
        super().__init__(ops)

        self.fid = fid

    def forward(self, feed_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        x = feed_dict[self.fid]
        return OpSequential.forward(self, x)

class EfficientViTCls(nn.Module):
    def __init__(self, backbone: EfficientViTBackbone or EfficientViTLargeBackbone or FlexibleEfficientViTBackbone, head: ClsHead or FlexibleClsHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)
        output = self.head(feed_dict)
        return output


def efficientvit_cls_b0(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0

    backbone = efficientvit_backbone_b0(**kwargs)

    head = ClsHead(
        in_channels=128,
        width_list=[1024, 1280],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model

#### WIDTH ADJUSTED COMPLETE B0 MODEL #####
def efficientvit_width_adjusted_cls_b0(width_multiplier = 1, height_multiplier = 1, **kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_modified_backbone_b0

    backbone = efficientvit_modified_backbone_b0(width_multiplier = width_multiplier,height_multiplier = height_multiplier,**kwargs)
    default_out_channels = 128
    new_out_channels = int(default_out_channels * width_multiplier)
    head = ClsHead(
        in_channels=new_out_channels,
        width_list=[1024, 1280],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model

def efficientvit_cls_b1(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b1

    backbone = efficientvit_backbone_b1(**kwargs)

    head = ClsHead(
        in_channels=256,
        width_list=[1536, 1600],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model

#### WIDTH ADJUSTED COMPLETE B1 MODEL #####
def efficientvit_width_adjusted_cls_b1(width_multiplier = 1, height_multiplier = 1, **kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_modified_backbone_b1

    backbone = efficientvit_modified_backbone_b1(width_multiplier = width_multiplier,height_multiplier = height_multiplier,**kwargs)
    default_out_channels = 256
    new_out_channels = int(default_out_channels * width_multiplier)
    head = ClsHead(
        in_channels=new_out_channels,
        width_list=[1536, 1600],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


#### WIDTH ADJUSTED COMPLETE B3 MODEL #####
def efficientvit_width_adjusted_cls_b3(width_multiplier = 1, height_multiplier = 1, **kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_modified_backbone_b3

    backbone = efficientvit_modified_backbone_b3(width_multiplier = width_multiplier,height_multiplier = height_multiplier,**kwargs)
    default_out_channels = 512
    new_out_channels = int(default_out_channels * width_multiplier)
    head = ClsHead(
        in_channels=new_out_channels,
        width_list=[2304, 2560],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model

def efficientvit_cls_b2(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2

    backbone = efficientvit_backbone_b2(**kwargs)

    head = ClsHead(
        in_channels=384,
        width_list=[2304, 2560],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_b3(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b3

    backbone = efficientvit_backbone_b3(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[2304, 2560],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_l1(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l1

    backbone = efficientvit_backbone_l1(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[3072, 3200],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_l2(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l2

    backbone = efficientvit_backbone_l2(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[3072, 3200],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_l3(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l3

    backbone = efficientvit_backbone_l3(**kwargs)

    head = ClsHead(
        in_channels=1024,
        width_list=[6144, 6400],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model

#### WIDTH ADJUSTED COMPLETE L3 MODEL #####
def efficientvit_width_adjusted_cls_l3(width_multiplier = 1, height_multiplier = 1, **kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_modified_backbone_l3

    backbone = efficientvit_modified_backbone_l3(width_multiplier = width_multiplier,height_multiplier = height_multiplier,**kwargs)
    default_out_channels = 1024
    new_out_channels = int(default_out_channels * width_multiplier)
    head = ClsHead(
        in_channels=new_out_channels,
        width_list=[6144, 6400],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model

#### Flexible Cls Models #####

def flexible_efficientvit_cls_b1(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.flexible_backbone import flexible_efficientvit_backbone_b1

    backbone = flexible_efficientvit_backbone_b1(**kwargs)

    head = FlexibleClsHead(
        in_channels=256,
        width_list=[1536, 1600],
        **build_kwargs_from_config(kwargs, FlexibleClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def flexible_efficientvit_cls_b3(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.flexible_backbone import flexible_efficientvit_backbone_b3

    backbone = flexible_efficientvit_backbone_b3(**kwargs)
    head = FlexibleClsHead(
        in_channels=512,
        width_list=[2304, 2560],
        **build_kwargs_from_config(kwargs, FlexibleClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model

