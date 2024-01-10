from efficientvit.models.nn import IdentityLayer, ResidualBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from efficientvit.models.nn.act import build_act
from efficientvit.models.nn.norm import build_norm, build_flexible_norm
from efficientvit.models.utils import get_same_padding, list_sum, resize, val2list, val2tuple

__all__ = [
    "FlexibleConvLayer",
    "FlexibleDSConv",
    "FlexibleMBConv",
    "FlexibleLiteMLA",
    "FlexibleBatchNorm2d"
    "FlexibleEfficientViTBlock",
]

# Take from configs / flags
WIDTH_LIST = [0.25, 0.50, 0.75, 1]

def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class FlexibleConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
        # Input Channels , Output Channels
        flex = [True, True]
    ):
        super(FlexibleConvLayer, self).__init__()

        # For same W x H output
        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None

        # Initializing defaults for later reference
        self.in_channels_basic = in_channels
        self.out_channels_basic = out_channels
        self.width_mult = None
        self.use_bias = use_bias
        self.groups_desc = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.flex = flex
        self.dilation = dilation

        # Full width conv weights initialized
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_flexible_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)
    
    def forward(self, input):
        if self.dropout is not None :
            input = self.dropout(input)
        in_channels = self.in_channels_basic
        out_channels = self.out_channels_basic
        # Removed Scaling by ratio
        if self.flex[0] :
            in_channels = int(make_divisible(self.in_channels_basic * self.width_mult))
        if self.flex[1] : 
            out_channels = int(make_divisible(self.out_channels_basic * self.width_mult))
        # Slicing default (max width) conv layer weights
        weight = self.conv.weight[:out_channels, :in_channels, :, :]
        if self.use_bias :
            bias = self.conv.bias[:out_channels]
        else:
            bias = self.conv.bias
        out = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups_desc if self.groups_desc == 1 else in_channels)
        
        # What is this exactly?
        # if getattr(FLAGS, 'conv_averaged', False):
            # out = out * (max(self.in_channels_list)/self.in_channels)
        if self.norm :
            out = self.norm(out)
        if self.act :
            out = self.act(out)

        print(out.shape)
        return out 
    
# Adapted from USBatchNorm2D
class FlexibleBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features):
        num_features_max = num_features

        # Using default BatchNorm2d configs --> affine and track not specified in configs
        super(FlexibleBatchNorm2d, self).__init__(num_features_max)
        self.num_features_basic = num_features

        # Different layer for each pre-determined width
        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(i, affine=False)
             for i in [
                     int(make_divisible(
                         num_features * width_mult))
                     for width_mult in WIDTH_LIST]
             ]
        )
        # self.ratio = ratio
        self.width_mult = None
        self.ignore_model_profiling = True

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        c = int(make_divisible(self.num_features_basic * self.width_mult))
        if self.width_mult in WIDTH_LIST:
            idx = WIDTH_LIST.index(self.width_mult)
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        return y
    

class FlexibleDSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(FlexibleDSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = FlexibleConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
            flex = [True, True]
        )
        self.point_conv = FlexibleConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
            flex = [True, True]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x
    

class FlexibleMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super(FlexibleMBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = FlexibleConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
            flex = [True, True]
        )
        self.depth_conv = FlexibleConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
            flex = [True, True]
        )
        self.point_conv = FlexibleConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
            flex = [True, True]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

# Currently not configured to reduce MLA dimensionality
class FlexibleLiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super(FlexibleLiteMLA, self).__init__()


        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = FlexibleConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            flex = [True, False]
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = FlexibleConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            flex = [False, True]
        )

    @autocast(enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )

        # Fixed hidden dimensionality tokens for attention
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (
            qkv[..., 0 : self.dim],
            qkv[..., self.dim : 2 * self.dim],
            qkv[..., 2 * self.dim :],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        return out
    
    # Reduced in-channels + Reduced out-channels (internal attention hidden dimension unchanged)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        out = self.relu_linear_att(multi_scale_qkv)
        out = self.proj(out)

        return out

    @staticmethod
    def configure_litemla(model: nn.Module, **kwargs) -> None:
        eps = kwargs.get("eps", None)
        for m in model.modules():
            if isinstance(m, LiteMLA):
                if eps is not None:
                    m.eps = eps


class FlexibleEfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        norm="bn2d",
        act_func="hswish",
    ):
        super(FlexibleEfficientViTBlock, self).__init__()
        self.context_module = ResidualBlock(
            FlexibleLiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
            ),
            IdentityLayer(),
        )
        local_module = FlexibleMBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x
    
# Convert to flexible model -- Linear layers would take different 'C' inputs
class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout=0,
        norm=None,
        act_func=None,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x