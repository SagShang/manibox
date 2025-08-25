# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""主干网络模块 - Backbone modules

本模块定义了用于机器人视觉编码的主干网络架构。
主要包含ResNet系列主干网络和深度图像处理网络，
用于从多模态图像输入中提取视觉特征。

This module defines backbone network architectures for robot vision encoding.
Includes ResNet-series backbones and depth image processing networks,
used for extracting visual features from multi-modal image inputs.

主要组件 / Main Components:
- FrozenBatchNorm2d: 冻结批正化层，参数不可训练 / Frozen batch normalization layer with non-trainable parameters
- BackboneBase: 主干网络基类 / Base class for backbone networks
- Backbone: ResNet主干网络实现 / ResNet backbone implementation
- Joiner: 组合主干网络和位置编码 / Combines backbone and position encoding
- DepthNet: 深度图像处理网络 / Depth image processing network
- RestNetBasicBlock/RestNetDownBlock: ResNet基本构建块 / ResNet basic building blocks

特点 / Features:
- 支持预训练ResNet模型（ResNet18/34/50/101）/ Supports pretrained ResNet models (ResNet18/34/50/101)
- 可配置的中间层输出用于特征融合 / Configurable intermediate layer outputs for feature fusion
- 专门的深度图像编码器 / Specialized depth image encoder
- 位置编码集成用于Transformer / Position encoding integration for Transformer
"""
# 标准库导入 / Standard library imports
from collections import OrderedDict  # 有序字典，用于保持层顺序 / Ordered dictionary for maintaining layer order
from typing import Dict, List  # 类型提示 / Type hints

# PyTorch相关导入 / PyTorch related imports
import torch  # PyTorch核心库 / PyTorch core library
import torch.nn.functional as F  # 神经网络函数 / Neural network functions
import torchvision  # 计算机视觉库 / Computer vision library
from torch import nn  # 神经网络模块 / Neural network modules
from torchvision.models._utils import IntermediateLayerGetter  # 中间层提取器 / Intermediate layer extractor

# 项目内部模块导入 / Internal module imports
from ManiBox.policy.misc import NestedTensor, is_main_process  # 嵌套张量和进程判断工具 / Nested tensor and process detection utilities
from .position_encoding import build_position_encoding  # 位置编码构建器 / Position encoding builder

# 调试工具 / Debugging tools
import IPython  # 交互式Python / Interactive Python
e = IPython.embed  # IPython调试快捷方式 / IPython debugging shortcut

class FrozenBatchNorm2d(torch.nn.Module):
    """冻结批正化层 - Frozen Batch Normalization Layer
    
    一种特殊的批正化层，其中批统计和仿射参数都是固定的。
    这在使用预训练模型时非常有用，可以防止预训练特征被破坏。
    
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    This is useful when using pretrained models to prevent disrupting pretrained features.
    
    技术详情 / Technical Details:
    - 从 torchvision.misc.ops 复制而来，在 rsqrt 前添加了 eps
    - Copy-paste from torchvision.misc.ops with added eps before rsqrt
    - 防止在非ResNet[18,34,50,101]模型中产生NaN / Prevents NaN in non-ResNet[18,34,50,101] models
    
    参数 / Parameters:
        n (int): 特征通道数 / Number of feature channels
    """

    def __init__(self, n):
        """初始化冻结批正化层 / Initialize frozen batch normalization layer
        
        Args:
            n (int): 输入特征的通道数 / Number of input feature channels
        """
        super(FrozenBatchNorm2d, self).__init__()  # 调用父类构造函数 / Call parent class constructor
        
        # 注册缓冲区参数（不参与梯度更新）/ Register buffer parameters (not involved in gradient updates)
        self.register_buffer("weight", torch.ones(n))      # 缩放参数，初始化为1 / Scale parameter, initialized to 1
        self.register_buffer("bias", torch.zeros(n))       # 偏置参数，初始化为0 / Bias parameter, initialized to 0
        self.register_buffer("running_mean", torch.zeros(n))  # 运行均值，初始化为0 / Running mean, initialized to 0
        self.register_buffer("running_var", torch.ones(n))   # 运行方差，初始化为1 / Running variance, initialized to 1

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """从状态字典加载参数 / Load parameters from state dictionary
        
        移除不需要的num_batches_tracked字段，因为冻结BN不跟踪批次数。
        Removes unnecessary num_batches_tracked field since frozen BN doesn't track batch count.
        """
        # 构建num_batches_tracked字段的完整键名 / Construct full key name for num_batches_tracked field
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:  # 如果存在该字段 / If the field exists
            del state_dict[num_batches_tracked_key]  # 删除不需要的字段 / Delete unnecessary field

        # 调用父类的加载方法 / Call parent class load method
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        """冻结批正化前向传播 / Frozen batch normalization forward pass
        
        Args:
            x (torch.Tensor): 输入特征张量 / Input feature tensor
            
        Returns:
            torch.Tensor: 正化后的特征张量 / Normalized feature tensor
        """
        # 将重塑操作移到开始位置，使其对融合器更友好 / Move reshapes to the beginning to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)    # 缩放权重重塑为4D / Reshape scale weight to 4D
        b = self.bias.reshape(1, -1, 1, 1)      # 偏置参数重塑为4D / Reshape bias parameter to 4D
        rv = self.running_var.reshape(1, -1, 1, 1)   # 运行方差重塑为4D / Reshape running variance to 4D
        rm = self.running_mean.reshape(1, -1, 1, 1)  # 运行均值重塑为4D / Reshape running mean to 4D
        
        eps = 1e-5  # 防止除零的小常数 / Small constant to prevent division by zero
        scale = w * (rv + eps).rsqrt()  # 计算缩放因子：weight * 1/sqrt(variance + eps) / Calculate scale factor: weight * 1/sqrt(variance + eps)
        bias = b - rm * scale           # 计算偏置：bias - mean * scale / Calculate bias: bias - mean * scale
        
        return x * scale + bias  # 应用批正化：y = (x * scale) + bias / Apply batch normalization: y = (x * scale) + bias


class BackboneBase(nn.Module):
    """主干网络基类 - Backbone Base Class
    
    为所有主干网络提供统一的接口，封装了中间层提取和参数管理功能。
    支持灵活的层选择和特征提取，用于构建不同复杂度的视觉编码器。
    
    Provides unified interface for all backbone networks, encapsulating intermediate 
    layer extraction and parameter management. Supports flexible layer selection 
    and feature extraction for building visual encoders of different complexity.
    """

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        """初始化主干网络基类 / Initialize backbone base class
        
        Args:
            backbone (nn.Module): 预训练的主干网络模型 / Pretrained backbone network model
            train_backbone (bool): 是否训练主干网络参数 / Whether to train backbone parameters
            num_channels (int): 输出特征通道数 / Number of output feature channels
            return_interm_layers (bool): 是否返回中间层特征 / Whether to return intermediate layer features
        """
        super().__init__()  # 调用父类构造函数 / Call parent class constructor
        
        # 注释掉的代码：只训练后面的层 / Commented code: only train later layers
        # for name, parameter in backbone.named_parameters(): # 只训练后面的层 TODO 是否需要这个? / only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)  # 冻结前面层的参数 / Freeze early layer parameters
        
        # 根据需求配置输出层 / Configure output layers based on requirements
        if return_interm_layers:  # 返回多层特征用于特征金字塔 / Return multi-layer features for feature pyramid
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}  # 所有ResNet层 / All ResNet layers
        else:  # 只返回最后一层特征 / Return only final layer features
            return_layers = {'layer4': "0"}  # 只有最后一层 / Only final layer
            
        # 创建中间层提取器，用于获取指定层的输出 / Create intermediate layer getter for specified layer outputs
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels  # 保存输出通道数 / Save number of output channels

    def forward(self, tensor):
        """主干网络前向传播 / Backbone forward pass
        
        Args:
            tensor (torch.Tensor): 输入图像张量 / Input image tensor
            
        Returns:
            Dict[str, torch.Tensor]: 层名到特征张量的映射 / Mapping from layer names to feature tensors
        """
        xs = self.body(tensor)  # 通过中间层提取器获取特征 / Extract features through intermediate layer getter
        return xs  # 返回特征字典 / Return feature dictionary
        
        # 注释掉的代码：原本用于创建嵌套张量的逻辑 / Commented code: originally for creating nested tensors
        # out: Dict[str, NestedTensor] = {}  # 输出字典 / Output dictionary
        # for name, x in xs.items():  # 遍历每个特征层 / Iterate through each feature layer
        #     m = tensor_list.mask  # 获取掩码 / Get mask
        #     assert m is not None  # 确保掩码存在 / Ensure mask exists
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]  # 插值掩码到特征尺寸 / Interpolate mask to feature size
        #     out[name] = NestedTensor(x, mask)  # 创建嵌套张量 / Create nested tensor
        # return out  # 返回嵌套张量字典 / Return nested tensor dictionary


class Backbone(BackboneBase):
    """ResNet主干网络，使用冻结批正化 - ResNet backbone with frozen BatchNorm
    
    基于TorchVision的ResNet模型构建的主干网络，用于视觉特征提取。
    支持ResNet18/34/50/101等不同深度的网络，可选择使用预训练权重。
    
    ResNet backbone based on TorchVision's ResNet models for visual feature extraction.
    Supports different depths like ResNet18/34/50/101, with optional pretrained weights.
    """
    
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        """初始化ResNet主干网络 / Initialize ResNet backbone
        
        Args:
            name (str): ResNet模型名称（如'resnet50'）/ ResNet model name (e.g., 'resnet50')
            train_backbone (bool): 是否训练主干网络参数 / Whether to train backbone parameters
            return_interm_layers (bool): 是否返回中间层 / Whether to return intermediate layers
            dilation (bool): 是否使用空洞卷积 / Whether to use dilated convolution
        """
        # 预训练模型加载配置 / Pretrained model loading configuration
        # pretrained = is_main_process()  # 注释：只在主进程中加载 / Commented: only load in main process
        pretrained = False  # 当前设置为不使用预训练模型 / Currently set to not use pretrained model
        # print("pretrained", pretrained)  # 调试输出 / Debug output
        
        # 从 TorchVision 创建 ResNet 模型 / Create ResNet model from TorchVision
        backbone = getattr(torchvision.models, name)(  # 动态获取指定的ResNet模型 / Dynamically get specified ResNet model
            replace_stride_with_dilation=[False, False, dilation],  # 空洞卷积配置：只在最后一层使用 / Dilation config: only use in last layer
            pretrained=pretrained,  # 是否使用预训练权重 / Whether to use pretrained weights
            norm_layer=FrozenBatchNorm2d  # 使用冻结批正化层 / Use frozen batch normalization layer
        )  # pretrained # TODO 是否需要冻结批正化? / TODO do we want frozen batch_norm??

        # 根据ResNet类型确定输出通道数 / Determine output channels based on ResNet type
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048  # 浅层ResNet使用512，深层使用2048 / Shallow ResNet uses 512, deep ResNet uses 2048

        # 调用父类构造函数 / Call parent class constructor
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    """主干网络和位置编码的组合器 - Joiner for backbone and position encoding
    
    将主干网络和位置编码器组合成一个统一的模块，
    用于为Transformer提供带有位置信息的视觉特征。
    
    Combines backbone network and position encoder into a unified module,
    providing visual features with positional information for Transformer.
    """
    
    def __init__(self, backbone, position_embedding):
        """初始化组合器 / Initialize joiner
        
        Args:
            backbone: 主干网络模型 / Backbone network model
            position_embedding: 位置编码器 / Position embedding module
        """
        super().__init__(backbone, position_embedding)  # 将两个模块组合成Sequential / Combine two modules into Sequential

    def forward(self, tensor_list: NestedTensor):
        """组合器前向传播 / Joiner forward pass
        
        Args:
            tensor_list (NestedTensor): 嵌套张量输入（包含图像和掩码）/ Nested tensor input (contains image and mask)
            
        Returns:
            tuple: (特征列表, 位置编码列表) / (feature list, position encoding list)
        """
        xs = self[0](tensor_list)  # 通过主干网络提取特征 / Extract features through backbone network
        out: List[NestedTensor] = []  # 初始化输出特征列表 / Initialize output feature list
        pos = []  # 初始化位置编码列表 / Initialize position encoding list
        
        # 遍历每个特征层 / Iterate through each feature layer
        for name, x in xs.items():
            out.append(x)  # 添加特征到输出列表 / Add feature to output list
            # 生成并添加位置编码 / Generate and add position encoding
            pos.append(self[1](x).to(x.dtype))  # 保持数据类型一致 / Maintain consistent data type

        return out, pos  # 返回特征和位置编码 / Return features and position encodings


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
        #                             RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [4, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [4, 1]),
                                    RestNetBasicBlock(256, 256, 1))
        self.num_channels = 256
        # self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
        #                             RestNetBasicBlock(512, 512, 1))

        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #
        # self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        # out = self.avgpool(out)
        # out = out.reshape(x.shape[0], -1)
        # out = self.fc(out)
        return out


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
