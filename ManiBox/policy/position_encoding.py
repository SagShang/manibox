# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""位置编码模块 - Positional Encoding Module

为Transformer模型提供各种位置编码实现。
位置编码是Transformer架构中的关键组件，用于为模型提供序列位置信息。
由于Transformer没有内置的位置感知能力，位置编码对于模型理解输入序列的空间关系至关重要。

Various positional encodings for the transformer.
Positional encoding is a key component in Transformer architecture, providing
sequential position information to the model. Since Transformers lack built-in
positional awareness, positional encoding is crucial for understanding spatial
relationships in input sequences.

实现类型 / Implementation Types:
1. PositionEmbeddingSine: 正弦位置编码（固定）/ Sinusoidal positional encoding (fixed)
   - 基于正弦和余弦函数的数学公式 / Based on sine and cosine mathematical formulas
   - 可外推到更长序列 / Can extrapolate to longer sequences
   - 适用于2D图像位置编码 / Suitable for 2D image positional encoding

2. PositionEmbeddingLearned: 可学习位置编码 / Learnable positional encoding
   - 通过神经网络学习得到 / Learned through neural networks
   - 更灵活但限制于训练长度 / More flexible but limited to training length
   - 适用于特定任务优化 / Suitable for task-specific optimization
"""
# 数学库导入 / Math library import
import math  # 数学运算函数 / Mathematical operation functions

# PyTorch相关导入 / PyTorch related imports
import torch       # PyTorch核心库 / PyTorch core library
from torch import nn  # 神经网络模块 / Neural network modules

# 项目内部模块导入 / Internal module imports
from ManiBox.policy.misc import NestedTensor  # 嵌套张量类 / Nested tensor class

# 调试工具 / Debugging tools
import IPython  # 交互式Python / Interactive Python
e = IPython.embed  # IPython调试快捷方式 / IPython debugging shortcut

class PositionEmbeddingSine(nn.Module):
    """正弦位置编码类 - Sinusoidal Position Embedding Class
    
    这是一个更标准版本的位置编码实现，非常类似于《Attention is All You Need》
    论文中使用的方法，并且被泛化为能够处理图像数据。
    
    This is a more standard version of the position embedding, very similar to the one
    used by the "Attention is All You Need" paper, generalized to work on images.
    
    核心特点 / Key Features:
    - 基于正弦和余弦函数 / Based on sine and cosine functions
    - 不需要训练参数 / No trainable parameters required
    - 可外推到任意尺寸 / Can extrapolate to arbitrary sizes
    - 对于2D图像进行了优化 / Optimized for 2D images
    
    数学公式 / Mathematical Formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    其中pos是位置，i是维度索引 / where pos is position, i is dimension index
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """初始化正弦位置编码器 / Initialize sinusoidal position encoder
        
        Args:
            num_pos_feats (int): 位置特征维度数，默认64 / Number of position feature dimensions, default 64
            temperature (int): 温度参数，控制频率范围，默认10000 / Temperature parameter controlling frequency range, default 10000
            normalize (bool): 是否对位置进行正则化 / Whether to normalize positions
            scale (float): 缩放因子，仅在normalize=True时使用 / Scale factor, only used when normalize=True
        """
        super().__init__()  # 调用父类构造函数 / Call parent class constructor
        
        # 保存初始化参数 / Store initialization parameters
        self.num_pos_feats = num_pos_feats  # 位置特征数量 / Number of position features
        self.temperature = temperature      # 温度参数，用于调节频率 / Temperature parameter for frequency adjustment
        self.normalize = normalize          # 是否正则化位置 / Whether to normalize positions
        
        # 参数验证 / Parameter validation
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")  # 只有在正则化时才能传入缩放因子 / Scale can only be passed when normalizing
            
        # 设置默认缩放因子 / Set default scale factor
        if scale is None:
            scale = 2 * math.pi  # 默认为2π，将位置映射到[0, 2π]范围 / Default to 2π, mapping positions to [0, 2π] range
        self.scale = scale  # 保存缩放因子 / Store scale factor

    def forward(self, tensor):
        """生成正弦位置编码 / Generate sinusoidal positional encoding
        
        Args:
            tensor (torch.Tensor): 输入张量，形状为[B, C, H, W] / Input tensor with shape [B, C, H, W]
            
        Returns:
            torch.Tensor: 位置编码张量，形状为[B, C, H, W] / Position encoding tensor with shape [B, C, H, W]
        """
        x = tensor  # 保存输入张量 / Store input tensor
        
        # 注释掉的代码：原本用于处理嵌套张量的掩码 / Commented code: originally for handling nested tensor masks
        # mask = tensor_list.mask      # 获取掩码信息 / Get mask information
        # assert mask is not None      # 确保掩码存在 / Ensure mask exists
        # not_mask = ~mask            # 反转掩码，获取有效区域 / Invert mask to get valid regions

        # 创建全1的掩码（表示所有位置都有效）/ Create all-ones mask (indicating all positions are valid)
        not_mask = torch.ones_like(x[0, [0]])  # 形状为[1, H, W] / Shape: [1, H, W]
        
        # 累积求和得到y和x坐标 / Cumulative sum to get y and x coordinates
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # 沿着高度维度累积求和 / Cumsum along height dimension
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 沿着宽度维度累积求和 / Cumsum along width dimension
        
        # 如果需要正则化位置 / If position normalization is required
        if self.normalize:
            eps = 1e-6  # 防止除零的小常数 / Small constant to prevent division by zero
            # 将y坐标正则化到[0, scale]范围 / Normalize y coordinates to [0, scale] range
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            # 将x坐标正则化到[0, scale]范围 / Normalize x coordinates to [0, scale] range
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 生成频率序列用于位置编码 / Generate frequency sequence for positional encoding
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  # [0, 1, ..., num_pos_feats-1]
        # 计算每个维度的频率因子 / Calculate frequency factor for each dimension
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # 频率递减序列 / Decreasing frequency sequence

        # 为x和y坐标应用频率变换 / Apply frequency transformation to x and y coordinates
        pos_x = x_embed[:, :, :, None] / dim_t  # 形状: [1, H, W, num_pos_feats] / Shape: [1, H, W, num_pos_feats]
        pos_y = y_embed[:, :, :, None] / dim_t  # 形状: [1, H, W, num_pos_feats] / Shape: [1, H, W, num_pos_feats]

        # 对偶数维度应用正弦，对奇数维度应用余弦 / Apply sine to even dimensions, cosine to odd dimensions
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # 正弦余弦交替 / Alternating sine-cosine
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # 正弦余弦交替 / Alternating sine-cosine
        
        # 拼x和y的位置编码，并调整维度顺序 / Concatenate x and y position encodings and adjust dimension order
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # 从[1, H, W, 2*num_pos_feats]到[1, 2*num_pos_feats, H, W] / From [1, H, W, 2*num_pos_feats] to [1, 2*num_pos_feats, H, W]
        return pos  # 返回最终的位置编码 / Return final positional encoding


class PositionEmbeddingLearned(nn.Module):
    """可学习的绝对位置编码类 - Learnable Absolute Position Embedding Class
    
    与正弦位置编码不同，这种位置编码通过神经网络学习得到。
    使用可训练的嵌入层来学习最适合特定任务的位置表示。
    
    Absolute positional embedding that is learned through training.
    Unlike sinusoidal encoding, this uses trainable embedding layers
    to learn the most suitable positional representation for specific tasks.
    
    优势 / Advantages:
    - 可以学习任务特定的位置表示 / Can learn task-specific positional representations
    - 更灵活的位置关系建模 / More flexible positional relationship modeling
    
    局限性 / Limitations:
    - 只能处理训练时见过的最大尺寸 / Limited to maximum size seen during training
    - 需要更多的训练数据 / Requires more training data
    """
    def __init__(self, num_pos_feats=256):
        """初始化可学习位置编码器 / Initialize learnable position encoder
        
        Args:
            num_pos_feats (int): 位置特征维度，默认256 / Position feature dimension, default 256
        """
        super().__init__()  # 调用父类构造函数 / Call parent class constructor
        
        # 创建行和列的嵌入层，最大支持50x50的图像 / Create row and column embedding layers, max support 50x50 images
        self.row_embed = nn.Embedding(50, num_pos_feats)  # 行位置嵌入层 / Row position embedding layer
        self.col_embed = nn.Embedding(50, num_pos_feats)  # 列位置嵌入层 / Column position embedding layer
        
        self.reset_parameters()  # 初始化参数 / Initialize parameters

    def reset_parameters(self):
        """重置嵌入层参数 / Reset embedding layer parameters
        
        使用均匀分布初始化权重，这是位置编码的常用做法。
        Initialize weights with uniform distribution, common practice for positional encodings.
        """
        nn.init.uniform_(self.row_embed.weight)  # 均匀分布初始化行嵌入权重 / Uniform initialization for row embedding weights
        nn.init.uniform_(self.col_embed.weight)  # 均匀分布初始化列嵌入权重 / Uniform initialization for column embedding weights

    def forward(self, tensor_list: NestedTensor):
        """生成可学习位置编码 / Generate learnable positional encoding
        
        Args:
            tensor_list (NestedTensor): 嵌套张量输入 / Nested tensor input
            
        Returns:
            torch.Tensor: 位置编码张量，形状为[B, 2*num_pos_feats, H, W] / Position encoding tensor with shape [B, 2*num_pos_feats, H, W]
        """
        x = tensor_list.tensors  # 提取张量数据 / Extract tensor data
        h, w = x.shape[-2:]      # 获取高度和宽度 / Get height and width
        
        # 生成坐标索引 / Generate coordinate indices
        i = torch.arange(w, device=x.device)  # 列索引: [0, 1, ..., w-1] / Column indices: [0, 1, ..., w-1]
        j = torch.arange(h, device=x.device)  # 行索引: [0, 1, ..., h-1] / Row indices: [0, 1, ..., h-1]
        
        # 通过嵌入层获取位置编码 / Get position encodings through embedding layers
        x_emb = self.col_embed(i)  # 列位置编码: [w, num_pos_feats] / Column position encoding: [w, num_pos_feats]
        y_emb = self.row_embed(j)  # 行位置编码: [h, num_pos_feats] / Row position encoding: [h, num_pos_feats]
        
        # 组合行列位置编码为2D网格 / Combine row and column encodings into 2D grid
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),  # 将列编码扩展到每一行 / Expand column encoding to each row
            y_emb.unsqueeze(1).repeat(1, w, 1),  # 将行编码扩展到每一列 / Expand row encoding to each column
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)  # 合并并调整维度 / Concatenate and adjust dimensions
        
        return pos  # 返回最终的位置编码 / Return final positional encoding


def build_position_encoding(args):
    """构建位置编码器工厂函数 / Position encoding builder factory function
    
    根据配置参数选择并创建相应的位置编码器。
    支持正弦位置编码和可学习位置编码两种类型。
    
    Select and create the appropriate position encoder based on configuration parameters.
    Supports both sinusoidal and learnable positional encoding types.
    
    Args:
        args: 配置参数对象，包含以下属性 / Configuration parameter object with following attributes:
            - hidden_dim: 隐藏维度大小 / Hidden dimension size
            - position_embedding: 位置编码类型 / Position embedding type
                'v2'/'sine': 正弦位置编码 / Sinusoidal position encoding
                'v3'/'learned': 可学习位置编码 / Learnable position encoding
    
    Returns:
        nn.Module: 位置编码器模块 / Position encoder module
    
    Raises:
        ValueError: 当位置编码类型不受支持时抛出 / Raised when position encoding type is not supported
    """
    # 计算位置特征维度（隐藏维度的一半）/ Calculate position feature dimension (half of hidden dimension)
    N_steps = args.hidden_dim // 2
    
    # 根据指定的位置编码类型创建相应的编码器 / Create corresponding encoder based on specified position encoding type
    if args.position_embedding in ('v2', 'sine'):  # 正弦位置编码 / Sinusoidal position encoding
        # TODO: 找到更好的方式来暴露其他参数 / TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)  # 启用正则化 / Enable normalization
        
    elif args.position_embedding in ('v3', 'learned'):  # 可学习位置编码 / Learnable position encoding
        position_embedding = PositionEmbeddingLearned(N_steps)
        
    else:  # 不支持的位置编码类型 / Unsupported position encoding type
        raise ValueError(f"not supported {args.position_embedding}")  # 抛出异常 / Raise exception

    return position_embedding  # 返回创建的位置编码器 / Return created position encoder
