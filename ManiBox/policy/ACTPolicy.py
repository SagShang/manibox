"""
ACTPolicy.py

Action Chunking Transformer (ACT) 策略实现
基于变分自编码器和Transformer架构的机器人动作序列预测模型。
该模型能够从多模态观测（图像、关节位置等）预测动作序列，支持多相机输入和历史信息。

Action Chunking Transformer (ACT) Policy Implementation
Robot action sequence prediction model based on Variational Autoencoder and Transformer architecture.
This model can predict action sequences from multi-modal observations (images, joint positions, etc.),
supporting multi-camera inputs and historical information.
"""

# 参数解析和深度学习框架 / Argument parsing and deep learning framework
import argparse  # 命令行参数解析工具 / Command line argument parsing utility
import torch.nn as nn  # PyTorch神经网络模块 / PyTorch neural network modules
from torch.nn import functional as F  # PyTorch功能性函数 / PyTorch functional functions
import torchvision.transforms as transforms  # 图像变换工具 / Image transformation utilities
import torch  # PyTorch深度学习框架 / PyTorch deep learning framework
from torch.autograd import Variable  # 自动梯度变量（旧版API） / Automatic gradient variable (legacy API)

# 数值计算 / Numerical computing
import numpy as np  # 数值计算库 / Numerical computing library

# 调试工具 / Debugging tools
import IPython  # 交互式Python环境 / Interactive Python environment
e = IPython.embed  # 调试嵌入函数的快捷方式 / Shortcut for debug embedding function

# 项目特定模块 / Project-specific modules
from ManiBox.policy.backbone import DepthNet, build_backbone  # 骨干网络和深度网络 / Backbone networks and depth networks
from ManiBox.policy.transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer  # Transformer架构组件 / Transformer architecture components

def kl_divergence(mu, logvar):
    """
    计算KL散度（Kullback-Leibler divergence）
    Calculate KL divergence for Variational Autoencoder
    
    在变分自编码器中，KL散度用于衡量编码器输出的潜在分布与先验分布（通常是标准正态分布）之间的差异。
    KL散度作为正则化项，确保学习到的潜在表示具有良好的统计特性。
    
    In Variational Autoencoder, KL divergence measures the difference between the encoder's
    output latent distribution and the prior distribution (usually standard normal distribution).
    KL divergence serves as a regularization term to ensure learned latent representations have good statistical properties.
    
    Args:
        mu: 编码器输出的均值参数 / Mean parameters from encoder output
        logvar: 编码器输出的对数方差参数 / Log variance parameters from encoder output
        
    Returns:
        tuple: (total_kld, dimension_wise_kld, mean_kld)
            - total_kld: 总KL散度 / Total KL divergence
            - dimension_wise_kld: 按维度的KL散度 / Dimension-wise KL divergence
            - mean_kld: 平均KL散度 / Mean KL divergence
    """
    batch_size = mu.size(0)  # 获取批大小 / Get batch size
    assert batch_size != 0  # 确保批大小不为零 / Ensure batch size is not zero
    
    # 处理4维张量，将其重塑为2维 / Handle 4D tensors by reshaping to 2D
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))  # 重塑均值张量 / Reshape mean tensor
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))  # 重塑对数方差张量 / Reshape log variance tensor

    # 计算KL散度：KL(q||p) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # 这里q是编码器输出的分布，p是标准正态分布N(0,1)
    # Calculate KL divergence: KL(q||p) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # where q is encoder output distribution, p is standard normal N(0,1)
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # 计算不同类型的KL散度统计 / Calculate different types of KL divergence statistics
    total_kld = klds.sum(1).mean(0, True)  # 总KL散度：先按维度求和，再按批次平均 / Total KL: sum over dimensions, then mean over batch
    dimension_wise_kld = klds.mean(0)  # 按维度的KL散度：按批次平均 / Dimension-wise KL: mean over batch
    mean_kld = klds.mean(1).mean(0, True)  # 平均KL散度：先按维度平均，再按批次平均 / Mean KL: mean over dimensions, then over batch

    return total_kld, dimension_wise_kld, mean_kld


def build_encoder(args):
    """
    构建Transformer编码器
    Build Transformer encoder for the VAE architecture
    
    该编码器用于ACT模型的变分自编码器部分，将动作序列编码为潜在表示。
    编码器采用标准的Transformer架构，支持多头注意力和前馈网络。
    
    This encoder is used in the VAE part of ACT model to encode action sequences into latent representations.
    The encoder uses standard Transformer architecture with multi-head attention and feed-forward networks.
    
    Args:
        args: 配置参数命名空间 / Configuration arguments namespace
        
    Returns:
        TransformerEncoder: 配置好的Transformer编码器 / Configured Transformer encoder
    """
    # 从配置参数中提取编码器超参数 / Extract encoder hyperparameters from config
    d_model = args.hidden_dim  # 隐藏维度，通常为256或512 / Hidden dimension, typically 256 or 512
    dropout = args.dropout     # Dropout率，通常为0.1 / Dropout rate, typically 0.1
    nhead = args.nheads        # 注意力头数，通常为8 / Number of attention heads, typically 8
    dim_feedforward = args.dim_feedforward  # 前馈网络维度，通常为2048 / Feed-forward dimension, typically 2048
    num_encoder_layers = args.enc_layers  # 编码器层数，通常为4 / Number of encoder layers, typically 4
    # TODO: 考虑与VAE解码器共享层数参数 / TODO: Consider sharing layers parameter with VAE decoder
    normalize_before = args.pre_norm  # 是否在注意力前进行层归一化，通常为False / Whether to normalize before attention, typically False
    activation = "relu"  # 激活函数类型 / Activation function type

    # 创建单个编码器层 / Create single encoder layer
    encoder_layer = TransformerEncoderLayer(
        d_model,           # 模型维度 / Model dimension
        nhead,             # 注意力头数 / Number of attention heads
        dim_feedforward,   # 前馈网络维度 / Feed-forward dimension
        dropout,           # Dropout率 / Dropout rate
        activation,        # 激活函数 / Activation function
        normalize_before   # 归一化位置 / Normalization position
    )
    
    # 根据配置决定是否使用层归一化 / Decide whether to use layer normalization based on config
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    
    # 构建完整的编码器 / Build complete encoder
    encoder = TransformerEncoder(
        encoder_layer,        # 编码器层 / Encoder layer
        num_encoder_layers,   # 层数 / Number of layers
        encoder_norm         # 归一化层 / Normalization layer
    )

    return encoder


def reparametrize(mu, logvar):
    """
    变分自编码器的重参数化技巧
    Reparameterization trick for Variational Autoencoder
    
    重参数化技巧允许我们从参数化的分布中采样，同时保持梯度的可反向传播性。
    通过将随机性从参数中分离出来，我们可以使用标准的反向传播算法训练VAE。
    
    The reparameterization trick allows us to sample from parameterized distributions
    while maintaining gradient backpropagation. By separating randomness from parameters,
    we can train VAE using standard backpropagation algorithms.
    
    数学公式 / Mathematical formula:
    z = μ + σ * ε, 其中 ε ~ N(0,1)
    z = μ + σ * ε, where ε ~ N(0,1)
    
    Args:
        mu: 分布的均值参数 / Mean parameters of the distribution
        logvar: 分布的对数方差参数 / Log variance parameters of the distribution
        
    Returns:
        torch.Tensor: 重参数化后的采样值 / Reparameterized sample
    """
    std = logvar.div(2).exp()  # 计算标准差：std = exp(logvar/2) = sqrt(var) / Calculate standard deviation: std = exp(logvar/2) = sqrt(var)
    eps = Variable(std.data.new(std.size()).normal_())  # 从标准正态分布采样噪声 / Sample noise from standard normal distribution
    return mu + std * eps  # 应用重参数化公式：z = μ + σ * ε / Apply reparameterization formula: z = μ + σ * ε


def get_sinusoid_encoding_table(n_position, d_hid):
    """
    生成正弦位置编码表
    Generate sinusoidal position encoding table
    
    位置编码是Transformer架构中的关键组件，用于为序列中的每个位置提供位置信息。
    正弦位置编码使用不同频率的正弦和余弦函数来编码位置，具有良好的外推性质。
    
    Position encoding is a key component in Transformer architecture to provide positional
    information for each position in a sequence. Sinusoidal position encoding uses sine and
    cosine functions of different frequencies to encode positions with good extrapolation properties.
    
    编码公式 / Encoding formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        n_position: 最大位置数量 / Maximum number of positions
        d_hid: 隐藏维度大小 / Hidden dimension size
        
    Returns:
        torch.FloatTensor: 位置编码表，形状为(1, n_position, d_hid) / Position encoding table with shape (1, n_position, d_hid)
    """
    def get_position_angle_vec(position):
        """
        计算给定位置的角度向量
        Calculate angle vector for given position
        
        Args:
            position: 序列中的位置索引 / Position index in sequence
            
        Returns:
            list: 该位置对应的角度向量 / Angle vector for the position
        """
        # 对每个隐藏维度计算角度：pos / 10000^(2i/d_hid)
        # Calculate angle for each hidden dimension: pos / 10000^(2i/d_hid)
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    # 为所有位置生成角度表 / Generate angle table for all positions
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    
    # 对偶数维度应用正弦函数 / Apply sine function to even dimensions
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 维度2i / dimension 2i
    
    # 对奇数维度应用余弦函数 / Apply cosine function to odd dimensions
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 维度2i+1 / dimension 2i+1

    # 转换为PyTorch张量并添加批次维度 / Convert to PyTorch tensor and add batch dimension
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ACTPolicy(nn.Module):
    """
    Action Chunking Transformer (ACT) 策略类
    Action Chunking Transformer (ACT) Policy Class
    
    ACT是一个基于Transformer和变分自编码器的机器人策略模型，能够从多模态观测中预测动作序列。
    该模型的核心思想是将连续的动作预测问题转化为序列到序列的学习问题，通过Transformer架构
    处理多相机图像和机器人状态，生成未来多个时间步的动作。
    
    ACT is a robot policy model based on Transformer and Variational Autoencoder that predicts
    action sequences from multi-modal observations. The core idea is to transform continuous action
    prediction into sequence-to-sequence learning, using Transformer architecture to process
    multi-camera images and robot states to generate actions for multiple future timesteps.
    
    主要特性 / Key Features:
    - 多相机图像处理 / Multi-camera image processing
    - 变分自编码器用于动作序列建模 / VAE for action sequence modeling
    - 支持深度图像输入 / Support for depth image input
    - 可配置的损失函数 / Configurable loss functions
    - 学习率调度器支持 / Learning rate scheduler support
    """
    
    def __init__(self, policy_config):
        """
        初始化ACT策略模型
        Initialize ACT policy model
        
        Args:
            policy_config: 包含模型配置参数的字典 / Dictionary containing model configuration parameters
        """
        super().__init__()  # 调用父类构造函数 / Call parent class constructor
        print("You are using ACTPolicy.")  # 输出策略类型信息 / Print policy type information
        
        # 将配置字典转换为命名空间对象，便于属性访问 / Convert config dict to namespace for easy attribute access
        args = argparse.Namespace(**policy_config)
        # 原始的构建函数调用（已注释）/ Original build function call (commented)
        # model, optimizer, scheduler = build_ACT_model_and_optimizer(policy_config)
        
        # 模型配置 / Model configuration:
        # 根据是否使用机器人底座确定状态维度 / Determine state dimension based on whether robot base is used
        if args.use_robot_base:
            state_dim = 16  # TODO: 硬编码的维度，应该从配置中获取 / TODO: hardcoded dimension, should be obtained from config
        else:
            state_dim = 14  # 标准的14维关节状态 / Standard 14-dimensional joint state

        # 视觉特征提取骨干网络配置 / Visual feature extraction backbone configuration
        # 从状态输入的情况（当前未使用）/ From state input (currently unused)
        # backbone = None # 仅从状态输入时不需要卷积网络 / No need for conv nets when only from state
        
        # 从图像输入的情况 / From image input
        backbones = []   # 存储每个相机对应的骨干网络 / List to store backbone for each camera
        depth_backbones = None  # 深度图像处理网络初始化 / Initialize depth image processing networks
        
        # 如果使用深度图像，初始化深度网络列表 / If using depth images, initialize depth network list
        if args.use_depth_image:
            depth_backbones = []

        # 为每个相机创建对应的骨干网络 / Create corresponding backbone network for each camera
        for _ in args.camera_names:  # 遍历相机名称列表 / Iterate through camera names list
            backbone = build_backbone(args)  # 构建视觉特征提取骨干网络 / Build visual feature extraction backbone
            backbones.append(backbone)  # 添加到骨干网络列表 / Add to backbone list
            
            # 如果使用深度图像，为每个相机创建深度处理网络 / If using depth images, create depth processing network for each camera
            if args.use_depth_image:
                depth_backbones.append(DepthNet())  # 添加深度网络到列表 / Add depth network to list

        # 构建Transformer架构 / Build Transformer architecture
        transformer = build_transformer(args)  # 主要的序列到序列变换器 / Main sequence-to-sequence transformer

        # 变分自编码器的编码器部分 / Encoder part of Variational Autoencoder
        encoder = None  # 默认不使用编码器 / Default to not using encoder
        if args.kl_weight != 0:  # 如果KL权重不为零，则需要VAE编码器 / If KL weight is non-zero, VAE encoder is needed
            encoder = build_encoder(args)  # 构建用于潜在表示学习的编码器 / Build encoder for latent representation learning

        # 创建DETRVAE模型实例 / Create DETRVAE model instance
        model = DETRVAE(
            backbones,              # 视觉骨干网络列表 / List of visual backbone networks
            depth_backbones,        # 深度处理网络列表 / List of depth processing networks
            transformer,            # Transformer架构 / Transformer architecture
            encoder,                # VAE编码器 / VAE encoder
            state_dim=state_dim,    # 机器人状态维度 / Robot state dimension
            num_queries=args.chunk_size,  # 查询数量（动作序列长度）/ Number of queries (action sequence length)
            camera_names=args.camera_names,  # 相机名称列表 / Camera names list
            num_next_action=args.num_next_action,  # 下一步动作数量 / Number of next actions
            kl_weight=args.kl_weight  # KL散度权重 / KL divergence weight
        )

        # 计算并打印模型参数数量 / Calculate and print number of model parameters
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 统计可训练参数总数 / Count total trainable parameters
        print("number of parameters: %.2fM" % (n_parameters/1e6,))  # 以百万为单位显示参数数量 / Display parameter count in millions

        # 配置分层学习率参数组 / Configure hierarchical learning rate parameter groups
        # 为不同的模型组件设置不同的学习率，通常骨干网络使用较小的学习率
        # Set different learning rates for different model components, backbone usually uses smaller learning rate
        param_dicts = [
            {   # 非骨干网络参数组（Transformer、头部等）/ Non-backbone parameter group (Transformer, heads, etc.)
                "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
            },
            {   # 骨干网络参数组（预训练的CNN特征提取器）/ Backbone parameter group (pretrained CNN feature extractor)
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,  # 为骨干网络设置特定的学习率 / Set specific learning rate for backbone
            },
        ]

        # 优化器和学习率调度器配置 / Optimizer and learning rate scheduler configuration
        from transformers import get_cosine_schedule_with_warmup  # 导入余弦预热调度器 / Import cosine warmup scheduler

        # 创建AdamW优化器 / Create AdamW optimizer
        optimizer = torch.optim.AdamW(
            param_dicts,                    # 参数组列表 / Parameter groups list
            lr=args.lr,                     # 主学习率 / Main learning rate
            weight_decay=args.weight_decay  # 权重衰减（L2正则化）/ Weight decay (L2 regularization)
        )
        
        # 根据配置选择学习率调度策略 / Select learning rate scheduling strategy based on configuration
        if args.use_scheduler == 'cos':  # 使用余弦调度器 / Use cosine scheduler
            # 计算预热步数和总步数 / Calculate warmup steps and total steps
            warmup_steps = int(args.epochs * args.train_loader_len * args.warmup_ratio)
            total_steps = int(args.epochs * args.train_loader_len)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,      # 优化器 / Optimizer
                warmup_steps,   # 预热步数 / Warmup steps
                total_steps     # 总训练步数 / Total training steps
            )
        elif args.use_scheduler == 'none':  # 不使用学习率调度 / No learning rate scheduling
            scheduler = None
        else:
            raise NotImplementedError(f"Scheduler '{args.use_scheduler}' not implemented")  # 未实现的调度器类型 / Unimplemented scheduler type
        
        # 保存模型组件和训练配置 / Save model components and training configuration
        self.model = model  # 主模型（CVAE解码器）/ Main model (CVAE decoder)
        self.optimizer = optimizer  # 优化器 / Optimizer
        self.scheduler = scheduler  # 学习率调度器 / Learning rate scheduler
        
        # 保存训练相关的超参数 / Save training-related hyperparameters
        self.kl_weight = policy_config['kl_weight']  # KL散度损失权重 / KL divergence loss weight
        self.loss_function = policy_config['loss_function']  # 损失函数类型 / Loss function type

        print(f'KL Weight {self.kl_weight}')  # 打印KL权重配置信息 / Print KL weight configuration

    def __call__(self, image, depth_image, robot_state, next_actions, next_actions_is_pad, actions=None,
                 action_is_pad=None, history_qpos=None):
        """
        ACT策略的前向传播函数
        Forward pass function for ACT policy
        
        该函数根据是否提供动作标签来决定是训练模式还是推理模式。
        训练时计算损失函数，推理时直接返回预测的动作序列。
        
        This function determines training or inference mode based on whether action labels are provided.
        During training, it calculates loss functions; during inference, it directly returns predicted action sequences.
        
        Args:
            image: 多相机图像输入，形状为(batch, num_cameras, channels, height, width) / Multi-camera image input
            depth_image: 深度图像输入（可选）/ Depth image input (optional)
            robot_state: 机器人状态（关节位置等），形状为(batch, state_dim) / Robot state (joint positions, etc.)
            next_actions: 下一步动作序列（用于条件生成）/ Next action sequence (for conditional generation)
            next_actions_is_pad: 下一步动作的填充掩码 / Padding mask for next actions
            actions: 目标动作序列（训练时使用）/ Target action sequence (used during training)
            action_is_pad: 动作序列的填充掩码（训练时使用）/ Padding mask for actions (used during training)
            history_qpos: 历史关节位置（可选）/ Historical joint positions (optional)
            
        Returns:
            训练时: (loss_dict, predicted_actions) / During training: (loss_dict, predicted_actions)
            推理时: predicted_actions / During inference: predicted_actions
        """

        # 图像标准化预处理 / Image standardization preprocessing
        # 使用ImageNet预训练模型的标准化参数 / Use ImageNet pretrained model's normalization parameters
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet RGB通道均值 / ImageNet RGB channel means
            std=[0.229, 0.224, 0.225]    # ImageNet RGB通道标准差 / ImageNet RGB channel standard deviations
        )
        
        # 深度图像标准化（如果有的话）/ Depth image normalization (if available)
        depth_normalize = transforms.Normalize(
            mean=[0.5],  # 深度图像均值 / Depth image mean
            std=[0.5]    # 深度图像标准差 / Depth image standard deviation
        )

        # 应用标准化变换 / Apply normalization transforms
        image = normalize(image)  # 标准化RGB图像 / Normalize RGB images
        if depth_image is not None:
            depth_image = depth_normalize(depth_image)  # 标准化深度图像 / Normalize depth images

        # 限制动作序列长度为模型的最大查询数 / Limit action sequence length to model's max queries
        if actions is not None:  # 训练时模式 / Training time mode
            # 截取动作序列到模型支持的最大长度 / Truncate action sequence to maximum length supported by model
            actions = actions[:, :self.model.num_queries]
            action_is_pad = action_is_pad[:, :self.model.num_queries]

            # 调试断点（已注释）/ Debug breakpoint (commented)
            # import pdb  # Python调试器 / Python debugger
            # pdb.set_trace()  # 设置断点 / Set breakpoint
            
            # 数据类型转换和梯度设置 / Data type conversion and gradient setup
            robot_state = robot_state.to(dtype=image.dtype)  # 确保状态数据类型与图像一致 / Ensure state dtype matches image
            actions = actions.to(dtype=image.dtype)  # 确保动作数据类型一致 / Ensure action dtype consistency
            image = image.requires_grad_()  # 为图像启用梯度计算 / Enable gradient computation for images
            robot_state = robot_state.requires_grad_()  # 为机器人状态启用梯度计算 / Enable gradient computation for robot state
            
            # 替代方法（已注释）/ Alternative approaches (commented)
            # robot_state = robot_state.clone().detach().requires_grad_()  # 克隆并分离再启用梯度 / Clone and detach then enable gradients
            # self.model.zero_grad()  # 清零模型梯度 / Zero model gradients
            # 模型前向传播 / Model forward pass
            a_hat, (mu, logvar) = self.model(
                image,                # 多相机图像输入 / Multi-camera image input
                depth_image,          # 深度图像输入 / Depth image input
                robot_state,          # 机器人状态 / Robot state
                next_actions,         # 下一步动作 / Next actions
                next_actions_is_pad,  # 下一步动作填充掩码 / Next actions padding mask
                actions,              # 目标动作序列 / Target action sequence
                action_is_pad,        # 动作填充掩码 / Action padding mask
                history_qpos=history_qpos  # 历史关节位置 / Historical joint positions
            )

            # 损失函数计算 / Loss function calculation
            loss_dict = dict()  # 存储各种损失项的字典 / Dictionary to store various loss terms
            
            # 根据配置选择不同的损失函数 / Select different loss functions based on configuration
            if self.loss_function == 'l1':  # L1损失（平均绝对误差）/ L1 loss (Mean Absolute Error)
                all_l1 = F.l1_loss(actions, a_hat, reduction='none')  # 逐元素L1损失 / Element-wise L1 loss
            elif self.loss_function == 'l2':  # L2损失（均方误差）/ L2 loss (Mean Squared Error)
                all_l1 = F.mse_loss(actions, a_hat, reduction='none')  # 逐元素MSE损失 / Element-wise MSE loss
            else:  # 平滑L1损失（Huber损失）/ Smooth L1 loss (Huber loss)
                all_l1 = F.smooth_l1_loss(actions, a_hat, reduction='none')  # 逐元素平滑L1损失 / Element-wise smooth L1 loss

            # 计算掩码损失：只在非填充位置计算损失 / Calculate masked loss: only compute loss at non-padding positions
            l1 = (all_l1 * ~action_is_pad.unsqueeze(-1)).mean()  # 应用填充掩码并求平均 / Apply padding mask and compute mean
            
            # 调试断点（已注释）/ Debug breakpoint (commented)
            # import pdb  # Python调试器 / Python debugger
            # pdb.set_trace()  # 设置断点 / Set breakpoint

            # 构建损失字典 / Build loss dictionary
            loss_dict['l1'] = l1  # 重建损失项 / Reconstruction loss term
            
            # 如果使用变分自编码器，添加KL散度损失 / If using VAE, add KL divergence loss
            if self.kl_weight != 0:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)  # 计算KL散度 / Calculate KL divergence
                loss_dict['kl'] = total_kld[0]  # KL散度损失项 / KL divergence loss term
                # 总损失 = 重建损失 + KL散度损失 * 权重 / Total loss = reconstruction loss + KL divergence * weight
                loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            else:
                # 仅使用重建损失 / Use only reconstruction loss
                loss_dict['loss'] = loss_dict['l1']

            return loss_dict, a_hat  # 返回损失字典和预测动作 / Return loss dictionary and predicted actions
        else:  # 推理时模式 / Inference time mode
            # 推理时不提供目标动作，从先验分布采样 / During inference, no target actions provided, sample from prior
            a_hat, (_, _) = self.model(
                image,                # 多相机图像输入 / Multi-camera image input
                depth_image,          # 深度图像输入 / Depth image input
                robot_state,          # 机器人状态 / Robot state
                next_actions,         # 下一步动作 / Next actions
                next_actions_is_pad,  # 下一步动作填充掩码 / Next actions padding mask
                history_qpos=history_qpos  # 历史关节位置 / Historical joint positions
            )  # 不提供动作标签，从先验分布采样 / No action labels provided, sample from prior
            return a_hat  # 返回预测的动作序列 / Return predicted action sequence

class DETRVAE(nn.Module):
    """
    DETR-VAE模型：结合DETR和变分自编码器的机器人策略网络
    DETR-VAE Model: Robot policy network combining DETR and Variational Autoencoder
    
    该模型改编自DETR目标检测架构，用于机器人动作序列预测任务。
    它结合了变分自编码器的生成能力和Transformer的序列建模能力，
    能够从多模态观测中生成连贯的动作序列。
    
    This model adapts the DETR object detection architecture for robot action sequence prediction tasks.
    It combines the generative capability of Variational Autoencoder with the sequence modeling ability
    of Transformer, capable of generating coherent action sequences from multi-modal observations.
    
    主要组件 / Main Components:
    - 多相机视觉编码器 / Multi-camera visual encoder
    - 变分编码器（用于潜在表示学习）/ Variational encoder (for latent representation learning)
    - Transformer解码器（用于序列生成）/ Transformer decoder (for sequence generation)
    - 动作预测头 / Action prediction head
    """
    
    def __init__(self, backbones, depth_backbones, transformer, encoder, state_dim, num_queries, camera_names,
                 num_next_action, kl_weight):
        """
        初始化DETR-VAE模型
        Initialize DETR-VAE model
        
        参数说明 / Parameters:
            backbones: 视觉特征提取骨干网络列表，见backbone.py / List of visual feature extraction backbones, see backbone.py
            depth_backbones: 深度图像处理网络列表 / List of depth image processing networks
            transformer: Transformer架构模块，见transformer.py / Transformer architecture module, see transformer.py
            encoder: 变分编码器模块 / Variational encoder module
            state_dim: 机器人状态维度 / Robot state dimension
            num_queries: 查询数量，即动作序列长度。这是模型能预测的最大动作步数 / Number of queries, i.e., action sequence length. Maximum action steps the model can predict
            camera_names: 相机名称列表 / List of camera names
            num_next_action: 下一步动作的数量 / Number of next actions
            kl_weight: KL散度损失权重 / KL divergence loss weight
        """
        super().__init__()  # 调用父类构造函数 / Call parent class constructor
        
        # 保存核心配置参数 / Save core configuration parameters
        self.num_queries = num_queries  # 查询数量（动作序列长度）/ Number of queries (action sequence length)
        self.camera_names = camera_names  # 相机名称列表 / Camera names list
        self.transformer = transformer  # Transformer架构 / Transformer architecture
        self.encoder = encoder  # 变分编码器 / Variational encoder
        
        # 从Transformer获取隐藏维度 / Get hidden dimension from Transformer
        hidden_dim = transformer.d_model  # Transformer的模型维度 / Transformer's model dimension
        
        # 核心网络组件 / Core network components
        self.action_head = nn.Linear(hidden_dim, state_dim)  # 动作预测头：隐藏状态->动作 / Action prediction head: hidden state -> action
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # 查询嵌入：为每个动作步提供位置信息 / Query embeddings: provide positional info for each action step
        
        # 训练相关参数 / Training-related parameters
        self.kl_weight = kl_weight  # KL散度权重 / KL divergence weight
        self.num_next_action = num_next_action  # 下一步动作数量 / Number of next actions
        
        # 视觉特征处理配置 / Visual feature processing configuration
        if backbones is not None:  # 如果提供了视觉骨干网络 / If visual backbones are provided
            # print("backbones[0]", backbones[0])  # 调试信息（已注释）/ Debug info (commented)
            
            # 配置深度图像处理 / Configure depth image processing
            if depth_backbones is not None:  # 如果使用深度图像 / If using depth images
                self.depth_backbones = nn.ModuleList(depth_backbones)  # 深度处理网络列表 / List of depth processing networks
                # 输入投影：RGB + 深度特征 -> 隐藏维度 / Input projection: RGB + depth features -> hidden dimension
                total_channels = backbones[0].num_channels + depth_backbones[0].num_channels
                self.input_proj = nn.Conv2d(total_channels, hidden_dim, kernel_size=1)
            else:
                self.depth_backbones = None  # 不使用深度图像 / Not using depth images
                # 输入投影：仅RGB特征 -> 隐藏维度 / Input projection: RGB features only -> hidden dimension
                self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            
            # 保存视觉骨干网络 / Save visual backbone networks
            self.backbones = nn.ModuleList(backbones)
            
            # 状态和动作的线性投影层 / Linear projection layers for state and actions
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)  # 机器人状态投影 / Robot state projection
            if num_next_action != 0:  # 如果使用下一步动作信息 / If using next action information
                self.input_proj_next_action = nn.Linear(state_dim, hidden_dim)  # 下一步动作投影 / Next action projection
        
        else:  # 不使用视觉输入的情况 / Case without visual input
            # input_dim = 14 + 7 # 原始注释：机器人状态 + 环境状态 / Original comment: robot_state + env_state
            
            # 仅基于状态的投影层 / State-only projection layers
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)  # 机器人状态投影 / Robot state projection
            if num_next_action != 0:  # 如果使用下一步动作 / If using next actions
                self.input_proj_next_action = nn.Linear(state_dim, hidden_dim)  # 下一步动作投影 / Next action projection
            
            # 位置嵌入（用于基于状态的输入）/ Position embeddings (for state-based input)
            self.pos = torch.nn.Embedding(2, hidden_dim)  # 2个位置的嵌入 / Embeddings for 2 positions
            self.backbones = None  # 不使用视觉骨干网络 / No visual backbone networks

        # 变分编码器额外参数 / Variational encoder extra parameters
        self.latent_dim = 32  # 潜在变量z的最终维度 / Final dimension of latent variable z
        # TODO: 这个维度需要调参优化 / TODO: This dimension needs hyperparameter tuning
        self.cls_embed = nn.Embedding(1, hidden_dim)  # 额外的CLS分类令牌嵌入 / Extra CLS classification token embedding

        # 解码器额外参数 / Decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # 将潜在采样投影到嵌入空间 / Project latent sample to embedding space

        # 位置嵌入配置 / Position embedding configuration
        if num_next_action != 0:  # 如果使用下一步动作 / If using next actions
            # 为下一步动作学习位置嵌入 / Learned position embeddings for next actions
            self.next_action_pos = nn.Embedding(num_next_action, hidden_dim)  # 下一步动作位置嵌入 / Next action position embeddings
        else:
            self.next_action_pos = None  # 不使用下一步动作位置嵌入 / No next action position embeddings
        
        # 其他组件的位置嵌入 / Position embeddings for other components
        self.latent_pos = nn.Embedding(1, hidden_dim)  # 潜在变量位置嵌入 / Latent variable position embedding
        self.robot_state_pos = nn.Embedding(1, hidden_dim)  # 机器人状态位置嵌入 / Robot state position embedding

        # 变分自编码器相关的投影层 / VAE-related projection layers
        if kl_weight != 0:  # 如果使用变分自编码器 / If using variational autoencoder
            # 编码器输入的投影层 / Projection layers for encoder input
            self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)  # 将动作投影到嵌入空间 / Project actions to embedding space
            self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # 将关节位置投影到嵌入空间 / Project joint positions to embedding space
            
            if num_next_action != 0:  # 如果使用下一步动作 / If using next actions
                self.encoder_next_action_proj = nn.Linear(state_dim, hidden_dim)  # 将下一步动作投影到嵌入 / Project next actions to embedding

            # 潜在空间投影：隐藏状态 -> (均值, 方差) / Latent space projection: hidden state -> (mean, variance)
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2)  # 输出潜在变量的均值和方差 / Output mean and variance of latent variable
            
            # 注册位置编码表作为缓冲区（不参与梯度更新）/ Register position encoding table as buffer (not updated by gradients)
            pos_length = 1 + 1 + num_next_action + num_queries  # [CLS] + qpos + next_actions + action_sequence
            pos_table = get_sinusoid_encoding_table(pos_length, hidden_dim)
            self.register_buffer('pos_table', pos_table)  # [CLS], qpos, a_seq的位置编码 / Position encoding for [CLS], qpos, action sequence

    def forward(self, image, depth_image, robot_state, next_actions, next_action_is_pad, 
                actions=None, action_is_pad=None, history_qpos=None):
        """
        DETR-VAE模型的前向传播
        Forward pass of DETR-VAE model
        
        处理多模态输入（图像、机器人状态、动作序列等），通过变分编码器学习潜在表示，
        然后使用Transformer解码器生成动作序列。支持训练和推理两种模式。
        
        Process multi-modal inputs (images, robot states, action sequences, etc.), learn latent
        representations through variational encoder, then use Transformer decoder to generate action sequences.
        Supports both training and inference modes.
        
        参数形状说明 / Parameter shape descriptions:
            robot_state: (batch, qpos_dim) - 机器人关节位置状态 / Robot joint position state
            image: (batch, num_cam, channel, height, width) - 多相机图像输入 / Multi-camera image input
            depth_image: (batch, num_cam, 1, height, width) - 多相机深度图像（可选）/ Multi-camera depth images (optional)
            next_actions: (batch, num_next_action, action_dim) - 下一步动作序列 / Next action sequence
            actions: (batch, seq_len, action_dim) - 目标动作序列（训练时）/ Target action sequence (during training)
            
        返回 / Returns:
            predicted_actions: (batch, num_queries, action_dim) - 预测的动作序列 / Predicted action sequence
            latent_info: [mu, logvar] - 潜在变量的均值和对数方差 / Mean and log variance of latent variable
        """

        # 调试信息（已注释）/ Debug info (commented)
        # print("forward: ", qpos.shape, image.shape, env_state, actions.shape, action_is_pad.shape)

        # 判断是训练还是推理模式 / Determine if in training or inference mode
        is_training = actions is not None  # 训练模式：提供了动作标签；推理模式：未提供动作标签 / Training: action labels provided; Inference: no action labels
        bs, _ = robot_state.shape  # 获取批大小 / Get batch size

        # 从动作序列中获取潜在变量z / Obtain latent variable z from action sequence
        if is_training and self.kl_weight != 0:  # 训练时且使用VAE（KL权重非零）/ During training and using VAE (KL weight non-zero)
            # 通过变分编码器学习潜在表示，隐藏维度通常为512 / Learn latent representation through variational encoder, hidden_dim typically 512
            # 将各种输入投影到统一的嵌入空间 / Project various inputs to unified embedding space
            action_embed = self.encoder_action_proj(actions)  # 动作序列嵌入: (bs, seq, hidden_dim) / Action sequence embedding
            
            # 处理下一步动作嵌入（如果使用）/ Handle next action embedding (if used)
            if self.num_next_action != 0 and next_actions is not None:
                next_action_embed = self.encoder_next_action_proj(next_actions)  # 下一步动作嵌入: (bs, seq, hidden_dim) / Next action embedding
            else:
                next_action_embed = None  # 不使用下一步动作 / Not using next actions
            
            # 机器人状态嵌入 / Robot state embedding
            robot_state_embed = self.encoder_joint_proj(robot_state)  # (bs, hidden_dim)
            robot_state_embed = torch.unsqueeze(robot_state_embed, axis=1)  # 添加序列维度: (bs, 1, hidden_dim) / Add sequence dimension
            
            # CLS分类令牌嵌入（用于聚合序列信息）/ CLS classification token embedding (for aggregating sequence info)
            cls_embed = self.cls_embed.weight  # 获取CLS嵌入权重: (1, hidden_dim) / Get CLS embedding weights
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # 扩展到批大小: (bs, 1, hidden_dim) / Expand to batch size

            # 构建编码器输入序列 / Build encoder input sequence
            if next_actions is not None:  # 如果有下一步动作 / If next actions are available
                # 连接所有嵌入：[CLS] + 机器人状态 + 下一步动作 + 动作序列 / Concatenate all embeddings
                encoder_input = torch.cat([cls_embed, robot_state_embed, next_action_embed, action_embed], axis=1)
            else:
                # 连接嵌入：[CLS] + 机器人状态 + 动作序列 / Concatenate embeddings without next actions
                encoder_input = torch.cat([cls_embed, robot_state_embed, action_embed], axis=1)  # (bs, seq+1, hidden_dim)
            
            # 转换维度以适配Transformer：(seq_len, batch_size, hidden_dim) / Transpose for Transformer: (seq_len, batch_size, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            
            # 构建填充掩码 / Build padding mask
            cls_joint_is_pad = torch.full((bs, 2), False).to(robot_state.device)  # CLS和状态永远不是填充 / CLS and state are never padding
            
            if next_action_is_pad is not None:  # 如果有下一步动作的填充掩码 / If next action padding mask exists
                # 合并所有填充掩码 / Combine all padding masks
                is_pad = torch.cat([cls_joint_is_pad, next_action_is_pad, action_is_pad], axis=1)  # (bs, seq+1)
            else:
                # 合并填充掩码（无下一步动作）/ Combine padding masks (no next actions)
                is_pad = torch.cat([cls_joint_is_pad, action_is_pad], axis=1)  # (bs, seq+1)

            # 获取位置嵌入 / Obtain position embedding
            pos_embed = self.pos_table.clone().detach()  # 克隆并分离位置编码表（不参与梯度计算）/ Clone and detach position table (no gradient)
            pos_embed = pos_embed.permute(1, 0, 2)  # 转换维度: (seq+1, 1, hidden_dim) / Transpose dimensions
            
            # 通过编码器处理输入 / Process input through encoder
            encoder_output = self.encoder(
                encoder_input,               # 编码器输入序列 / Encoder input sequence
                pos=pos_embed,              # 位置嵌入 / Position embeddings
                src_key_padding_mask=is_pad  # 填充掩码 / Padding mask
            )
            encoder_output = encoder_output[0]  # 只取CLS令牌的输出用于潜在变量学习 / Take only CLS token output for latent learning
            
            # 生成潜在变量参数（均值和方差）/ Generate latent variable parameters (mean and variance)
            latent_info = self.latent_proj(encoder_output)  # 投影到潜在空间: hidden_dim -> latent_dim*2 / Project to latent space
            mu = latent_info[:, :self.latent_dim]  # 提取均值参数 / Extract mean parameters
            logvar = latent_info[:, self.latent_dim:]  # 提取对数方差参数 / Extract log variance parameters
            
            # 重参数化采样 / Reparameterization sampling
            latent_sample = reparametrize(mu, logvar)  # 从潜在分布中采样 / Sample from latent distribution
            latent_input = self.latent_out_proj(latent_sample)  # 将潜在采样投影回嵌入空间 / Project latent sample back to embedding space
        else:  # 推理模式或不使用VAE / Inference mode or not using VAE
            mu = logvar = None  # 不计算潜在变量参数 / No latent variable parameters
            # 创建零潜在变量（从先验分布采样）/ Create zero latent variable (sample from prior)
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(robot_state.device)
            latent_input = self.latent_out_proj(latent_sample)  # 投影零采样到嵌入空间 / Project zero sample to embedding space
        # 图像观测特征和位置嵌入处理 / Image observation features and position embeddings processing
        all_cam_features = []  # 存储所有相机的特征 / Store features from all cameras
        all_cam_depth_features = []  # 存储所有相机的深度特征 / Store depth features from all cameras
        all_cam_pos = []  # 存储所有相机的位置嵌入 / Store position embeddings from all cameras
        
        # 处理每个相机的输入 / Process input from each camera
        for cam_id, cam_name in enumerate(self.camera_names):
            # 原始硬编码版本（已注释）/ Original hardcoded version (commented)
            # features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
            
            # 通过对应的骨干网络提取特征和位置信息 / Extract features and position info through corresponding backbone
            features, src_pos = self.backbones[cam_id](image[:, cam_id])  # 当前实现：为每个相机使用独立的骨干网络 / Current implementation: independent backbone for each camera
            
            # 深度编码器测试代码（已注释）/ Depth encoder test code (commented)
            # image_test = image[:, cam_id][:, 0].unsqueeze(dim=1)
            # print("depth_encoder:", self.depth_encoder(image_test))
            
            # 提取最后一层的特征和位置信息 / Extract features and position info from last layer
            features = features[0]  # 取最后一层特征 / Take last layer features
            src_pos = src_pos[0]    # 取对应的位置嵌入 / Take corresponding position embeddings
            
            # 处理深度图像特征（如果有）/ Process depth image features (if available)
            if self.depth_backbones is not None and depth_image is not None:
                # 提取深度特征 / Extract depth features
                features_depth = self.depth_backbones[cam_id](depth_image[:, cam_id].unsqueeze(dim=1))
                # 合并RGB和深度特征后投影 / Combine RGB and depth features then project
                combined_features = torch.cat([features, features_depth], axis=1)
                all_cam_features.append(self.input_proj(combined_features))
            else:
                # 仅使用RGB特征 / Use only RGB features
                all_cam_features.append(self.input_proj(features))
            
            all_cam_pos.append(src_pos)  # 添加位置嵌入 / Add position embeddings
            
            # 调试断点（已注释）/ Debug breakpoint (commented)
            # import pdb
            # pdb.set_trace()
            
        # 本体感受特征处理 / Proprioception features processing
        # 将机器人状态投影到隐藏空间 / Project robot state to hidden space
        robot_state_input = self.input_proj_robot_state(robot_state)  # (bs, hidden_dim)
        robot_state_input = torch.unsqueeze(robot_state_input, axis=0)  # 添加序列维度: (1, bs, hidden_dim) / Add sequence dimension
        
        # 处理下一步动作输入（如果有）/ Process next action input (if available)
        if self.num_next_action != 0 and next_actions is not None:
            # 投影并转换维度以适配Transformer / Project and transpose for Transformer
            next_action_input = self.input_proj_next_action(next_actions).permute(1, 0, 2)  # (num_next_action, bs, hidden_dim)
        else:
            next_action_input = None  # 不使用下一步动作 / Not using next actions
        
        # 为潜在输入添加序列维度 / Add sequence dimension to latent input
        latent_input = torch.unsqueeze(latent_input, axis=0)  # (1, bs, hidden_dim)
        # 将相机维度折叠到宽度维度 / Fold camera dimension into width dimension
        # 这样做可以将多相机的特征图拼接成一个更宽的特征图，便于Transformer处理
        # This concatenates multi-camera feature maps into a wider feature map for Transformer processing
        src = torch.cat(all_cam_features, axis=3)  # 在宽度维度上拼接所有相机特征 / Concatenate all camera features along width dimension
        src_pos = torch.cat(all_cam_pos, axis=3)   # 在宽度维度上拼接所有相机位置嵌入 / Concatenate all camera position embeddings along width dimension
        
        # 调试断点（已注释）/ Debug breakpoint (commented)
        # import pdb
        # pdb.set_trace()
        # Transformer解码器处理 / Transformer decoder processing
        if self.num_next_action != 0 and next_actions is not None:  # 使用下一步动作的情况 / Case with next actions
            # 完整的Transformer解码，包含所有输入组件 / Full Transformer decoding with all input components
            hs = self.transformer(
                self.query_embed.weight,        # 查询嵌入（动作序列的位置查询）/ Query embeddings (positional queries for action sequence)
                src, src_pos, None,             # 视觉特征、位置嵌入、注意力掩码 / Visual features, position embeddings, attention mask
                robot_state_input, self.robot_state_pos.weight,  # 机器人状态及其位置嵌入 / Robot state and its position embedding
                next_action_input, self.next_action_pos.weight, next_action_is_pad,  # 下一步动作、位置嵌入、填充掩码 / Next actions, position embeddings, padding mask
                latent_input, self.latent_pos.weight  # 潜在变量及其位置嵌入 / Latent variable and its position embedding
            )[0]  # 取第一个输出（解码器的最后一层输出）/ Take first output (last decoder layer output)
        else:  # 不使用下一步动作的情况 / Case without next actions
            # 简化的Transformer解码 / Simplified Transformer decoding
            hs = self.transformer(
                self.query_embed.weight,        # 查询嵌入 / Query embeddings
                src, src_pos, None,             # 视觉输入 / Visual inputs
                robot_state_input, self.robot_state_pos.weight,  # 机器人状态 / Robot state
                None, None, None,               # 无下一步动作相关输入 / No next action related inputs
                latent_input, self.latent_pos.weight  # 潜在变量 / Latent variable
            )[0]
        
        # 通过动作预测头生成最终的动作序列 / Generate final action sequence through action prediction head
        a_hat = self.action_head(hs)  # 隐藏状态 -> 动作预测 / Hidden states -> action predictions
        
        return a_hat, [mu, logvar]  # 返回预测动作和潜在变量参数 / Return predicted actions and latent variable parameters