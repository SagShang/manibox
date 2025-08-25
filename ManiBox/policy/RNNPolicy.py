
"""循环神经网络策略模块 - RNN-based Policy Module

本模块实现了基于循环神经网络（RNN）的机器人控制策略。
与ACTPolicy中的Transformer架构不同，RNNPolicy使用LSTM来处理时序信息，
适合需要记忆先前状态的连续控制任务。

This module implements an RNN-based robot control policy using LSTM architecture.
Unlike the Transformer-based ACTPolicy, RNNPolicy uses LSTM to handle temporal information,
suitable for continuous control tasks that require memory of previous states.

主要组件 / Main Components:
- RNNPolicy: 主策略类，集成优化器、调度器和损失函数 / Main policy class with optimizer, scheduler and loss function
- RNN: 核心LSTM网络，处理多模态输入并预测动作 / Core LSTM network for multi-modal input and action prediction  
- MLP_block: 多层感知机构建块 / Multi-layer perceptron building block
- initialize_weights: 网络权重初始化函数 / Network weight initialization function

架构特点 / Architecture Features:
- 使用YOLO预处理的视觉特征（物体边界框信息）/ Uses YOLO-preprocessed visual features (object bounding boxes)
- LSTM处理时序依赖，支持循环推理 / LSTM handles temporal dependencies with recurrent inference
- 支持多相机输入和机器人状态融合 / Supports multi-camera input and robot state fusion
- 可配置的损失函数（L1、L2、Smooth L1）/ Configurable loss functions (L1, L2, Smooth L1)
"""

# 标准库导入 / Standard library imports
import argparse  # 命令行参数解析 / Command line argument parsing

# PyTorch相关导入 / PyTorch related imports
import torch.nn as nn  # 神经网络模块 / Neural network modules
from torch.nn import functional as F  # 神经网络函数 / Neural network functions
import torchvision.transforms as transforms  # 图像变换 / Image transformations
import torch  # PyTorch核心库 / PyTorch core library
from torch import nn  # 神经网络模块（重复导入）/ Neural network modules (duplicate import)

# 调试和进度条工具 / Debugging and progress bar tools
import IPython  # 交互式Python / Interactive Python
from tqdm import tqdm  # 进度条显示 / Progress bar display

# 项目内部模块导入 / Internal module imports
from ManiBox.policy.backbone import DepthNet, build_backbone  # 视觉主干网络 / Visual backbone networks
from ManiBox.yolo_process_data import YoloProcessDataByTimeStep  # YOLO数据处理器 / YOLO data processor
from transformers import get_cosine_schedule_with_warmup  # 余弦学习率调度器 / Cosine learning rate scheduler
e = IPython.embed  # IPython调试快捷方式 / IPython debugging shortcut


class RNNPolicy(nn.Module):
    """循环神经网络策略类 - RNN Policy Class
    
    基于LSTM的机器人控制策略，用于处理时序机器人控制任务。
    与ACTPolicy不同，该策略使用循环神经网络来维持状态记忆，
    适合需要连续决策和状态依赖的任务。
    
    LSTM-based robot control policy for temporal robot control tasks.
    Unlike ACTPolicy, this policy uses recurrent neural networks to maintain state memory,
    suitable for tasks requiring continuous decision-making and state dependencies.
    
    参数说明 / Arguments:
        policy_config (dict): 策略配置字典，包含以下关键参数 / Policy configuration dictionary with key parameters:
            - camera_names: 相机名称列表 / List of camera names
            - lr: 学习率 / Learning rate  
            - lr_backbone: 主干网络学习率 / Backbone learning rate
            - weight_decay: 权重衰减 / Weight decay
            - epochs: 训练轮数 / Training epochs
            - rnn_layers: LSTM层数 / Number of LSTM layers
            - rnn_hidden_dim: LSTM隐藏维度 / LSTM hidden dimension
            - actor_hidden_dim: 动作网络隐藏维度 / Actor network hidden dimension
            - state_dim: 机器人状态维度 / Robot state dimension
            - action_dim: 动作维度 / Action dimension
            - loss_function: 损失函数类型 / Loss function type
    
    组件说明 / Components:
        - model: RNN核心模型 / Core RNN model
        - optimizer: AdamW优化器，支持不同学习率 / AdamW optimizer with different learning rates
        - scheduler: 余弦学习率调度器 / Cosine learning rate scheduler
        - loss_function: 可配置损失函数（L1/L2/Smooth L1）/ Configurable loss function (L1/L2/Smooth L1)
    """
    def __init__(self, policy_config):
        """初始化RNN策略网络 / Initialize RNN policy network
        
        Args:
            policy_config (dict): 包含所有策略配置参数的字典 / Dictionary containing all policy configuration parameters
        """
        super().__init__()  # 调用父类构造函数 / Call parent class constructor
        print("You are using RNNPolicy.")  # 打印策略类型提示 / Print policy type notification
        
        # 注释掉的代码：原本用于构建多个视觉主干网络的逻辑
        # Commented code: Originally for building multiple visual backbone networks
        # backbones = []   # 主干网络列表 / List of backbone networks
        # depth_backbones = None  # 深度图像主干网络 / Depth image backbone networks
        # if policy_config['use_depth_image']:  # 如果使用深度图像 / If using depth images
        #     depth_backbones = []  # 初始化深度主干网络列表 / Initialize depth backbone list
        
        print("policy_config", policy_config)  # 打印配置信息用于调试 / Print config for debugging
        args = argparse.Namespace(**policy_config)  # 将字典转换为命名空间对象，便于属性访问 / Convert dict to namespace for attribute access
        camera_num = 0  # 相机数量计数器 / Camera count counter
        for _ in policy_config['camera_names']:  # 遍历相机名称列表 / Iterate through camera names list
            # 注释掉的代码：为每个相机构建主干网络
            # Commented code: Build backbone network for each camera
            # backbone = build_backbone(args)  # 从DETR策略中构建主干网络，需要args.xxx格式参数 / Build backbone from DETR policy, needs args.xxx format
            # backbones.append(backbone)  # 添加到主干网络列表 / Add to backbone list
            camera_num += 1  # 增加相机计数 / Increment camera count
            # if policy_config['use_depth_image']:  # 如果使用深度图像 / If using depth images
            #     depth_backbones.append(DepthNet())  # 添加深度网络 / Add depth network
        
        # 创建核心RNN模型 / Create core RNN model
        self.model = RNN(
            camera_num,  # 相机数量，影响视觉特征维度 / Number of cameras, affects visual feature dimension
            policy_config,  # 策略配置参数 / Policy configuration parameters
        )  # .to(policy_config['device'])  # 可选：移动到指定设备 / Optional: move to specified device
        
        # 构建参数字典，为不同组件设置不同学习率 / Build parameter dictionaries with different learning rates for different components
        param_dicts = [
            # 时序模型参数（非主干网络）使用默认学习率 / Temporal model parameters (non-backbone) use default learning rate
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            # 视觉编码器（主干网络）参数使用较小学习率 / Visual encoder (backbone) parameters use smaller learning rate
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": policy_config['lr_backbone'],  # 主干网络使用独立的较小学习率 / Backbone uses independent smaller learning rate
            },  # 主干网络（视觉编码器）模型，使用另一个学习率 / Backbone (visual encoder) model, using another learning rate
        ]
        # 创建AdamW优化器，支持权重衰减和分组学习率 / Create AdamW optimizer with weight decay and grouped learning rates
        self.optimizer = torch.optim.AdamW(param_dicts, lr=policy_config['lr'],
                                  weight_decay=policy_config['weight_decay'])
        
        # 根据配置选择学习率调度器 / Select learning rate scheduler based on configuration
        if args.use_scheduler == 'cos':  # 使用余弦退火调度器 / Use cosine annealing scheduler
            # 计算预热步数：总轮数 × 每轮步数 × 预热比例 / 梯度累积步数 / Calculate warmup steps: epochs × steps_per_epoch × warmup_ratio / gradient_accumulation
            warmup_steps = int(args.epochs * args.train_loader_len * args.warmup_ratio / args.gradient_accumulation_steps)
            # 计算总训练步数：总轮数 × 每轮步数 / 梯度累积步数 / Calculate total training steps: epochs × steps_per_epoch / gradient_accumulation
            total_steps = int(args.epochs * args.train_loader_len / args.gradient_accumulation_steps)
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
        elif args.use_scheduler == 'none':  # 不使用调度器 / No scheduler
            self.scheduler = None
        
        # 设置损失函数类型 / Set loss function type
        self.loss_function = policy_config['loss_function']  # 支持'l1', 'l2', 'smooth_l1'等 / Supports 'l1', 'l2', 'smooth_l1', etc.
        
        # 计算并打印模型参数统计信息 / Calculate and print model parameter statistics
        # 注释掉的代码：计算总参数数量 / Commented code: calculate total parameter count
        # n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print("total number of parameters: %.2fM" % (n_parameters/1e6,))
        
        # 分别统计主干网络和时序模型的参数数量 / Separately count backbone and temporal model parameters
        backbone_parameters = sum(p.numel() for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad)
        temporal_parameters = sum(p.numel() for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad)
        print("backbone visual encoder. number of parameters: %.2fM" % (backbone_parameters/1e6,))  # 打印主干网络参数量（百万） / Print backbone parameters (millions)
        print("temporal model. number of parameters: %.2fM" % (temporal_parameters/1e6,))  # 打印时序模型参数量（百万） / Print temporal model parameters (millions)
        

    def __call__(self, image, depth_image, robot_state, next_actions, next_actions_is_pad, actions=None,
                 action_is_pad=None):
        """RNN策略前向传播过程 / RNN Policy Forward Pass
        
        执行整个轨迹的前向传播。如果actions==None，将进行推理模式，只预测一个动作。
        推理模式会循环地预测动作序列。
        
        Forward process of whole trajectory. If actions == None, it will infer only one action.
        Inference mode will recurrently infer actions using for-loop.
        
        推理使用示例 / Inference Example:
        ```python
        yolo_process_data = YoloProcessDataByTimeStep()  # 创建YOLO数据处理器 / Create YOLO data processor
        yolo_process_data.reset_new_episode()  # 重置新回合 / Reset new episode
        
        policy = RNNPolicy(policy_config)  # 创建策略 / Create policy
        policy.reset_recur(batch_size, image.device)  # 重置循环状态 / Reset recurrent state
        for i in range(context_len):  # 循环预测每一步 / Loop predict each step
            image_data = yolo_process_data.process(cam_high, cam_left_wrist, cam_right_wrist)  # 处理图像 / Process images
            action = policy(image_data, None, robot_state, None, None, actions=None,
                 action_is_pad=None)  # 预测动作 / Predict action
        ```
        
        参数说明 / Arguments:
            image (torch.Tensor): 视觉特征张量 / Visual feature tensor
                训练时：batch_size * context_len * feature_dim / During training: batch_size * context_len * feature_dim
                推理时：batch_size * feature_dim / During inference: batch_size * feature_dim
            depth_image: 深度图像（可为None）/ Depth image (can be None)
            robot_state (torch.Tensor): 机器人状态张量 / Robot state tensor
                训练时：batch_size * context_len * 14 / During training: batch_size * context_len * 14  
                推理时：batch_size * 14 / During inference: batch_size * 14
            next_actions: 下一步动作（训练时可用，推理时为None）/ Next actions (used in training, None in inference)
            next_actions_is_pad: 下一步动作的填充掩码（可为None）/ Padding mask for next actions (can be None)
            actions (torch.Tensor, optional): 真实动作标签，仅训练时使用 / Ground truth actions, only used in training
                形状：batch_size * context_len * 14 / Shape: batch_size * context_len * 14
            action_is_pad: 动作填充掩码（可为None）/ Action padding mask (can be None)
        
        返回值 / Returns:
            训练模式 / Training mode: (loss_dict, predicted_actions)
            推理模式 / Inference mode: predicted_action
        
        注意事项 / Notes:
            - batch_size 至少为1 / batch_size is at least 1
            - context_len 至少为1 / context_len is at least 1
            - 推理时需要先调用reset_recur()重置循环状态 / Must call reset_recur() before inference to reset recurrent state
        """
        env_state = None  # TODO: 环境状态（待实现）/ Environment state (to be implemented)

        # 注释掉的代码：RGB图像标准化（ImageNet预训练权重的标准化参数）
        # Commented code: RGB image normalization (ImageNet pretrained weights normalization parameters)
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值 / ImageNet mean
        #                                  std=[0.229, 0.224, 0.225])   # ImageNet标准差 / ImageNet std
        
        # 深度图像标准化：将[0,1]映射到[-1,1] / Depth image normalization: map [0,1] to [-1,1]
        depth_normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        # image = normalize(image)  # 注释掉的RGB图像标准化 / Commented RGB image normalization
        if depth_image is not None:  # 如果提供了深度图像 / If depth image is provided
            depth_image = depth_normalize(depth_image)  # 对深度图像进行标准化 / Normalize depth image
        if actions is not None:  # training time
            # actions = actions[:, 0]  # 
            
            bs, context_len, state_dim = robot_state.shape
            self.reset_recur(bs, image.device)  # h, c
            # if not self.training:
            #     # recurrent inference
            #     # every time step, given a image and robot_state, the model will predict next action
            #     a_hat = torch.zeros((bs, context_len, self.model.action_dim)).to(image.device)
            #     state = torch.cat([image, robot_state], dim=-1)  # (bs, context_len, 24+14)
            #     for i in range(context_len):
            #         action, self.rnn_hidden = self.model(
            #             state[:, i:i+1, :], self.rnn_hidden
            #         )  # depth_image could be None
            #         a_hat[:, i:i+1, :] = action
            # else:
            state = torch.cat([image, robot_state], dim=-1)  # (bs, context_len, 24+14)
            a_hat, _ = self.model(
                state, self.rnn_hidden
            )
            
            grasp_start = 0
            grasp_end = 32
            
            action_label = actions.reshape(a_hat.shape)
            # 
            if self.loss_function == 'l1':
                # loss = F.l1_loss(action_label, a_hat)
                weighted_l1 = torch.abs(action_label - a_hat)
                # enlarge the grasp loss
                weighted_l1[:, grasp_start:grasp_end:, ] *= 3.0
                # weighted_l1[:, :, 6] *= 3.0  # gripper
                loss = weighted_l1.mean()
            elif self.loss_function == 'l2':
                loss = F.mse_loss(action_label, a_hat)
                
                # weighted_l2 = (action_label - a_hat) ** 2
                # # enlarge the grasp loss
                # weighted_l2[:, grasp_start:grasp_end:, ] *= 3.0
                # # weighted_l1[:, :, 6] *= 3.0  # gripper
                # loss = weighted_l2.mean()
            else:
                loss = F.smooth_l1_loss(action_label, a_hat)

            loss_dict = dict()
            loss_dict['loss'] = loss
            return loss_dict, a_hat

        else:  # inference time
            # image: (batch_size, 24), robot_state: (batch_size, 14)
            image = image.unsqueeze(dim=1)  # (bs, 1, 24)
            robot_state = robot_state.unsqueeze(dim=1)  # (bs, 1, 14)
            state = torch.cat([image, robot_state], dim=-1)  # (bs, 1, 24+14)
            # print("dtype: ", image.dtype, robot_state.dtype, self.rnn_hidden[0].dtype)
            a_hat, self.rnn_hidden = self.model(
                state, self.rnn_hidden
            )  # depth_image could be None
            return a_hat[:, 0, :]  # (bs, 14)

    def reset_recur(self, bs, device):
        self.rnn_hidden = self.model.init_hidden(bs, device)

def MLP_block(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


import torch.nn.init as init

def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)


class RNN(nn.Module):
    def __init__(self, camera_num, policy_config):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.policy_config = policy_config
        self.camera_num = camera_num
        
        camera_names = policy_config['camera_names']
        num_next_action = policy_config['num_next_action']
        state_dim = policy_config['state_dim']
        
        self.state_dim = state_dim
        self.action_dim = policy_config['action_dim']
        self.device = policy_config['device']
        
        self.num_next_action = num_next_action
        self.camera_names = camera_names
        
        self.rnn_layers = policy_config['rnn_layers']
        self.rnn_hidden_dim = policy_config['rnn_hidden_dim']
        self.actor_hidden_dim = policy_config['actor_hidden_dim']
        
        visual_embedding_dim = len(YoloProcessDataByTimeStep.objects_names) * 4 * camera_num # + camera_num * camera_num
        self.visual_embedding_dim = visual_embedding_dim
        # self.visual_encoder = MultiCameraModel(visual_embedding_dim)
        # ResNet3DWithPositionEncoding
        # self.backbone_visual_encoder = ResNet3DWithPositionEncoding(BasicBlock3D, [2, 2, 2, 2], visual_embedding_dim)
        # self.backbone_visual_encoder.apply(initialize_weights)

        state_total_dim = visual_embedding_dim + state_dim + state_dim * self.num_next_action
        self.rnn = nn.LSTM(state_total_dim, self.rnn_hidden_dim, num_layers=self.rnn_layers, batch_first=True)
        self.fc = nn.Linear(self.rnn_hidden_dim, self.actor_hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.action_head = nn.Linear(self.actor_hidden_dim, self.action_dim)
        # TODO: extract the vision encoder, try to feed the whole trajectory and preprocess the images

    
    def init_hidden(self, bs, device):
        h = torch.zeros(self.rnn_layers, bs, self.rnn_hidden_dim).to(device)
        c = torch.zeros(self.rnn_layers, bs, self.rnn_hidden_dim).to(device)
        return h, c
    
    def forward(self, all_state, hidden=None):
        """
        robot_state: batch, (context_len), qpos_dim
        image: batch, (context_len), image_dim(24)
        actions: batch, (context_len), action_dim
        env_state: None
        """
        # is_parallel_input = len(robot_state.shape) == 3
        # # print("image, robot_state, actions: ", image.shape, robot_state.shape, actions.shape)
        # bs = robot_state.shape[0]
        # # _, _, cam_num, channel, height, width = image.shape
        
        # flattened_visual_features = image
        # # import pdb; pdb.set_trace()
        
        # # concat visual features and robot state
        # all_feature = torch.cat([flattened_visual_features, robot_state], axis=1+is_parallel_input)  # qpos: 14
        
        # if not is_parallel_input:
        #     all_feature = all_feature.unsqueeze(dim=1)
        # all_feature: (bs, context_len or 1, 768 * cam_num + state_dim)
        # the dim-1 is context_len
        
        self.rnn.flatten_parameters()
        if hidden is None:  # hidden could be None
            rnn_output, h = self.rnn(all_state)
        else:
            # import pdb; pdb.set_trace()
            # hidden[0], hidden[1]: (num_layers * num_directions=2, bs, rnn_hidden_dim)
            rnn_output, h = self.rnn(all_state, hidden)  # h~hidden
            # rnn_output: (bs, 1, rnn_hidden_dim)
            # if not is_parallel_input:
            #     rnn_output = rnn_output.squeeze(dim=1)
        actions = F.gelu(self.fc(rnn_output))
        actions = self.dropout(actions)
        actions = self.action_head(actions)
        
        # import pdb; pdb.set_trace()
        if hidden is None:
            return actions
        else:
            return actions, h
        
        # return a_hat
