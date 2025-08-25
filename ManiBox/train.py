"""机器人学习策略训练主程序 - Robot Learning Policy Training Main Program

这是ManiBox机器人学习框架的主要训练脚本。
支持多种不同的机器人控制策略训练，包括：
- ACT (Action Chunking Transformer)
- RNN/LSTM 策略
- CNN-MLP 策略  
- Diffusion 策略

This is the main training script for the ManiBox robot learning framework.
Supports training of various robot control policies including:
- ACT (Action Chunking Transformer)
- RNN/LSTM policies
- CNN-MLP policies
- Diffusion policies

主要功能 / Main Features:
- 多策略架构支持 / Multi-policy architecture support
- 分布式训练 / Distributed training
- 自动混合精度 / Automatic mixed precision
- 数据增强 / Data augmentation
- 实时训练监控 / Real-time training monitoring
- 梯度累积 / Gradient accumulation
"""

# 标准库导入 / Standard library imports
import os                    # 操作系统接口 / Operating system interface
import pickle               # 对象序列化 / Object serialization
import argparse             # 命令行参数解析 / Command line argument parsing
from copy import deepcopy   # 深度复制 / Deep copy
import json                 # JSON数据处理 / JSON data processing
from datetime import datetime # 日期时间处理 / Date and time processing
import shutil               # 高级文件操作 / High-level file operations
import sys                  # 系统相关参数和函数 / System-specific parameters and functions
sys.path.append("./")       # 添加当前目录到Python路径 / Add current directory to Python path

# 第三方库导入 / Third-party library imports
from tqdm import tqdm       # 进度条显示 / Progress bar display
import numpy as np          # 数值计算库 / Numerical computing library
import matplotlib.pyplot as plt # 绘图库 / Plotting library
# from einops import rearrange  # 注释掉的张量重排库 / Commented tensor rearrangement library

# PyTorch相关导入 / PyTorch related imports
import torch.multiprocessing as mp  # 多进程支持 / Multi-processing support
import torch                         # PyTorch核心库 / PyTorch core library
import torch.nn.functional as F      # 神经网络函数 / Neural network functions
from torch.utils.data import TensorDataset, DataLoader  # 数据加载器 / Data loaders
from torch.cuda.amp import autocast, GradScaler         # 自动混合精度 / Automatic mixed precision
from accelerate import Accelerator   # 分布式训练加速器 / Distributed training accelerator

# 调试工具 / Debugging tools
import IPython              # 交互式Python / Interactive Python
# e = IPython.embed         # 注释掉的IPython调试 / Commented IPython debugging

# 项目内部模块导入 / Internal module imports
from ManiBox.utils import compute_dict_mean, set_seed, detach_dict  # 工具函数 / Utility functions
from ManiBox.dataloader.data_load import load_data  # 数据加载主函数 / Main data loading function
from ManiBox.dataloader.EpisodicDataset import EpisodicDataset  # 基本回合数据集 / Basic episodic dataset
from ManiBox.dataloader.HistoryEpisodicDataset import HistoryEpisodicDataset  # 历史回合数据集 / Historical episodic dataset
from ManiBox.dataloader.BBoxHistoryEpisodicDataset import BBoxHistoryEpisodicDataset  # 边界框历史数据集 / Bounding box historical dataset


def get_arguments():
    """解析命令行参数 / Parse command line arguments
    
    配置训练过程中的所有参数，包括数据路径、模型架构、训练参数等。
    支持多种策略类型和灵活的超参数配置。
    
    Configure all parameters for training process including data paths, 
    model architectures, training parameters, etc. Supports multiple 
    policy types and flexible hyperparameter configuration.
    
    Returns:
        argparse.Namespace: 包含所有训练参数的命名空间对象 / Namespace object containing all training parameters
    """
    parser = argparse.ArgumentParser()
    
    # 基本路径和任务参数 / Basic paths and task parameters
    # 注释掉的参数：原本要求必须指定 / Commented parameters: originally required
    # parser.add_argument('--dataset', action='store', type=str, help='dataset_dir', default='./dataset', required=True)
    # parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    
    parser.add_argument('--dataset', action='store', type=str, help='dataset_dir', default='./dataset')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', default='./ckpt')
    parser.add_argument('--pretrain_timestamp', action='store', type=str, help='pretrain_timestamp, like 2024-03-27_16-52-32', default='', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='', required=False)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', default=2000, required=False)
    
    # 模型和训练配置 / Model and training configuration
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize, CNNMLP, ACT, Diffusion', default='ACT', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=32, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=50, required=False)

    # 训练步骤参数 / Training step parameters
    parser.add_argument('--num_eval_step', action='store', type=int, help='num_eval_step', default=1, required=False)
    parser.add_argument('--num_train_step', action='store', type=int, help='num_train_step', default=5, required=False)

    # 优化器参数 / Optimizer parameters
    parser.add_argument('--lr', action='store', type=float, help='lr', default=7e-5, required=False)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)", required=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features", required=False)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=7e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--dropout', default=0.2, type=float, help="Dropout applied in the transformer", required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)
    parser.add_argument('--gradient_accumulation_steps', action='store', type=int, help='gradient_accumulation_steps', default=16, required=False)
    
    # which aug we use, default is None, now support aug, distort
    parser.add_argument('--aug', default=None, type=str)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', default=32, required=False)
    parser.add_argument('--max_pos_lookahead', action='store', type=int, help='max_pos_lookahead', default=0, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg',  action='store', type=bool, help='temporal_agg', default=True, required=False)
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help='warmup ratio for lr schedule')
    parser.add_argument('--scheduler', default='cos', type=str, help='schedule, support cos, none now')
    parser.add_argument('--use_accelerate', action='store', type=bool, help='whether use accelerate', default=False, required=False)
    parser.add_argument('--device', type=str, help='device', default='cuda:0')

    # for Diffusion
    parser.add_argument('--observation_horizon', action='store', type=int, help='observation_horizon', default=1, required=False)
    parser.add_argument('--action_horizon', action='store', type=int, help='action_horizon', default=8, required=False)
    parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=10, required=False)
    parser.add_argument('--ema_power', action='store', type=int, help='ema_power', default=0.75, required=False)

    # for CNNRNN, rnn_layers, rnn_hidden_dim
    parser.add_argument('--rnn_layers', action='store', type=int, help='rnn_layers', default=3, required=False)
    parser.add_argument('--rnn_hidden_dim', action='store', type=int, help='rnn_hidden_dim', default=512, required=False)
    parser.add_argument('--actor_hidden_dim', action='store', type=int, help='actor_hidden_dim', default=512, required=False)
    
    # for DiffusionState: max_time_steps, time_embed_dim
    parser.add_argument('--max_time_steps', action='store', type=int, help='max_time_steps', default=1000, required=False)
    parser.add_argument('--time_embed_dim', action='store', type=int, help='time_embed_dim', default=128, required=False)
    parser.add_argument('--num_samples_per_traj', action='store', type=int, help='num_samples_per_traj', default=10, required=False)
    parser.add_argument('--alpha', action='store', type=float, help='alpha', default=3.0, required=False)
    parser.add_argument('--fcnet_hidden_dim', action='store', type=int, help='fcnet_hidden_dim', default=512, required=False)
    parser.add_argument('--n_modes', action='store', type=int, help='n_modes', default=10, required=False)
    parser.add_argument('--n_layer', action='store', type=int, help='n_layer', default=4, required=False)
    parser.add_argument('--is_chunk_wise', action='store', type=bool, help='is_chunk_wise', default=False, required=False)
    parser.add_argument('--context_len', action='store', type=int, help='context_len', default=90, required=False)
    parser.add_argument('--ffn_hidden_size', action='store', type=int, help='ffn_hidden_size', default=1024, required=False)
    
    
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base', default=False, required=False)

    parser.add_argument('--arm_delay_time', action='store', type=int, help='arm_delay_time', default=0, required=False)

    parser.add_argument('--use_dataset_action', action='store', type=bool, help='use_dataset_action', default=True, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image', default=False, required=False)

    args = parser.parse_args()
    return args


args = get_arguments()
    
# torch.backends.cudnn.enabled = False
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
accelerator.wait_for_everyone()

time_now = datetime.now()
timestamp = time_now.strftime("%Y-%m-%d_%H-%M-%S")

print(f"Timestamp: {timestamp}")


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    """绘制和保存训练历史曲线 / Plot and save training history curves
    
    为每个损失指标生成训练和验证曲线，用于监控训练进程。
    Generate training and validation curves for each loss metric to monitor training progress.
    
    Args:
        train_history: 训练历史记录 / Training history records
        validation_history: 验证历史记录 / Validation history records  
        num_epochs: 训练轮数 / Number of training epochs
        ckpt_dir: 检查点保存目录 / Checkpoint save directory
        seed: 随机种子 / Random seed
    """
    # 保存训练曲线 / Save training curves
    for key in train_history[0]:  # 遍历每个损失指标 / Iterate through each loss metric
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')  # 构建保存路径 / Build save path
        plt.figure()  # 创建新图形 / Create new figure
        
        # 提取训练和验证数值 / Extract training and validation values
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        
        # 绘制训练和验证曲线 / Plot training and validation curves
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])  # 注释掉的y轴限制 / Commented y-axis limit
        
        # 设置图形属性 / Set figure properties
        plt.tight_layout()  # 紧凑布局 / Tight layout
        plt.legend()        # 显示图例 / Show legend
        plt.title(key)      # 设置标题 / Set title
        plt.savefig(plot_path)  # 保存图形 / Save figure
    print(f'Saved plots to {ckpt_dir}')  # 打印保存信息 / Print save information
    

import torchvision.transforms as transforms

# Define distortion operations
def distort_image(image):
    # transform = transforms.Compose([
    transform = transforms.RandomChoice([
        # transforms.RandomCrop(size=(224, 224)),  # Random crop
        # transforms.RandomHorizontalFlip(),  # Random horizontal flip
        # transforms.RandomVerticalFlip(),  # Random vertical flip
        # transforms.RandomRotation(degrees=15),  # random rotation [-15, 15]
        # # transforms.RandomCrop(size=(450, 600), padding=(15, 20), padding_mode='edge'),
        # transforms.RandomResizedCrop(size=(480, 640)),
        transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.2),  # Random adjustments to brightness, contrast, saturation, and hue
    ])
    return transform(image)


def make_policy(policy_class, policy_config, pretrain_ckpt_dir):
    """创建策略模型工厂函数 / Create policy model factory function
    
    根据指定的策略类型创建相应的策略模型实例。
    支持多种策略架构并可加载预训练模型。
    
    Create corresponding policy model instance based on specified policy type.
    Supports multiple policy architectures and can load pretrained models.
    
    Args:
        policy_class (str): 策略类型名称 / Policy type name
        policy_config (dict): 策略配置参数 / Policy configuration parameters
        pretrain_ckpt_dir (str): 预训练模型路径 / Pretrained model path
        
    Returns:
        Policy: 创建的策略模型实例 / Created policy model instance
    """
    # 处理预训练模型路径 / Handle pretrained model path
    if len(pretrain_ckpt_dir) != 0:
        pretrain_ckpt_dir = os.path.join(pretrain_ckpt_dir, "policy_best.ckpt")
        
    # ACT策略创建 / ACT policy creation
    if policy_class == 'ACT':
        from ManiBox.policy.ACTPolicy import ACTPolicy  # 动态导入ACT策略 / Dynamic import ACT policy
        policy = ACTPolicy(policy_config)  # 创建ACT策略实例 / Create ACT policy instance
        
        # 加载预训练模型 / Load pretrained model
        if len(pretrain_ckpt_dir) != 0:
            state_dict = torch.load(pretrain_ckpt_dir)  # 加载状态字典 / Load state dictionary
            new_state_dict = {}  # 创建新状态字典 / Create new state dictionary
            
            # 过滤不需要的模型参数 / Filter unnecessary model parameters
            for key, value in state_dict.items():
                # 过滤填充头参数 / Filter padding head parameters
                if key in ["model.is_pad_head.weight", "model.is_pad_head.bias"]:
                    continue
                # 过滤下一步动作投射层参数 / Filter next action projection layer parameters
                if policy_config['num_next_action'] == 0 and key in ["model.input_proj_next_action.weight",
                                                                     "model.input_proj_next_action.bias"]:
                    continue
                new_state_dict[key] = value  # 保留有效参数 / Keep valid parameters
                
            # 加载模型状态 / Load model state
            loading_status = policy.load_state_dict(new_state_dict)
            if not loading_status:
                print("ckpt path not exist")  # 模型加载失败提示 / Model loading failure notification
    elif policy_class == 'CNNMLP':
        from ManiBox.policy.CNNMLPPolicy import CNNMLPPolicy
        policy = CNNMLPPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'HistoryCNNMLP':
        from ManiBox.policy.HistoryCNNMLPPolicy import HistoryCNNMLPPolicy
        policy = HistoryCNNMLPPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'CNNRNN':
        # CNN-RNN混合策略：结合卷积神经网络和循环神经网络 / CNN-RNN hybrid policy: combines CNN and RNN
        from ManiBox.policy.CNNRNNPolicy import CNNRNNPolicy
        policy = CNNRNNPolicy(policy_config)  # 创建CNN-RNN策略实例 / Create CNN-RNN policy instance
        if len(pretrain_ckpt_dir) != 0:  # 如果提供了预训练模型路径 / If pretrained model path provided
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))  # 加载预训练权重 / Load pretrained weights
            if not loading_status:
                print("ckpt path not exist")  # 检查点路径不存在 / Checkpoint path does not exist
    
    elif policy_class == 'FPNRNN':
        # FPN-RNN策略：特征金字塔网络+循环神经网络 / FPN-RNN policy: Feature Pyramid Network + RNN
        from ManiBox.policy.FPNRNNPolicy import FPNRNNPolicy
        policy = FPNRNNPolicy(policy_config)  # 创建FPN-RNN策略实例 / Create FPN-RNN policy instance
        if len(pretrain_ckpt_dir) != 0:  # 如果提供了预训练模型路径 / If pretrained model path provided
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))  # 加载预训练权重 / Load pretrained weights
            if not loading_status:
                print("ckpt path not exist")  # 检查点路径不存在 / Checkpoint path does not exist
    
    elif policy_class == 'RNN':
        # 循环神经网络策略：基于LSTM的时序建模 / RNN policy: LSTM-based temporal modeling
        from ManiBox.policy.RNNPolicy import RNNPolicy
        policy = RNNPolicy(policy_config)  # 创建RNN策略实例 / Create RNN policy instance
        if len(pretrain_ckpt_dir) != 0:  # 如果提供了预训练模型路径 / If pretrained model path provided
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))  # 加载预训练权重 / Load pretrained weights
            if not loading_status:
                print("ckpt path not exist")  # 检查点路径不存在 / Checkpoint path does not exist
    
    elif policy_class == 'DiffusionState':
        # 扩散状态策略：基于扩散模型的状态空间方法 / Diffusion State policy: diffusion model-based state space approach
        from ManiBox.policy.DiffusionStatePolicy import DiffusionStatePolicy
        policy = DiffusionStatePolicy(policy_config)  # 创建扩散状态策略实例 / Create Diffusion State policy instance
        if len(pretrain_ckpt_dir) != 0:  # 如果提供了预训练模型路径 / If pretrained model path provided
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))  # 加载预训练权重 / Load pretrained weights
            if not loading_status:
                print("ckpt path not exist")  # 检查点路径不存在 / Checkpoint path does not exist
    
    elif policy_class == 'Diffusion':
        # 扩散策略：基于扩散概率模型的动作生成 / Diffusion policy: diffusion probabilistic model-based action generation
        from ManiBox.policy.DiffusionPolicy import DiffusionPolicy
        policy = DiffusionPolicy(policy_config)  # 创建扩散策略实例 / Create Diffusion policy instance
        if len(pretrain_ckpt_dir) != 0:  # 如果提供了预训练模型路径 / If pretrained model path provided
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))  # 加载预训练权重 / Load pretrained weights
            if not loading_status:
                print("ckpt path not exist")  # 检查点路径不存在 / Checkpoint path does not exist
    
    else:
        # 不支持的策略类型 / Unsupported policy type
        raise NotImplementedError(f"策略类型 '{policy_class}' 尚未实现 / Policy type '{policy_class}' not implemented")
    
    return policy  # 返回创建的策略实例 / Return created policy instance

def parse_dataloader(train_dataloader: torch.utils.data.DataLoader):
    """解析数据加载器结构 / Parse dataloader structure
    
    调试工具函数，用于检查训练数据加载器的数据格式和形状。
    打印数据批次的详细信息后退出程序，主要用于开发和调试阶段。
    
    Debug utility function to inspect the data format and shapes from training dataloader.
    Prints detailed information about data batches and exits program, mainly used during
    development and debugging phases.
    
    Args:
        train_dataloader (torch.utils.data.DataLoader): 训练数据加载器 / Training data loader
    """
    print("---------------------  解析数据加载器 / parse_dataloader ---------------------")
    
    # 遍历数据加载器获取第一个批次 / Iterate through dataloader to get first batch
    for data in train_dataloader:
        """
        数据结构说明 / Data structure explanation:
        data[0] images: 图像数据，形状 batch_size * 3 * 3 * 480 * 640 / Image data, shape batch_size * 3 * 3 * 480 * 640
        data[2] qpos state: 关节位置状态，形状 batch_size * 14 / Joint position state, shape batch_size * 14  
        data[5] qpos actions: 关节位置动作序列，形状 batch_size * 112 * 14 / Joint position action sequence, shape batch_size * 112 * 14
        """   
        print(f"数据项数量 / Number of data items: {len(data)}")  # 数据项总数 / Total number of data items
        print(f"图像数据形状 / Image data shape: {data[0].shape}")  # 图像数据维度 / Image data dimensions
        print(f"状态数据形状 / State data shape: {data[2].shape}")  # 状态数据维度 / State data dimensions
        print(f"动作数据形状 / Action data shape: {data[5].shape}")  # 动作数据维度 / Action data dimensions
        break  # 只处理第一个批次 / Process only first batch
    
    print("---------------------  解析完成 / parsing completed ---------------------")
    exit(0)  # 退出程序用于调试 / Exit program for debugging purposes


# ========================================== 训练配置区域 / Training Configuration Section ==========================================
# ========================================== 以下为训练相关功能 / Below are training-related functions ==========================================
# ==========================================================================================================

def set_model_config(args, camera_names, len_train_dataloader):
    """设置模型配置参数 / Set model configuration parameters
    
    根据不同的策略类型和命令行参数，构建对应的策略配置字典。
    每种策略都有其特定的参数需求，该函数负责整合和组织这些参数。
    
    Build corresponding policy configuration dictionary based on different policy types 
    and command line arguments. Each policy has specific parameter requirements, 
    this function is responsible for integrating and organizing these parameters.
    
    Args:
        args: 命令行参数对象 / Command line arguments object
        camera_names (list): 相机名称列表 / Camera names list
        len_train_dataloader (int): 训练数据加载器长度 / Training dataloader length
        
    Returns:
        dict: 策略配置字典 / Policy configuration dictionary
    """
    
    # 根据策略类型设置固定参数 / Set fixed parameters based on policy type
    if args.policy_class == 'ACT':
        # ACT（Action Chunking Transformer）策略配置 / ACT (Action Chunking Transformer) policy configuration
        policy_config = {
            # 优化器参数 / Optimizer parameters
            'lr': args.lr,                                    # 主学习率 / Main learning rate
            'lr_backbone': args.lr_backbone,                  # 主干网络学习率 / Backbone learning rate
            'weight_decay': args.weight_decay,                # 权重衰减 / Weight decay
            'warmup_ratio': args.warmup_ratio,                # 学习率预热比例 / Learning rate warmup ratio
            'use_scheduler': args.scheduler,                  # 学习率调度器类型 / Learning rate scheduler type
            
            # 网络架构参数 / Network architecture parameters  
            'backbone': args.backbone,                        # 主干网络类型（如ResNet18）/ Backbone type (e.g., ResNet18)
            'hidden_dim': args.hidden_dim,                   # 隐藏层维度 / Hidden layer dimension
            'dim_feedforward': args.dim_feedforward,          # 前馈网络维度 / Feedforward network dimension
            'enc_layers': args.enc_layers,                    # 编码器层数 / Number of encoder layers
            'dec_layers': args.dec_layers,                    # 解码器层数 / Number of decoder layers
            'nheads': args.nheads,                           # 多头注意力头数 / Number of attention heads
            'dropout': args.dropout,                          # Dropout比例 / Dropout ratio
            'pre_norm': args.pre_norm,                        # 是否使用预归一化 / Whether to use pre-normalization
            
            # 位置编码和特征提取 / Position encoding and feature extraction
            'position_embedding': args.position_embedding,    # 位置编码类型（正弦/学习）/ Position encoding type (sine/learned)
            'dilation': args.dilation,                        # 是否使用膨胀卷积 / Whether to use dilated convolution
            'masks': args.masks,                              # 是否使用掩码 / Whether to use masks
            
            # 训练和数据参数 / Training and data parameters
            'loss_function': args.loss_function,              # 损失函数类型 / Loss function type
            'chunk_size': args.chunk_size,                    # 动作序列长度 / Action sequence length
            'kl_weight': args.kl_weight,                      # KL散度损失权重 / KL divergence loss weight
            'camera_names': camera_names,                     # 相机名称列表 / Camera names list
            'num_next_action': args.max_pos_lookahead,        # 前瞻动作数量 / Number of lookahead actions
            'use_depth_image': args.use_depth_image,          # 是否使用深度图像 / Whether to use depth images
            'use_robot_base': args.use_robot_base,            # 是否使用机器人底座 / Whether to use robot base
            
            # 系统和训练环境参数 / System and training environment parameters
            'epochs': args.num_epochs,                        # 训练轮数 / Training epochs
            'train_loader_len': len_train_dataloader,         # 训练数据加载器长度（用于调度器）/ Training dataloader length (for scheduler)
            'use_accelerate': args.use_accelerate,            # 是否使用加速库 / Whether to use accelerate library
            'device': args.device,                            # 计算设备 / Computing device
            'state_dim': args.state_dim,                      # 状态维度 / State dimension
            'action_dim': args.action_dim,                    # 动作维度 / Action dimension
            'policy_class': args.policy_class,                # 策略类名 / Policy class name
        }
    elif args.policy_class == 'CNNMLP' or args.policy_class == 'HistoryCNNMLP':
        # CNN-MLP策略配置：卷积神经网络+多层感知机 / CNN-MLP policy configuration: Convolutional Neural Network + Multi-Layer Perceptron
        policy_config = {
            # 基础训练参数 / Basic training parameters
            'lr': args.lr,                                    # 主学习率 / Main learning rate
            'lr_backbone': args.lr_backbone,                  # CNN主干网络学习率 / CNN backbone learning rate
            'epochs': args.num_epochs,                        # 训练轮数 / Training epochs
            'train_loader_len': len_train_dataloader,         # 训练数据加载器长度，约为seq_len/batch_size=100/16≈6 / Training dataloader length, approx seq_len/batch_size=100/16≈6
            'warmup_ratio': args.warmup_ratio,                # 学习率预热比例 / Learning rate warmup ratio
            'use_scheduler': args.scheduler,                  # 学习率调度器 / Learning rate scheduler
            'weight_decay': args.weight_decay,                # 权重衰减正则化 / Weight decay regularization
            
            # CNN视觉处理参数 / CNN visual processing parameters
            'backbone': args.backbone,                        # CNN主干网络架构 / CNN backbone architecture
            'masks': args.masks,                              # 是否使用掩码 / Whether to use masks
            'dilation': args.dilation,                        # 是否使用膨胀卷积 / Whether to use dilated convolution
            'position_embedding': args.position_embedding,    # 位置编码方案 / Position embedding scheme
            'hidden_dim': args.hidden_dim,                   # MLP隐藏层维度 / MLP hidden layer dimension
            
            # 策略特定参数 / Policy-specific parameters
            'loss_function': args.loss_function,              # 损失函数类型 / Loss function type
            'chunk_size': 1,                                  # 动作块大小（MLP每次预测单个动作）/ Action chunk size (MLP predicts single action each time)
            'camera_names': camera_names,                     # 相机配置 / Camera configuration
            'num_next_action': args.max_pos_lookahead,        # 前瞻动作数量 / Number of lookahead actions
            'use_depth_image': args.use_depth_image,          # 是否集成深度信息 / Whether to integrate depth information
            'use_robot_base': args.use_robot_base,            # 是否包含移动底座 / Whether to include mobile base
            
            # 系统配置 / System configuration
            'device': args.device,                            # 计算设备（CPU/GPU）/ Computing device (CPU/GPU)
            'state_dim': args.state_dim,                      # 机器人状态维度 / Robot state dimension
            'action_dim': args.action_dim,                    # 动作空间维度 / Action space dimension
            'policy_class': args.policy_class,                # 策略类别标识 / Policy class identifier
        }
    elif args.policy_class == 'CNNRNN' or args.policy_class == 'FPNRNN' or args.policy_class == 'RNN':
        # RNN系列策略配置：循环神经网络相关策略 / RNN-series policy configuration: Recurrent Neural Network-related policies
        # 包括CNNRNN（CNN+RNN混合）、FPNRNN（特征金字塔+RNN）、纯RNN策略 / Includes CNNRNN (CNN+RNN hybrid), FPNRNN (Feature Pyramid+RNN), pure RNN policy
        policy_config = {
            # 优化和训练参数 / Optimization and training parameters
            'lr': args.lr,                                    # 主学习率 / Main learning rate
            'lr_backbone': args.lr_backbone,                  # 视觉主干网络学习率 / Visual backbone learning rate
            'epochs': args.num_epochs,                        # 训练总轮数 / Total training epochs
            'train_loader_len': len_train_dataloader,         # 训练数据加载器长度，用于学习率调度 / Training dataloader length for learning rate scheduling
            'warmup_ratio': args.warmup_ratio,                # 学习率预热阶段比例 / Learning rate warmup phase ratio
            'use_scheduler': args.scheduler,                  # 学习率调度策略 / Learning rate scheduling strategy
            'weight_decay': args.weight_decay,                # 权重衰减防止过拟合 / Weight decay for overfitting prevention
            'gradient_accumulation_steps': args.gradient_accumulation_steps,  # 梯度累积步数 / Gradient accumulation steps
            
            # 视觉处理模块参数 / Visual processing module parameters
            'backbone': args.backbone,                        # 视觉主干网络（ResNet等）/ Visual backbone network (ResNet, etc.)
            'masks': args.masks,                              # 掩码处理开关 / Mask processing switch
            'dilation': args.dilation,                        # 膨胀卷积开关 / Dilated convolution switch
            'position_embedding': args.position_embedding,    # 位置编码类型 / Position embedding type
            'hidden_dim': args.hidden_dim,                   # 通用隐藏层维度 / General hidden layer dimension
            
            # RNN特定架构参数 / RNN-specific architecture parameters
            'rnn_layers': args.rnn_layers,                    # RNN（LSTM）层数 / Number of RNN (LSTM) layers
            'rnn_hidden_dim': args.rnn_hidden_dim,            # RNN隐藏状态维度 / RNN hidden state dimension
            'actor_hidden_dim': args.actor_hidden_dim,        # 动作网络隐藏层维度 / Actor network hidden layer dimension
            
            # 序列和动作处理 / Sequence and action processing
            'loss_function': args.loss_function,              # 损失函数选择 / Loss function selection
            'chunk_size': 1,                                  # RNN逐步预测，块大小为1 / RNN step-by-step prediction, chunk size is 1
            'camera_names': camera_names,                     # 多相机视觉输入配置 / Multi-camera visual input configuration
            'num_next_action': args.max_pos_lookahead,        # 前瞻动作序列长度 / Lookahead action sequence length
            'use_depth_image': args.use_depth_image,          # 深度图像融合开关 / Depth image fusion switch
            'use_robot_base': args.use_robot_base,            # 移动底座控制开关 / Mobile base control switch
            
            # 系统运行环境 / System runtime environment
            'device': args.device,                            # 计算设备分配 / Computing device allocation
            'state_dim': args.state_dim,                      # 机器人状态空间维度 / Robot state space dimension
            'action_dim': args.action_dim,                    # 机器人动作空间维度 / Robot action space dimension
            'policy_class': args.policy_class,                # 当前策略类型标识符 / Current policy type identifier
        }
    elif args.policy_class in ['DiffusionState']:
        policy_config = {
            'lr': args.lr,
            'lr_backbone': args.lr_backbone,
            'epochs': args.num_epochs,
            'train_loader_len': len_train_dataloader, # 6 \approx seq_len / batch_size = 100 / 16
            'warmup_ratio': args.warmup_ratio,
            'use_scheduler': args.scheduler,
            'backbone': args.backbone,
            'masks': args.masks,
            'weight_decay': args.weight_decay,
            'dilation': args.dilation,
            'position_embedding': args.position_embedding,
            'loss_function': args.loss_function,
            'chunk_size': 1,    
            'camera_names': camera_names,
            'num_next_action': args.max_pos_lookahead,
            'use_depth_image': args.use_depth_image,
            'use_robot_base': args.use_robot_base,
            'hidden_dim': args.hidden_dim,
            'device': args.device,
            'state_dim': args.state_dim,
            'action_dim': args.action_dim,
            'observation_horizon': args.observation_horizon,
            'action_horizon': args.action_horizon,
            'num_inference_timesteps': args.num_inference_timesteps,
            'ema_power': args.ema_power,
            # for DiffusionState
            'alpha': args.alpha,
            'max_time_steps': args.max_time_steps,
            'time_embed_dim': args.time_embed_dim,
            "context_len": args.context_len,
            'num_samples_per_traj': args.num_samples_per_traj,
            'policy_class': args.policy_class,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
    return policy_config
        
def train(args):
    set_seed(1)

    DATA_DIR = os.path.expanduser(args.dataset) 
    
    TASK_CONFIGS = {
        args.task_name: {
            # 'dataset_dir': DATA_DIR + args.task_name,
            'dataset_dir': DATA_DIR,
            'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
            'num_episodes': args.num_episodes,
        }
    }


    task_config = TASK_CONFIGS[args.task_name]
    cur_path = os.path.dirname(os.path.abspath(__file__))
    print(f"cur_path {cur_path}")
    print('num episodes', task_config['num_episodes'])
    
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    camera_names = task_config['camera_names']
    
    # ----------------------------- Begin of load data -----------------------------
    prefetch_factor = 2 if not args.use_accelerate else 2
    # 2 for single-gpu machine, 8 for multi-gpu machine
    if args.policy_class in ["RNN", "DiffusionState"]:
        episode_begin = 0
        episode_end = 90
        dataset_type = BBoxHistoryEpisodicDataset  # it will be preprocessed in load_data()
        prefetch_factor = 8
    elif "History" in args.policy_class:
        episode_begin = 3
        episode_end = 72
        dataset_type = HistoryEpisodicDataset
    else:
        episode_begin = 3
        episode_end = 90
        dataset_type = EpisodicDataset
    # 加载训练和验证数据 / Load training and validation data
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,              # 数据集目录 / Dataset directory
        num_episodes,             # 训练回合数 / Training episodes
        args.arm_delay_time,      # 机械臂延迟时间 / Robot arm delay time
        args.max_pos_lookahead,   # 最大位置前瞻 / Maximum position lookahead
        args.use_dataset_action,  # 是否使用数据集动作 / Whether to use dataset actions
        args.use_depth_image,     # 是否使用深度图像 / Whether to use depth images
        args.use_robot_base,      # 是否使用机器人底座 / Whether to use robot base
        camera_names,             # 相机名称列表 / Camera names list
        args.batch_size,          # 训练批次大小 / Training batch size
        args.batch_size,          # 验证批次大小 / Validation batch size
        episode_begin=episode_begin,  # 回合开始时间（避免苹果消失）/ Episode begin time (avoid apple disappearing)
        episode_end=episode_end,      # 回合结束时间 / Episode end time
        context_len=args.context_len, # 上下文长度 / Context length
        prefetch_factor=prefetch_factor,  # 预取因子 / Prefetch factor
        dataset_type=dataset_type     # 数据集类型 / Dataset type
    )
    # parse_dataloader(train_dataloader)  # 调试用：解析数据加载器结构 / For debugging: parse dataloader structure
    print(f'训练数据加载器长度 / Length of train dataloader: {len(train_dataloader)}')
    # ============================= 数据加载结束 / End of data loading =============================
    
    # 动态调整状态和动作维度 / Dynamically adjust state and action dimensions
    args.state_dim = args.state_dim if not args.use_robot_base else args.state_dim + 2  # 如果使用底座，增加2维（线速度+角速度）/ If using base, add 2 dims (linear+angular velocity)
    args.action_dim = args.state_dim  # 动作维度等于状态维度 / Action dimension equals state dimension

    # 设置策略配置参数 / Set policy configuration parameters
    policy_config = set_model_config(args, camera_names, len(train_dataloader))
    
    # 创建检查点保存目录 / Create checkpoint save directory
    original_ckpt_dir = args.ckpt_dir  # 保存原始检查点目录 / Save original checkpoint directory
    args.ckpt_dir = os.path.join(args.ckpt_dir, timestamp + args.policy_class)  # 创建带时间戳的新目录 / Create new directory with timestamp
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)  # 创建目录 / Create directory
    
    # 预训练模型路径设置 / Pre-trained model path setup
    if len(args.pretrain_timestamp) != 0:
        # 如果指定了预训练时间戳，构建预训练模型路径 / If pre-train timestamp specified, build pre-train model path
        pretrain_ckpt_dir = os.path.join(original_ckpt_dir, args.pretrain_timestamp)
    else:
        pretrain_ckpt_dir = ''  # 没有预训练模型 / No pre-trained model
    
    # if you load the old model, then it will cover the old policy config
    # if len(pretrain_timestamp) != 0:
    #     config_file = os.path.join(original_ckpt_dir, args.pretrain_timestamp, "config.json")
    #     with open(config_file, 'r') as f:
    #         config_json = f.read()
    #     config = json.loads(config_json)
    #     policy_config = config["policy_config"]
    #     args.policy_class = 
    #     print("")
    
    # 构建完整的训练配置字典 / Build complete training configuration dictionary
    config = {
        'num_epochs': args.num_epochs,                        # 训练轮数 / Number of training epochs
        'num_episodes': args.num_episodes,                    # 训练回合数 / Number of training episodes
        'ckpt_stats_name': args.ckpt_stats_name,              # 统计数据文件名 / Statistics file name
        'use_dataset_action': args.use_dataset_action,        # 是否使用数据集动作 / Whether to use dataset actions
        'ckpt_dir': args.ckpt_dir,                            # 检查点保存目录 / Checkpoint save directory
        'policy_class': args.policy_class,                    # 策略类名 / Policy class name
        'policy_config': policy_config,                       # 策略配置参数 / Policy configuration parameters
        'seed': args.seed,                                    # 随机种子 / Random seed
        'pretrain_timestamp': args.pretrain_timestamp,        # 预训练模型时间戳 / Pre-trained model timestamp
        'pretrain_ckpt_dir': pretrain_ckpt_dir,               # 预训练模型目录 / Pre-trained model directory
        'num_eval_step': args.num_eval_step,                  # 验证步数 / Number of evaluation steps
        'num_train_step': args.num_train_step,                # 训练步数 / Number of training steps
        'use_scheduler': args.scheduler,                      # 是否使用学习率调度器 / Whether to use learning rate scheduler
        'use_accelerate': args.use_accelerate,                # 是否使用加速库 / Whether to use accelerate library
        'device': args.device,                                # 计算设备 / Computing device
        'aug': args.aug,                                      # 数据增强设置 / Data augmentation settings
    }
    # 预训练模型配置加载 / Pre-trained model configuration loading
    if len(args.pretrain_timestamp) != 0:
        # 从旧模型加载策略配置 / Load policy config from old model as well
        config_file = os.path.join(args.ckpt_dir, 'config.json')  # 配置文件路径 / Config file path
        with open(config_file, 'r') as f:
            config_json = f.read()  # 读取JSON配置 / Read JSON configuration
        config['policy_config'] = json.loads(config_json)['policy_config']  # 更新策略配置 / Update policy configuration
        print("RESUME! policy config covered.")
    
    # 检查点目录创建 / Checkpoint directory creation
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)  # 创建检查点目录 / Create checkpoint directory
    
    # 保存数据集统计信息 / Save dataset statistics
    stats_path = os.path.join(args.ckpt_dir, args.ckpt_stats_name)
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)  # 序列化保存统计数据 / Serialize and save statistics data

    # 保存训练配置 / Save training configuration
    config_json = json.dumps(config, indent=4)  # 格式化JSON配置 / Format JSON configuration
    with open(os.path.join(args.ckpt_dir, "config.json"), 'w') as f:
        f.write(config_json)  # 写入配置文件 / Write configuration file

    # 启动训练进程 / Start training process
    best_ckpt_info = train_process(train_dataloader, val_dataloader, config, stats)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info  # 解包最佳检查点信息 / Unpack best checkpoint info

    # 复制训练日志文件 / Copy training log file
    source_path = os.path.join(os.getcwd(), 'train.log')  # 源日志文件路径 / Source log file path
    destination_path = os.path.join(args.ckpt_dir, "train.log")  # 目标日志文件路径 / Destination log file path
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)  # 复制日志文件 / Copy log file
        print(f"train.log File copied to {destination_path}")
    else:
        print(f"Source file '{source_path}' does not exist.")
    
    # 保存最佳检查点（已经保存！）/ Save best checkpoint (have been saved!)
    # ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    # torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')  # 输出最佳检查点信息 / Output best checkpoint info


def train_process(train_dataloader: DataLoader, val_dataloader: DataLoader, config, stats):
    """训练进程主函数 / Main training process function
    
    Args:
        train_dataloader: 训练数据加载器 / Training data loader
        val_dataloader: 验证数据加载器 / Validation data loader
        config: 训练配置字典 / Training configuration dictionary
        stats: 数据集统计信息 / Dataset statistics
    
    Returns:
        tuple: (best_epoch, min_val_loss, best_state_dict) 最佳检查点信息 / Best checkpoint info
    """
    # 数据后处理函数定义 / Data post-processing function definition
    if config['use_dataset_action']:
        # 使用动作数据的反标准化 / Denormalization for action data
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    else:
        # 使用关节位置数据的反标准化 / Denormalization for joint position data
        post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']

    # 配置参数解析 / Configuration parameter parsing
    num_epochs = config['num_epochs']                    # 训练轮数 / Number of epochs
    ckpt_dir = config['ckpt_dir']                        # 检查点目录 / Checkpoint directory
    seed = config['seed']                                # 随机种子 / Random seed
    policy_class = config['policy_class']                # 策略类名 / Policy class name
    policy_config = config['policy_config']              # 策略配置 / Policy configuration
    pretrain_ckpt_dir = config['pretrain_ckpt_dir']      # 预训练模型目录 / Pre-trained model directory
    num_eval_step = config['num_eval_step']              # 验证步数 / Number of evaluation steps
    num_train_step = config['num_train_step']            # 训练步数 / Number of training steps
    set_seed(seed)  # 设置随机种子 / Set random seed

    # 策略模型初始化 / Policy model initialization
    policy = make_policy(policy_class, policy_config, pretrain_ckpt_dir)
    # policy = torch.load('/home/ycy17/Desktop/code/low-level-ACT/aloha-devel/act/ckpt_old4/policy_epoch_9740_seed_0.ckpt')
    optimizer = policy.optimizer      # 优化器 / Optimizer
    
    scheduler = policy.scheduler     # 学习率调度器 / Learning rate scheduler

    
    # Accelerate库集成设置 / Accelerate library integration setup
    if config['use_accelerate']:
        if config['use_scheduler'] == 'cos':
            # 使用余弦调度器的分布式训练准备 / Distributed training preparation with cosine scheduler
            policy, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(policy, optimizer, scheduler, train_dataloader, val_dataloader)
        elif config['use_scheduler'] == 'none':
            # 不使用调度器的分布式训练准备 / Distributed training preparation without scheduler
            policy, optimizer, train_dataloader, val_dataloader = accelerator.prepare(policy, optimizer, train_dataloader, val_dataloader)
        else:
            raise NotImplementedError  # 未实现的调度器类型 / Unimplemented scheduler type
    else:
        # 单设备训练，将模型移至指定设备 / Single device training, move model to specified device
        policy = policy.to(device=config['device'])
    # scaler = GradScaler()  # 梯度缩放器（已注释）/ Gradient scaler (commented)

    # 训练状态和历史记录初始化 / Training state and history initialization
    train_history = []           # 训练损失历史 / Training loss history
    validation_history = []      # 验证损失历史 / Validation loss history
    min_val_loss = np.inf        # 最小验证损失初始化 / Minimum validation loss initialization
    best_ckpt_info = (0, np.inf, None)  # 最佳检查点信息 / Best checkpoint info
    print_interval = 1           # 打印间隔 / Print interval
    eval_interval = 1            # 验证间隔 / Evaluation interval
    save_interval = 20           # 保存间隔 / Save interval
    
    # 训练循环开始 / Training loop start
    import time
    start_time = time.time()  # 记录开始时间 / Record start time
    for epoch in tqdm(range(num_epochs)):
        # TODO: save file with timestamps, including png, ckpt, config, fewer tqdm info
        train_dataloader.dataset.shuffle()  # 训练数据集随机打乱 / Shuffle training dataset
        val_dataloader.dataset.shuffle()    # 验证数据集随机打乱 / Shuffle validation dataset
        # ---------------------------- 训练阶段 / Training Phase ----------------------------
        policy.train()  # 设置模型为训练模式 / Set model to training mode
        # 第一个epoch显示详细进度条，后续epoch简化显示 / Detailed progress bar for first epoch, simplified for others
        iterator = (tqdm(enumerate(train_dataloader), mininterval=10, leave=False)
                    if epoch == 0 else
                    enumerate(train_dataloader))
        for batch_idx, data in iterator:
            """
            数据批次格式说明 / Batch data format explanation:
            data[0] images: 图像数据 batch_size * context_len * 3 * 3 * 480 * 640
            data[2] qpos state: 关节位置状态 batch_size * context_len * 14
            data[5] qpos actions: 关节位置动作 batch_size * context_len * 14
            """   
            # print(len(data), data[0].shape)  # 调试：打印数据维度 / Debug: print data dimensions
            # print(len(data), data[2].shape)
            # print(len(data), data[5].shape)
            # exit(0)
            
            # 数据设备转移 / Data device transfer
            if not config['use_accelerate']:
                # stream = torch.cuda.Stream()  # CUDA流（已注释）/ CUDA stream (commented)
                # to_t1 = time.time()  # 计时开始 / Timing start
                # data = [d.to(dtype=torch.float16) for d in data]  # 半精度转换（已注释）/ Half precision conversion (commented)
                # with torch.cuda.stream(stream):  # 异步流（已注释）/ Async stream (commented)
                data = [d.to(device=config['device'], non_blocking=True) for d in data]  # 数据转移到GPU，非阻塞 / Transfer data to GPU, non-blocking
                # CPU到CUDA的数据复制较慢（原始代码每批0.8秒）/ CPU to CUDA data copy is slow (0.8s per batch in raw code)
                # data = [d.to(dtype=torch.float32) for d in data]  # 单精度转换（已注释）/ Single precision conversion (commented)
                # print(f'to device time: {time.time() - to_t1} s')  # 打印转移时间 / Print transfer time
                
            # 数据增强处理 / Data augmentation processing
            # import pdb  # Python调试器（已注释）/ Python debugger (commented)
            # pdb.set_trace()  # 设置断点（已注释）/ Set breakpoint (commented)
            if config['aug'] == 'distort':
                data[0] = distort_image(data[0])  # 应用图像扭曲增强 / Apply image distortion augmentation
            elif config['aug'] == None:
                # print('we do not aug')  # 不使用数据增强 / No data augmentation
                pass
            else:
                raise NotImplementedError  # 未实现的增强方式 / Unimplemented augmentation method
            
            # 前向传播和损失计算 / Forward pass and loss computation
            # with autocast(dtype=torch.bfloat16):  # 混合精度训练（已注释）/ Mixed precision training (commented)
            with accelerator.accumulate(policy):  # 梯度累积上下文 / Gradient accumulation context
                forward_dict, result = forward_pass(policy_config, data, policy)  # 执行前向传播 / Execute forward pass
                
                # print("result:", post_process(result.cpu().detach().numpy())[0, :, 7:])  # 调试：打印结果 / Debug: print result
                # 反向传播 / Backward propagation
                loss = forward_dict['loss']  # 提取损失值 / Extract loss value
                # 反向传播和梯度更新 / Backward propagation and gradient update
                if config['use_accelerate']:
                    accelerator.backward(loss)  # 使用Accelerate库的反向传播 / Backward propagation using Accelerate
                    # accelerator.backward(scaler.scale(loss))  # 梯度缩放（已注释）/ Gradient scaling (commented)
                    # scaler.step(optimizer)  # 优化器步骤（已注释）/ Optimizer step (commented)
                    # scaler.update()  # 缩放器更新（已注释）/ Scaler update (commented)
                else:
                    loss.backward()  # 标准反向传播 / Standard backward propagation
                    # scaler.scale(loss).backward()  # 缩放损失反向传播（已注释）/ Scaled loss backward (commented)
                    
                    # 注意：不要删除这些行，用于调试 / NOTE: don't delete these lines, for debugging
                    # 打印和比较梯度（视觉编码器 vs. 时序模型）/ print and compare grad (visual encoder vs. temporal model)
                    # visual_encoder_grad = []
                    # temporal_model_grad = []
                    # for name, param in policy.model.named_parameters():
                    #     if param.requires_grad and 'backbone' in name and param.grad != None:
                    #         print(f"visual encoder. Gradient for {name}: {param.grad.mean()}")
                    #         visual_encoder_grad.append(param.grad.mean().item())
                    #     elif param.requires_grad and 'backbone' not in name and param.grad != None:
                    #         print(f"temporal model. Gradient for {name}: {param.grad.mean()}")
                    #         temporal_model_grad.append(param.grad.mean().item())
                    # print(f"visual encoder grad: {np.mean(visual_encoder_grad)}")
                    # print(f"temporal model grad: {np.mean(temporal_model_grad)}")
                    # scaler.step(optimizer)  # 缩放器优化步骤（已注释）/ Scaler optimizer step (commented)
                    # scaler.update()  # 缩放器更新（已注释）/ Scaler update (commented)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)  # 梯度裁剪防止梯度爆炸 / Gradient clipping to prevent gradient explosion

                optimizer.step()  # 执行优化器步骤 / Execute optimizer step
                train_history.append(detach_dict(forward_dict))  # 记录训练历史 / Record training history
                if scheduler != None:
                    scheduler.step()  # 更新学习率调度器 / Update learning rate scheduler
                optimizer.zero_grad()  # 清零梯度 / Zero gradients
                
            
        # 计算epoch统计信息 / Compute epoch statistics
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])  # 计算该epoch的平均损失 / Compute average loss for this epoch
        epoch_train_loss = epoch_summary['loss']  # 提取epoch训练损失 / Extract epoch training loss
        
        # 打印训练信息 / Print training information
        if epoch % print_interval == 0:
            print(f'\nEpoch {epoch}, lr:', optimizer.param_groups[0]['lr'])  # 打印epoch和学习率（这里是lr，不是lr_backbone）/ Print epoch and learning rate (this lr means lr, not lr_backbone)
            print(f'Train loss: {epoch_train_loss:.5f}')  # 打印训练损失 / Print training loss
            summary_string = ''
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '  # 构建统计信息字符串 / Build statistics summary string
            print(summary_string)  # 打印统计信息 / Print statistics summary
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)  # 绘制训练曲线 / Plot training curves

        # 定期保存检查点（已注释）/ Periodic checkpoint saving (commented)
        # if epoch % save_interval == 0:
        #     ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
        #     if config['use_accelerate']:
        #         torch.save(policy, ckpt_path)  # 保存整个模型 / Save entire model
        #     else:
        #         torch.save(policy.state_dict(), ckpt_path)  # 保存模型状态字典 / Save model state dict
        
        # ---------------------------- 验证阶段 / Validation Phase ----------------------------
        # TODO: ACT
        if epoch % eval_interval == 0:  # 每隔eval_interval个epoch进行验证 / Validate every eval_interval epochs
            with torch.inference_mode():  # 推理模式，禁用梯度计算 / Inference mode, disable gradient computation
                policy.eval()  # 设置模型为评估模式 / Set model to evaluation mode
                epoch_dicts = []  # 存储所有batch的验证结果 / Store validation results from all batches
                for batch_idx, data in enumerate(val_dataloader):
                    if not config['use_accelerate']:
                        data = [d.to(device=config['device']) for d in data]  # 验证数据转移到GPU / Transfer validation data to GPU
                    forward_dict, result = forward_pass(policy_config, data, policy)  # 验证前向传播 / Validation forward pass
                    # print("result:", post_process(result.cpu().detach().numpy())[0, :, 7:])  # 调试：打印验证结果 / Debug: print validation result
                    epoch_dicts.append(forward_dict)  # 收集验证结果 / Collect validation results
                    # if batch_idx >= num_eval_step:  # 限制验证步数（已注释）/ Limit validation steps (commented)
                    #     break
                epoch_summary = compute_dict_mean(epoch_dicts)  # 计算验证统计 / Compute validation statistics
                validation_history.append(epoch_summary)  # 添加到验证历史 / Add to validation history

                epoch_val_loss = epoch_summary['loss']  # 获取epoch验证损失 / Get epoch validation loss
                
                accelerator.wait_for_everyone()  # 等待所有进程同步 / Wait for all processes to synchronize
                # 检查是否为最佳模型 / Check if this is the best model
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss  # 更新最小验证损失 / Update minimum validation loss
                    best_path = os.path.join(ckpt_dir, f'policy_best.ckpt')  # 最佳模型保存路径 / Best model save path
                    if config['use_accelerate']:
                        best_ckpt_info = (epoch, min_val_loss, deepcopy(policy))  # 保存最佳检查点信息 / Save best checkpoint info
                        if accelerator.is_main_process:  # 仅在主进程中保存 / Save only in main process
                            unwrapped_model = accelerator.unwrap_model(policy)  # 解包模型 / Unwrap model
                            torch.save(unwrapped_model.state_dict(), best_path)  # 保存模型状态 / Save model state
                            # 解包模型，否则加载时会出现`Missing 'module.' key(s) in state_dict`错误 / unwrap model, otherwise it will occur `Missing 'module.' key(s) in state_dict` error when loading model
                            # torch.save(policy.state_dict(), best_path)  # 直接保存（已注释）/ Direct save (commented)
                    else:
                        best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))  # 单设备最佳检查点信息 / Single device best checkpoint info
                        torch.save(policy.state_dict(), best_path)  # 保存模型状态字典 / Save model state dict
                    print(f'Best ckpt saved, val loss {min_val_loss:.6f} @ epoch{best_ckpt_info[0]}')  # 打印最佳模型信息 / Print best model info
            accelerator.wait_for_everyone()  # 确保所有进程在继续之前同步 / Ensure all processes sync up before continuing
            
            # print('time 2', time.time() - start_time)  # 计时调试（已注释）/ Timing debug (commented)
            # print(f'Val loss:   {epoch_val_loss:.5f}')  # 简单验证损失打印（已注释）/ Simple validation loss print (commented)
            print(f'Val loss:   {epoch_val_loss:.5f}.   Best val loss: {min_val_loss:.5f} at epoch {best_ckpt_info[0]}')  # 打印当前和最佳验证损失 / Print current and best validation loss

            # 构建和打印验证统计信息 / Build and print validation statistics
            summary_string = ''
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '  # 格式化统计信息 / Format statistics info
            print(summary_string)  # 打印统计摘要 / Print statistics summary

    # 保存最后一个epoch的模型（已注释）/ Save model from last epoch (commented)
    # ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    # if config['use_accelerate']:
    #     if accelerator.is_main_process:
    #         unwrapped_model = accelerator.unwrap_model(policy)
    #         torch.save(unwrapped_model.state_dict(), ckpt_path)
    #         # torch.save(policy.state_dict(), ckpt_path)
    # else:
    #     torch.save(policy.state_dict(), ckpt_path)
    accelerator.wait_for_everyone()  # 最终同步所有进程 / Final synchronization of all processes
    
    # 训练结束后处理 / Post-training processing
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info  # 解包最佳检查点信息 / Unpack best checkpoint info
    # ckpt_path = os.path.join(ckpt_dir, f'{min_val_loss:.4f}_policy_epoch_{best_epoch}_seed_{seed}.ckpt')  # 带损失值的文件名（已注释）/ Filename with loss value (commented)
    # torch.save(best_state_dict, ckpt_path)  # 保存最佳模型（已注释）/ Save best model (commented)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')  # 打印训练结束信息 / Print training completion info

    # 保存训练曲线 / Save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info  # 返回最佳检查点信息 / Return best checkpoint info


def forward_pass(policy_config, data, policy):
    image_data, image_depth_data, qpos_data, next_action_data, next_action_is_pad, action_data, action_is_pad = data
    
    if policy_config['use_depth_image']:
        image_depth_data = image_depth_data
    else:
        image_depth_data = None
    if policy_config['num_next_action'] != 0:
        return policy(image_data, image_depth_data, qpos_data, next_action_data, next_action_is_pad, action_data, action_is_pad)
    else:  # this branch
        return policy(image_data, image_depth_data, qpos_data, None, None, action_data, action_is_pad)


def main():
    print('scheduler:', args.scheduler, "args.gradient_accumulation_steps", args.gradient_accumulation_steps)
    print('whether use acclerator:', args.use_accelerate)
    train(args)


if __name__ == '__main__':
    main()

