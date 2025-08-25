"""
data_load.py

数据加载模块，提供机器人学习数据集的加载、预处理和数据加载器创建功能。
该模块包括数据归一化统计计算、训练验证集分割、不同类型数据集的创建等功能。

Data loading module providing dataset loading, preprocessing, and dataloader creation
functionalities for robotics learning datasets. This module includes data normalization
statistics calculation, train/validation split, and creation of different dataset types.
"""

# 数值计算和机器学习库 / Numerical computing and ML libraries
import numpy as np  # 数值计算库 / Numerical computing library
import torch  # PyTorch深度学习框架 / PyTorch deep learning framework

# 文件和数据处理库 / File and data processing libraries
import os  # 操作系统接口 / Operating system interface
import h5py  # HDF5文件格式处理 / HDF5 file format handling
import random  # 随机数生成 / Random number generation

# PyTorch数据处理 / PyTorch data handling
from torch.utils.data import TensorDataset, DataLoader  # PyTorch数据加载工具 / PyTorch data loading utilities

# 图像处理库 / Image processing libraries
import cv2  # OpenCV计算机视觉库 / OpenCV computer vision library
from PIL import Image  # Python图像库 / Python Imaging Library

# 调试工具 / Debugging tools
import IPython  # 交互式Python环境 / Interactive Python environment
e = IPython.embed  # 调试嵌入函数的快捷方式 / Shortcut for debug embedding function

# 项目特定导入 / Project-specific imports
import ManiBox  # ManiBox项目主模块 / ManiBox main project module
# from ManiBox.utils import get_norm_stats  # 原始归一化统计工具（已注释）/ Original normalization statistics utility (commented)

# 数据集类导入 / Dataset class imports
from ManiBox.dataloader.EpisodicDataset import EpisodicDataset  # 基础情节数据集类 / Base episodic dataset class
from ManiBox.dataloader.HistoryEpisodicDataset import HistoryEpisodicDataset  # 历史情节数据集类 / Historical episodic dataset class
from ManiBox.dataloader.BBoxHistoryEpisodicDataset import BBoxHistoryEpisodicDataset  # 边界框历史数据集类 / Bounding box historical dataset class


def get_norm_stats(dataset_dir, num_episodes, episode_begin, episode_end):
    """
    计算数据集的归一化统计信息
    Calculate normalization statistics for the dataset
    
    从预处理的积分数据文件中计算关节位置和动作数据的均值和标准差，
    用于Z-score归一化。这有助于稳定神经网络训练过程。
    
    Calculate mean and standard deviation for joint position and action data from
    preprocessed integration data file for Z-score normalization. This helps stabilize
    neural network training process.
    
    Args:
        dataset_dir: 数据集目录路径 / Dataset directory path
        num_episodes: 使用的情节数量 / Number of episodes to use
        episode_begin: 情节开始索引 / Episode begin index
        episode_end: 情节结束索引 / Episode end index
        
    Returns:
        dict: 包含各类数据的均值和标准差的字典 / Dictionary containing mean and std for different data types
    """
    # 加载预处理的积分数据文件 / Load preprocessed integration data file
    integration_path = os.path.join(dataset_dir, "integration.pkl")
    data = torch.load(integration_path, map_location='cpu')  # 加载数据到CPU内存 / Load data to CPU memory
    
    # 计算关节位置数据的归一化统计信息 / Calculate normalization statistics for joint position data
    # 对所有指定情节和时间步范围的qpos_data进行归一化 / Normalize all qpos_data in specified episodes and timestep range
    qpos_subset = data["qpos_data"][:num_episodes, episode_begin:episode_end]  # 提取指定子集 / Extract specified subset
    qpos_mean = qpos_subset.mean(dim=[0, 1], keepdim=True)  # 计算均值（跨情节和时间步维度）/ Calculate mean (across episodes and timesteps dimensions)
    qpos_std = qpos_subset.std(dim=[0, 1], keepdim=True)  # 计算标准差 / Calculate standard deviation
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # 限制最小标准差以避免除以零 / Clip minimum std to avoid division by zero
    
    # 计算动作数据的归一化统计信息 / Calculate normalization statistics for action data
    action_subset = data["action_data"][:num_episodes, episode_begin:episode_end]  # 提取动作数据子集 / Extract action data subset
    action_mean = action_subset.mean(dim=[0, 1], keepdim=True)  # 计算动作均值 / Calculate action mean
    action_std = action_subset.std(dim=[0, 1], keepdim=True)  # 计算动作标准差 / Calculate action standard deviation
    action_std = torch.clip(action_std, 1e-2, np.inf)  # 限制最小标准差 / Clip minimum std
    
    # 对原始数据进行就地归一化（可选）/ In-place normalization of original data (optional)
    data["qpos_data"] = (data["qpos_data"] - qpos_mean) / qpos_std  # Z-score归一化关节位置 / Z-score normalize joint positions
    data["action_data"] = (data["action_data"] - action_mean) / action_std  # Z-score归一化动作 / Z-score normalize actions
    
    # 构建归一化统计信息字典 / Build normalization statistics dictionary
    norm_stats = {
        "action_mean": action_mean.numpy().squeeze(),  # 动作均值（转为numpy并去除单一维度）/ Action mean (convert to numpy and squeeze)
        "action_std": action_std.numpy().squeeze(),    # 动作标准差 / Action standard deviation
        "qpos_mean": qpos_mean.numpy().squeeze(),      # 关节位置均值 / Joint position mean
        "qpos_std": qpos_std.numpy().squeeze()        # 关节位置标准差 / Joint position standard deviation
    }
    
    return norm_stats  # 返回归一化统计信息 / Return normalization statistics


def load_data(dataset_dir, num_episodes, arm_delay_time, max_pos_lookahead, use_dataset_action, 
              use_depth_image, use_robot_base, camera_names, batch_size_train, batch_size_val, episode_begin=0, episode_end=-1,
              context_len=1, prefetch_factor=2, dataset_type=HistoryEpisodicDataset):
    """
    主数据加载函数，创建训练和验证数据加载器
    Main data loading function to create training and validation dataloaders
    
    该函数负责整个数据加载流程，包括训练验证集分割、归一化统计计算、
    数据集创建和数据加载器配置等。支持多种数据集类型和灵活的参数配置。
    
    This function handles the entire data loading pipeline including train/validation split,
    normalization statistics calculation, dataset creation, and dataloader configuration.
    Supports multiple dataset types and flexible parameter configuration.
    
    Args:
        dataset_dir: 数据集目录路径 / Dataset directory path
        num_episodes: 总情节数量 / Total number of episodes
        arm_delay_time: 机械臂延迟时间 / Arm delay time
        max_pos_lookahead: 最大位置前瞻步数 / Maximum position lookahead steps
        use_dataset_action: 是否使用数据集动作 / Whether to use dataset actions
        use_depth_image: 是否使用深度图像 / Whether to use depth images
        use_robot_base: 是否使用机器人底座 / Whether to use robot base
        camera_names: 相机名称列表 / List of camera names
        batch_size_train: 训练批大小 / Training batch size
        batch_size_val: 验证批大小 / Validation batch size
        episode_begin: 情节开始索引 / Episode begin index
        episode_end: 情节结束索引 / Episode end index
        context_len: 上下文长度（历史数据集使用）/ Context length (for historical datasets)
        prefetch_factor: 数据预取因子 / Data prefetch factor
        dataset_type: 数据集类型 / Dataset type class
        
    Returns:
        tuple: (train_dataloader, val_dataloader, norm_stats, is_sim)
            - train_dataloader: 训练数据加载器 / Training dataloader
            - val_dataloader: 验证数据加载器 / Validation dataloader
            - norm_stats: 归一化统计信息 / Normalization statistics
            - is_sim: 是否为仿真数据 / Whether data is from simulation
    """
    print(f'\nData from: {dataset_dir}\n')  # 打印数据源信息 / Print data source information
    
    # 获取训练验证集分割 / Obtain train-validation split
    train_ratio = 0.9  # 训练集比例，默认0.8，这里设为0.9 / Training set ratio, default 0.8, set to 0.9 here
    shuffled_indices = np.random.permutation(num_episodes)  # 随机打乱情节索引 / Randomly shuffle episode indices

    # TODO: 调试时只使用1个数据集，同时用于训练和测试 / TODO: For debugging, use single dataset for both training and testing
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]  # 获取训练集索引 / Get training set indices
    # print(f"train_indices {train_indices}")  # 调试：打印训练集索引 / Debug: print training indices
    
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]  # 获取验证集索引 / Get validation set indices
    # val_indices = shuffled_indices[:int(train_ratio * num_episodes)]  # 替代方案（已注释）/ Alternative approach (commented)

    # 初始化归一化统计信息 / Initialize normalization statistics
    norm_stats = None
    
    # 获取qpos和action的归一化统计信息 / Obtain normalization statistics for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes, episode_begin, episode_end)

    # 构建数据集和数据加载器 / Construct dataset and dataloader
    
    # 原始数据集创建方案（已注释，基于上下文长度选择数据集类型）/ Original dataset creation scheme (commented, select dataset type based on context length)
    # if context_len == 1:  # 如果上下文长度为1，使用基础情节数据集 / If context length is 1, use basic episodic dataset
    #     train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, arm_delay_time,
    #                                     max_pos_lookahead, use_dataset_action, use_depth_image, use_robot_base,
    #                                     episode_begin, episode_end)  # 创建训练数据集 / Create training dataset
    #
    #     val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, arm_delay_time, 
    #                                 max_pos_lookahead, use_dataset_action, use_depth_image, use_robot_base,
    #                                 episode_begin, episode_end)  # 创建验证数据集 / Create validation dataset
    # else:  # 否则使用历史情节数据集 / Otherwise use historical episodic dataset
    #     train_dataset = HistoryEpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, arm_delay_time,
    #                                     max_pos_lookahead, use_dataset_action, use_depth_image, use_robot_base,
    #                                     episode_begin, episode_end)  # 创建历史训练数据集 / Create historical training dataset
    #
    #     val_dataset = HistoryEpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, arm_delay_time, 
    #                                 max_pos_lookahead, use_dataset_action, use_depth_image, use_robot_base,
    #                                 episode_begin, episode_end)  # 创建历史验证数据集 / Create historical validation dataset
    
    # 特殊处理：边界框历史数据集的预处理 / Special handling: preprocessing for bounding box historical dataset
    if dataset_type is BBoxHistoryEpisodicDataset:  # 如果使用边界框历史数据集 / If using bounding box historical dataset
        integration_file = os.path.join(dataset_dir, "integration.pkl")  # 积分数据文件路径 / Integration data file path
        
        if not os.path.exists(integration_file):  # 如果积分数据文件不存在 / If integration data file doesn't exist
            # 动态导入YOLO数据处理模块 / Dynamically import YOLO data processing module
            from yolo_process_data import ProcessDataFromHDF5
            
            # 创建数据处理器实例 / Create data processor instance
            process_data = ProcessDataFromHDF5(shuffled_indices, dataset_dir, camera_names, norm_stats, arm_delay_time,
                                    max_pos_lookahead, use_dataset_action, use_depth_image, use_robot_base,
                                    episode_begin, episode_end)
            
            # 执行数据预处理（将HDF5数据转换为YOLO边界框格式）/ Execute data preprocessing (convert HDF5 data to YOLO bbox format)
            process_data.process_data()  # TODO: 可能需要进一步优化 / May need further optimization
    
    # 使用指定的数据集类型创建训练数据集 / Create training dataset using specified dataset type
    train_dataset = dataset_type(
        train_indices,           # 训练集情节索引 / Training set episode indices
        dataset_dir,            # 数据集目录 / Dataset directory
        camera_names,           # 相机名称列表 / Camera names list
        norm_stats,             # 归一化统计信息 / Normalization statistics
        arm_delay_time,         # 机械臂延迟时间 / Arm delay time
        max_pos_lookahead,      # 最大前瞻步数 / Maximum lookahead steps
        use_dataset_action,     # 是否使用数据集动作 / Whether to use dataset actions
        use_depth_image,        # 是否使用深度图像 / Whether to use depth images
        use_robot_base,         # 是否使用机器人底座 / Whether to use robot base
        episode_begin,          # 情节开始索引 / Episode begin index
        episode_end,            # 情节结束索引 / Episode end index
        random_mask_ratio=0.3   # 随机遮蔽比例（数据增强，仅训练集使用）/ Random masking ratio (data augmentation, training only)
    )

    # 创建验证数据集（不使用数据增强）/ Create validation dataset (without data augmentation)
    val_dataset = dataset_type(
        val_indices,            # 验证集情节索引 / Validation set episode indices
        dataset_dir,            # 数据集目录 / Dataset directory
        camera_names,           # 相机名称列表 / Camera names list
        norm_stats,             # 归一化统计信息 / Normalization statistics
        arm_delay_time,         # 机械臂延迟时间 / Arm delay time
        max_pos_lookahead,      # 最大前瞻步数 / Maximum lookahead steps
        use_dataset_action,     # 是否使用数据集动作 / Whether to use dataset actions
        use_depth_image,        # 是否使用深度图像 / Whether to use depth images
        use_robot_base,         # 是否使用机器人底座 / Whether to use robot base
        episode_begin,          # 情节开始索引 / Episode begin index
        episode_end,            # 情节结束索引 / Episode end index
        random_mask_ratio=0     # 不使用随机遮蔽（验证集保持稳定）/ No random masking (validation set kept stable)
    )
    
    # 调试断点（已注释）/ Debug breakpoint (commented)
    # import pdb  # Python调试器 / Python debugger
    # pdb.set_trace()  # 设置断点 / Set breakpoint

    # 创建训练数据加载器 / Create training dataloader
    train_dataloader = DataLoader(
        train_dataset,                    # 训练数据集 / Training dataset
        batch_size=batch_size_train,      # 训练批大小 / Training batch size
        shuffle=True,                     # 每个epoch打乱数据 / Shuffle data every epoch
        pin_memory=True,                  # 锁定内存以加速GPU传输 / Pin memory for faster GPU transfer
        num_workers=8,                    # 并行数据加载进程数 / Number of parallel data loading processes
        prefetch_factor=prefetch_factor   # 每个worker预取的批数 / Number of batches prefetched by each worker
    )

    # 创建验证数据加载器 / Create validation dataloader
    val_dataloader = DataLoader(
        val_dataset,                      # 验证数据集 / Validation dataset
        batch_size=batch_size_val,        # 验证批大小 / Validation batch size
        shuffle=True,                     # 验证时也打乱数据 / Also shuffle validation data
        pin_memory=True,                  # 锁定内存 / Pin memory
        num_workers=8,                    # 并行加载进程数 / Number of parallel loading processes
        prefetch_factor=prefetch_factor   # 预取因子 / Prefetch factor
    )

    # 返回数据加载器、归一化统计信息和仿真标志 / Return dataloaders, normalization stats, and simulation flag
    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim
