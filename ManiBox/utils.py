"""工具函数模块 - Utility Functions Module

本模块包含了ManiBox机器人学习框架中使用的各种工具函数。
主要包括数据预处理、统计计算、随机采样和辅助函数。

This module contains various utility functions used in the ManiBox robot 
learning framework. Mainly includes data preprocessing, statistical computation, 
random sampling, and helper functions.

主要功能 / Main Functions:
- get_norm_stats_old: 旧版数据正则化统计 / Legacy data normalization statistics
- sample_box_pose/sample_insertion_pose: 随机位姿采样 / Random pose sampling
- compute_dict_mean: 字典平均值计算 / Dictionary mean computation
- detach_dict: 张量分离操作 / Tensor detach operations
- set_seed: 随机种子设置 / Random seed setting
"""

# 数值计算库导入 / Numerical computation library imports
import numpy as np          # 数值计算 / Numerical computing
import random               # Python随机数生成 / Python random number generation

# PyTorch相关导入 / PyTorch related imports
import torch                # PyTorch核心库 / PyTorch core library
from torch.utils.data import TensorDataset, DataLoader  # 数据加载器 / Data loaders

# 文件和数据处理 / File and data processing
import os                   # 操作系统接口 / Operating system interface
import h5py                 # HDF5文件处理 / HDF5 file processing
import cv2                  # OpenCV计算机视觉库 / OpenCV computer vision library
from PIL import Image       # Python图像处理库 / Python Image Library

# 调试工具 / Debugging tools
import IPython              # 交互式Python / Interactive Python
e = IPython.embed           # IPython调试快捷方式 / IPython debugging shortcut

def get_norm_stats_old(dataset_dir, num_episodes, use_robot_base):
    """获取数据正则化统计信息（旧版）/ Get data normalization statistics (legacy version)
    
    从数据集中计算机器人位置和动作数据的统计信息，用于数据正则化。
    这是旧版实现，主要用于向后兼容。
    
    Calculate statistical information for robot position and action data from dataset 
    for data normalization. This is a legacy implementation for backward compatibility.
    
    Args:
        dataset_dir (str): 数据集目录路径 / Dataset directory path
        num_episodes (int): 回合数量 / Number of episodes
        use_robot_base (bool): 是否使用机器人底座数据 / Whether to use robot base data
        
    Returns:
        dict: 包含均值和标准差的统计字典 / Statistics dictionary containing means and standard deviations
    """
    # 初始化数据列表 / Initialize data lists
    all_qpos_data = []    # 所有机器人位置数据 / All robot position data
    all_action_data = []  # 所有动作数据 / All action data

    # 遍历所有回合数据 / Iterate through all episode data
    for episode_idx in range(num_episodes):
        # 构建数据文件路径 / Build data file path
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        
        # 打开HDF5文件读取数据 / Open HDF5 file to read data
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]    # 机器人位置 / Robot position
            qvel = root['/observations/qvel'][()]    # 机器人速度（未使用）/ Robot velocity (unused)
            action = root['/action'][()]             # 动作数据 / Action data
            
            # 如果使用机器人底座，拼接底座动作数据 / If using robot base, concatenate base action data
            if use_robot_base:
                qpos = np.concatenate((qpos, root['/base_action'][()]), axis=1)     # 拼接位置数据 / Concatenate position data
                action = np.concatenate((action, root['/base_action'][()]), axis=1) # 拼接动作数据 / Concatenate action data
                
        # 转换为PyTorch张量并添加到列表 / Convert to PyTorch tensors and add to lists
        all_qpos_data.append(torch.from_numpy(qpos))     # 添加位置数据 / Add position data
        all_action_data.append(torch.from_numpy(action)) # 添加动作数据 / Add action data

    # 将列表转换为张量堆叠 / Convert lists to stacked tensors
    all_qpos_data = torch.stack(all_qpos_data)    # 形状: [num_episodes, timesteps, qpos_dim] / Shape: [num_episodes, timesteps, qpos_dim]
    all_action_data = torch.stack(all_action_data)  # 形状: [num_episodes, timesteps, action_dim] / Shape: [num_episodes, timesteps, action_dim]
    all_action_data = all_action_data  # 冗余赋值（保留为兼容性）/ Redundant assignment (kept for compatibility)

    # 正则化动作数据 / Normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)  # 计算动作数据均值 / Calculate action data mean
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)    # 计算动作数据标准差 / Calculate action data standard deviation
    action_std = torch.clip(action_std, 1e-2, np.inf)  # 裁剪标准差防止过小 / Clip standard deviation to prevent being too small

    # 正则化机器人位置数据 / Normalize robot position data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)  # 计算位置数据均值 / Calculate position data mean
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)    # 计算位置数据标准差 / Calculate position data standard deviation
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # 裁剪标准差防止过小 / Clip standard deviation to prevent being too small

    # 构建统计信息字典 / Build statistics information dictionary
    stats = {
        "action_mean": action_mean.numpy().squeeze(),  # 动作数据均值 / Action data mean
        "action_std": action_std.numpy().squeeze(),    # 动作数据标准差 / Action data standard deviation
        "qpos_mean": qpos_mean.numpy().squeeze(),      # 位置数据均值 / Position data mean
        "qpos_std": qpos_std.numpy().squeeze(),        # 位置数据标准差 / Position data standard deviation
        "example_qpos": qpos  # 示例位置数据（最后一个回合）/ Example position data (last episode)
    }

    return stats  # 返回统计信息 / Return statistics information


# 环境工具函数 / Environment utility functions
def sample_box_pose():
    """随机采样立方体位姿 / Random sampling of cube pose
    
    为立方体（盒子）对象生成随机的3D位姿。
    位姿包含位置（x, y, z）和四元数姿态（w, x, y, z）。
    用于在机器人操作任务中随机化物体的初始位置。
    
    Generate random 3D pose for cube (box) objects.
    Pose includes position (x, y, z) and quaternion orientation (w, x, y, z).
    Used for randomizing object initial positions in robot manipulation tasks.
    
    Returns:
        np.ndarray: 7维数组，包含[x, y, z, qw, qx, qy, qz] / 7D array containing [x, y, z, qw, qx, qy, qz]
    """
    # 定义位置采样范围 / Define position sampling ranges
    x_range = [0.0, 0.2]    # X轴范围：0到0.2米 / X-axis range: 0 to 0.2 meters
    y_range = [0.4, 0.6]    # Y轴范围：0.4到0.6米 / Y-axis range: 0.4 to 0.6 meters
    z_range = [0.05, 0.05]  # Z轴高度：固定在0.05米 / Z-axis height: fixed at 0.05 meters

    # 将范围堆叠为矩阵 / Stack ranges into matrix
    ranges = np.vstack([x_range, y_range, z_range])  # 形状: [3, 2] / Shape: [3, 2]

    # 在指定范围内随机采样位置 / Random sampling position within specified ranges
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])  # [x, y, z]

    # 设置默认姿态（无旋转的四元数）/ Set default orientation (no rotation quaternion)
    cube_quat = np.array([1, 0, 0, 0])  # [w, x, y, z] - 单位四元数 / Unit quaternion
    return np.concatenate([cube_position, cube_quat])  # 返回完整的7D位姿 / Return complete 7D pose


def sample_insertion_pose():
    """随机采样插入任务的位姿对 / Random sampling of insertion task pose pairs
    
    为插入任务生成随机的插头（peg）和插座（socket）位姿。
    插入任务是机器人学习中的经典操作任务，需要精确的空间定位。
    该函数确保插头和插座位于不同的空间区域，便于任务执行。
    
    Generate random peg and socket poses for insertion tasks.
    Insertion task is a classic manipulation task in robotics requiring precise spatial positioning.
    This function ensures peg and socket are in different spatial regions for task feasibility.
    
    Returns:
        tuple: (peg_pose, socket_pose) 两个7D位姿数组的元组 / Tuple of two 7D pose arrays
            - peg_pose: 插头位姿 [x, y, z, qw, qx, qy, qz] / Peg pose [x, y, z, qw, qx, qy, qz]
            - socket_pose: 插座位姿 [x, y, z, qw, qx, qy, qz] / Socket pose [x, y, z, qw, qx, qy, qz]
    """
    # === 插头（Peg）位姿采样 / Peg pose sampling ===
    # 定义插头的位置范围（正X区域）/ Define peg position ranges (positive X region)
    x_range = [0.1, 0.2]    # X轴：0.1到0.2米（右侧区域）/ X-axis: 0.1 to 0.2m (right region)
    y_range = [0.4, 0.6]    # Y轴：0.4到0.6米（前方区域）/ Y-axis: 0.4 to 0.6m (front region) 
    z_range = [0.05, 0.05]  # Z轴：固定高度0.05米 / Z-axis: fixed height 0.05m

    # 组合插头采样范围 / Combine peg sampling ranges
    ranges = np.vstack([x_range, y_range, z_range])
    # 随机采样插头位置 / Random sample peg position
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    # 设置插头姿态（无旋转）/ Set peg orientation (no rotation)
    peg_quat = np.array([1, 0, 0, 0])  # 单位四元数 / Unit quaternion
    peg_pose = np.concatenate([peg_position, peg_quat])  # 合成插头完整位姿 / Compose complete peg pose

    # === 插座（Socket）位姿采样 / Socket pose sampling ===
    # 定义插座的位置范围（负X区域，与插头分离）/ Define socket position ranges (negative X region, separated from peg)
    x_range = [-0.2, -0.1]  # X轴：-0.2到-0.1米（左侧区域）/ X-axis: -0.2 to -0.1m (left region)
    y_range = [0.4, 0.6]    # Y轴：与插头相同区域 / Y-axis: same region as peg
    z_range = [0.05, 0.05]  # Z轴：与插头相同高度 / Z-axis: same height as peg

    # 组合插座采样范围 / Combine socket sampling ranges  
    ranges = np.vstack([x_range, y_range, z_range])
    # 随机采样插座位置 / Random sample socket position
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    # 设置插座姿态（无旋转）/ Set socket orientation (no rotation)
    socket_quat = np.array([1, 0, 0, 0])  # 单位四元数 / Unit quaternion
    socket_pose = np.concatenate([socket_position, socket_quat])  # 合成插座完整位姿 / Compose complete socket pose

    return peg_pose, socket_pose  # 返回插头和插座位姿对 / Return peg and socket pose pair


# 辅助工具函数 / Helper utility functions
def compute_dict_mean(epoch_dicts):
    """计算字典列表的均值 / Compute mean of dictionary list
    
    对包含数值的字典列表计算逐键平均值。
    常用于训练过程中对多个epoch的指标进行平均，如损失值、准确率等。
    所有字典必须具有相同的键结构。
    
    Compute element-wise mean across a list of dictionaries containing numerical values.
    Commonly used to average metrics across multiple epochs during training, such as loss values, accuracy, etc.
    All dictionaries must have the same key structure.
    
    Args:
        epoch_dicts (list): 字典列表，每个字典包含相同的键 / List of dictionaries with identical keys
    
    Returns:
        dict: 包含平均值的字典 / Dictionary containing averaged values
    
    Example:
        >>> dicts = [{'loss': 1.0, 'acc': 0.8}, {'loss': 2.0, 'acc': 0.9}]
        >>> compute_dict_mean(dicts)
        {'loss': 1.5, 'acc': 0.85}
    """
    # 初始化结果字典，使用第一个字典的键 / Initialize result dictionary using keys from first dict
    result = {k: None for k in epoch_dicts[0]}  # 创建空结果字典 / Create empty result dict
    num_items = len(epoch_dicts)  # 字典数量 / Number of dictionaries
    
    # 对每个键计算平均值 / Compute average for each key
    for k in result:
        value_sum = 0  # 初始化累加器 / Initialize accumulator
        # 遍历所有字典，累加当前键的值 / Iterate through all dicts, accumulate current key's value
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]  # 累加值 / Accumulate value
        result[k] = value_sum / num_items  # 计算平均值 / Calculate mean
    
    return result  # 返回平均值字典 / Return averaged dictionary


def detach_dict(d):
    """分离字典中的PyTorch张量 / Detach PyTorch tensors in dictionary
    
    将字典中的所有PyTorch张量从计算图中分离，创建新的不跟踪梯度的张量。
    这在训练过程中记录统计信息时非常有用，可以避免不必要的梯度计算和内存占用。
    分离后的张量仍保持原始数值，但不会参与反向传播。
    
    Detach all PyTorch tensors in a dictionary from the computation graph,
    creating new tensors that don't track gradients. This is useful when logging
    statistics during training to avoid unnecessary gradient computation and memory usage.
    Detached tensors retain original values but won't participate in backpropagation.
    
    Args:
        d (dict): 包含PyTorch张量的字典 / Dictionary containing PyTorch tensors
        
    Returns:
        dict: 包含分离张量的新字典 / New dictionary with detached tensors
        
    Example:
        >>> import torch
        >>> d = {'loss': torch.tensor(1.0, requires_grad=True)}
        >>> detached_d = detach_dict(d)
        >>> detached_d['loss'].requires_grad  # False
    """
    new_d = dict()  # 创建新字典 / Create new dictionary
    # 遍历原字典的键值对 / Iterate through original dictionary key-value pairs
    for k, v in d.items():
        new_d[k] = v.detach()  # 分离张量并添加到新字典 / Detach tensor and add to new dictionary
    return new_d  # 返回分离后的字典 / Return dictionary with detached tensors


def set_seed(seed):
    """设置全局随机种子 / Set global random seed
    
    为所有随机数生成器设置相同的种子，确保实验的可重现性。
    这对于机器学习实验非常重要，可以保证相同的初始化和随机操作产生相同的结果。
    涵盖PyTorch、NumPy和Python标准库的随机数生成器。
    
    Set the same seed for all random number generators to ensure experiment reproducibility.
    This is crucial for machine learning experiments to guarantee that identical initialization
    and random operations produce identical results. Covers PyTorch, NumPy, and Python's
    standard library random number generators.
    
    Args:
        seed (int): 随机种子值 / Random seed value
        
    Example:
        >>> set_seed(42)  # 设置种子为42，确保可重现性 / Set seed to 42 for reproducibility
        >>> torch.randn(2, 3)  # 每次运行都会产生相同的随机数 / Will produce same random numbers each run
        
    注意 / Note:
        在分布式训练中，可能需要额外的随机种子设置以确保完全一致性。
        In distributed training, additional random seed settings may be needed for full consistency.
    """
    torch.manual_seed(seed)      # 设置PyTorch CPU随机种子 / Set PyTorch CPU random seed
    np.random.seed(seed)         # 设置NumPy随机种子 / Set NumPy random seed  
    random.seed(seed)            # 设置Python标准库随机种子 / Set Python standard library random seed
