"""
EpisodicDataset.py

基础情节数据集类，用于加载和处理HDF5格式的机器人学习数据。
该数据集处理单个时间步的观测和动作数据，是其他历史数据集的基础类。

Basic episodic dataset class for loading and processing HDF5 format robotics learning data.
This dataset handles single timestep observations and actions, serving as the base class for other historical datasets.
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

# 项目特定导入 / Project-specific imports
import ManiBox  # ManiBox项目主模块 / ManiBox main project module
from ManiBox.utils import get_norm_stats_old  # 数据归一化统计工具 / Data normalization statistics utility

class EpisodicDataset(torch.utils.data.Dataset):
    """
    基础情节数据集类
    
    用于加载和处理HDF5格式的机器人学习数据集。该数据集处理单个时间步的数据样本，
    包括多视角图像观测、关节位置、动作序列等。支持数据归一化和各种配置选项。
    
    Basic Episodic Dataset Class
    
    Loads and processes HDF5 format robotics learning datasets. This dataset handles single timestep
    data samples including multi-view image observations, joint positions, action sequences, etc.
    Supports data normalization and various configuration options.
    
    Args:
        episode_ids: 情节ID列表 / List of episode IDs
        dataset_dir: 数据集目录路径 / Dataset directory path
        camera_names: 相机名称列表 / List of camera names
        norm_stats: 归一化统计信息 / Normalization statistics
        arm_delay_time: 机械臂延迟时间 / Arm delay time
        max_pos_lookahead: 最大位置前瞻步数 / Maximum position lookahead steps
        use_dataset_action: 是否使用数据集动作 / Whether to use dataset actions
        use_depth_image: 是否使用深度图像 / Whether to use depth images
        use_robot_base: 是否使用机器人底座 / Whether to use robot base
        episode_begin: 情节开始索引 / Episode begin index
        episode_end: 情节结束索引 / Episode end index
    """

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, arm_delay_time, max_pos_lookahead,
                 use_dataset_action, use_depth_image, use_robot_base, episode_begin, episode_end):
        super(EpisodicDataset).__init__()  # 调用父类构造函数 / Call parent class constructor
        
        # 基础数据集配置 / Basic dataset configuration
        self.episode_ids = episode_ids  # 情节ID列表，通常包含1000个情节 / List of episode IDs, typically contains 1000 episodes
        self.dataset_dir = dataset_dir  # 数据集根目录路径 / Dataset root directory path
        self.camera_names = camera_names  # 相机名称列表，如['cam_high', 'cam_left_wrist', 'cam_right_wrist'] / Camera names list
        self.norm_stats = norm_stats  # 归一化统计信息（均值和标准差） / Normalization statistics (mean and std)
        self.is_sim = None  # 是否为仿真数据的标志（稍后初始化） / Flag indicating if data is from simulation (initialized later)
        
        # 数据处理配置 / Data processing configuration
        self.max_pos_lookahead = max_pos_lookahead  # 最大前瞻位置步数 / Maximum lookahead position steps
        self.use_dataset_action = use_dataset_action  # 是否使用数据集中的动作数据 / Whether to use action data from dataset
        self.use_depth_image = use_depth_image  # 是否使用深度图像数据 / Whether to use depth image data
        self.arm_delay_time = arm_delay_time  # 机械臂响应延迟时间 / Arm response delay time
        self.use_robot_base = use_robot_base  # 是否包含机器人底座信息 / Whether to include robot base information
        
        # 情节切片配置 / Episode slicing configuration
        self.episode_begin = episode_begin  # 情节开始时间步（避免边界情况） / Episode begin timestep (avoid edge cases)
        self.episode_end = episode_end  # 情节结束时间步 / Episode end timestep
        
        # 通过获取第一个样本来初始化仿真标志 / Initialize simulation flag by getting first sample
        self.__getitem__(0)  # 初始化self.is_sim / Initialize self.is_sim

    def __len__(self):
        """
        返回数据集中的情节数量
        Return the number of episodes in the dataset
        """
        return len(self.episode_ids)  # 返回情节ID列表的长度 / Return length of episode IDs list

    def shuffle(self):
        """
        随机打乱情节顺序，用于训练时的数据随机化
        Randomly shuffle episode order for data randomization during training
        """
        np.random.shuffle(self.episode_ids)  # 使用numpy随机打乱情节ID列表 / Use numpy to randomly shuffle episode IDs list
        
    def __getitem__(self, index):
        """
        获取指定索引的数据样本
        Get data sample at specified index
        
        Args:
            index: 样本索引 / Sample index
        
        Returns:
            tuple: (image_data, image_depth_data, qpos_data, next_action_data, 
                   next_action_is_pad, action_data, action_is_pad)
        """
        # 调试信息（已注释） / Debug info (commented out)
        # print('episode_ids', self.episode_ids)  # 打印情节ID列表 / Print episode IDs list
        
        episode_id = self.episode_ids[index]  # 获取当前索引对应的情节ID / Get episode ID for current index
        
        # 构建HDF5文件路径 / Build HDF5 file path
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        # print(f"episode_id {episode_id}")  # 调试：打印情节ID / Debug: print episode ID
        # print('load path:', dataset_path)  # 调试：打印加载路径 / Debug: print load path

        # 打开HDF5文件进行读取 / Open HDF5 file for reading
        with h5py.File(dataset_path, 'r') as root:
            # 调试断点（已注释） / Debug breakpoint (commented out)
            # import pdb  # Python调试器 / Python debugger
            # pdb.set_trace()  # 设置断点 / Set breakpoint
            
            # HDF5文件结构说明 / HDF5 file structure explanation:
            # root中的键: ['action', 'base_action', 'observations'] / Keys in root: ['action', 'base_action', 'observations']
            # root['action']: 形状为500 * 14的动作数据 / root['action']: Action data with shape 500 * 14
            # root['observations']: 包含'effort', 'images', 'images_depth', 'qpos', 'qvel' / Contains 'effort', 'images', 'images_depth', 'qpos', 'qvel'
            # root.attrs中的键: sim（是否仿真）, compress（是否压缩） / Keys in root.attrs: sim (simulation flag), compress (compression flag)
            
            # 情节切片方案（已注释，现在使用不同的切片策略） / Episode slicing scheme (commented out, now using different slicing strategy)
            # 以下代码用于直接修改HDF5数据集的切片，但现在使用索引切片 / Following code for directly modifying HDF5 dataset slices, but now using index slicing
            # if self.episode_begin is not None:  # 如果设置了情节开始索引 / If episode begin index is set
            #     root['action'] = root['action'][self.episode_begin:self.episode_end]  # 切片动作数据 / Slice action data
            #     root['base_action'] = root['base_action'][self.episode_begin:self.episode_end]  # 切片底座动作数据 / Slice base action data
            #     root['observations/effort'] = root['observations/effort'][self.episode_begin:self.episode_end]  # 切片力矩数据 / Slice effort data
            #     root['observations/qpos'] = root['observations/qpos'][self.episode_begin:self.episode_end]  # 切片关节位置数据 / Slice joint position data
            #     root['observations/qvel'] = root['observations/qvel'][self.episode_begin:self.episode_end]  # 切片关节速度数据 / Slice joint velocity data
            #     # 切片各相机图像数据 / Slice image data from each camera
            #     root['observations/images/cam_high'] = root['observations/images/cam_high'][self.episode_begin:self.episode_end]
            #     root['observations/images/cam_left_wrist'] = root['observations/images/cam_left_wrist'][self.episode_begin:self.episode_end]
            #     root['observations/images/cam_right_wrist'] = root['observations/images/cam_right_wrist'][self.episode_begin:self.episode_end]
            #     # 切片各相机深度图像数据 / Slice depth image data from each camera
            #     root['observations/images_depth/cam_high'] = root['observations/images_depth/cam_high'][self.episode_begin:self.episode_end]
            #     root['observations/images_depth/cam_left_wrist'] = root['observations/images_depth/cam_left_wrist'][self.episode_begin:self.episode_end]
            #     root['observations/images_depth/cam_right_wrist'] = root['observations/images_depth/cam_right_wrist'][self.episode_begin:self.episode_end]
            
            
            # 读取数据集元属性 / Read dataset metadata attributes
            is_sim = root.attrs['sim']  # 是否为仿真数据标志 / Simulation data flag
            is_compress = root.attrs['compress']  # 图像数据是否压缩标志 / Image data compression flag
            
            # 计算动作序列的原始形状 / Calculate original shape of action sequence
            original_action_shape = root['/action'][self.episode_begin:self.episode_end].shape  # 获取切片后的动作数据形状 / Get sliced action data shape
            max_action_len = original_action_shape[0]  # 最大动作序列长度（时间步数） / Maximum action sequence length (number of timesteps)
            
            # 如果使用机器人底座，调整动作维度 / If using robot base, adjust action dimensions
            if self.use_robot_base:
                # 添加2个底座控制维度（通常是线速度和角速度） / Add 2 base control dimensions (typically linear and angular velocity)
                original_action_shape = (original_action_shape[0], original_action_shape[1] + 2)

            # 随机选择起始时间步进行数据采样 / Randomly select starting timestep for data sampling
            # max_action_len可能等于self.episode_end / max_action_len may equal self.episode_end
            start_ts = np.random.choice(max_action_len)  # 随机选择起始时间步 / Randomly select starting timestep
            next_action_size = random.randint(0, self.max_pos_lookahead)  # 随机确定前瞻动作数量 / Randomly determine number of lookahead actions
            
            # 根据配置选择动作数据源 / Select action data source based on configuration
            if self.use_dataset_action:
                # 使用数据集中的动作标签 / Use action labels from dataset
                actions = root['/action'][self.episode_begin:self.episode_end]
            else:
                # 使用关节位置的差分作为动作（下一时刻位置） / Use joint position differences as actions (next timestep positions)
                actions = root['/observations/qpos'][self.episode_begin:self.episode_end][1:]  # 从第二个时间步开始 / Start from second timestep
                actions = np.append(actions, actions[-1][np.newaxis, :], axis=0)  # 复制最后一帧以保持长度 / Duplicate last frame to maintain length
            
            # 计算前瞻动作的结束索引 / Calculate end index for lookahead actions
            end_next_action_index = min(start_ts + next_action_size, max_action_len - 1)
            
            # 获取当前时间步的关节位置 / Get joint positions at current timestep
            qpos = root['/observations/qpos'][self.episode_begin:self.episode_end][start_ts]
            
            # 如果使用机器人底座，将底座状态添加到关节位置 / If using robot base, add base state to joint positions
            if self.use_robot_base:
                base_action = root['/base_action'][self.episode_begin:self.episode_end][start_ts]  # 获取底座动作 / Get base action
                qpos = np.concatenate((qpos, base_action), axis=0)  # 拼接关节位置和底座状态 / Concatenate joint positions and base state
            # 初始化图像数据字典 / Initialize image data dictionaries
            image_dict = dict()  # 存储RGB图像数据 / Store RGB image data
            image_depth_dict = dict()  # 存储深度图像数据 / Store depth image data
            
            # 遍历所有相机，加载图像数据 / Iterate through all cameras to load image data
            for cam_name in self.camera_names:
                if is_compress:  # 如果图像数据是压缩格式 / If image data is compressed
                    # 读取压缩的图像数据 / Read compressed image data
                    decoded_image = root[f'/observations/images/{cam_name}'][self.episode_begin:self.episode_end][start_ts]
                    
                    # 解压缩图像并转换颜色空间 / Decompress image and convert color space
                    # cv2.imdecode(decoded_image, 1)  # 原始解码方案（已注释） / Original decoding scheme (commented)
                    compressed_data = np.frombuffer(decoded_image, np.uint8)  # 从缓冲区读取字节数据 / Read byte data from buffer
                    bgr_image = cv2.imdecode(compressed_data, 1)  # 解码为BGR格式 / Decode to BGR format
                    image_dict[cam_name] = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式 / Convert to RGB format
                    
                    # 调试代码：保存图像文件（已注释） / Debug code: save image files (commented)
                    # Image.fromarray(image_dict[cam_name], 'RGB').save(f'{cam_name}_first_image.jpg', 'JPEG')  # 保存为JPEG / Save as JPEG
                    # cv2.imwrite(f'~/Videos/ACT_reconstruction_image.jpg', image_dict[cam_name])  # 使用OpenCV保存 / Save using OpenCV
                    # print("save image!")  # 打印保存信息 / Print save info
                    # exit(0)  # 退出程序 / Exit program
                    # print(image_dict[cam_name].shape)  # 打印图像形状 / Print image shape
                    # exit(-1)  # 退出程序 / Exit program
                else:
                    # 直接读取未压缩的图像数据 / Directly read uncompressed image data
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][self.episode_begin:self.episode_end][start_ts]

                # 如果使用深度图像，加载深度数据 / If using depth images, load depth data
                if self.use_depth_image:
                    image_depth_dict[cam_name] = root[f'/observations/images_depth/{cam_name}'][self.episode_begin:self.episode_end][start_ts]

            # 动作序列切片和时间对齐 / Action sequence slicing and temporal alignment
            start_action = end_next_action_index  # 动作序列的起始索引 / Starting index for action sequence
            
            # 原始切片方案（已注释） / Original slicing scheme (commented)
            # action = actions[start_action:]  # 从起始动作索引开始的动作序列 / Action sequence starting from start action index
            # next_action = actions[start_ts:start_action]  # 前瞻动作序列 / Lookahead action sequence
            # action_len = max_action_len - start_action  # 动作序列长度 / Action sequence length
            # next_action_len = start_action - start_ts  # 前瞻动作序列长度 / Lookahead action sequence length
            
            # 考虑机械臂延迟的时间对齐策略 / Temporal alignment strategy considering arm delay
            index = max(0, start_action - self.arm_delay_time)  # 考虑延迟后的实际起始索引 / Actual starting index considering delay
            
            # 调试信息（已注释） / Debug info (commented)
            # print("qpos:", qpos[7:-2])  # 打印关节位置的部分维度 / Print partial dimensions of joint positions
            
            # 时间对齐的动作切片（hack方式） / Temporally aligned action slicing (hack approach)
            action = actions[index:]  # 从调整后的索引开始的动作序列，用于时间步对齐 / Action sequence from adjusted index for timestep alignment
            
            # 调试信息（已注释） / Debug info (commented)  
            # print("action:", action[:30, 7:-2])  # 打印前30个动作的部分维度 / Print partial dimensions of first 30 actions
            
            next_action = actions[start_ts:index]  # 前瞻动作序列，如果start_ts == index则可能为空 / Lookahead action sequence, could be empty if start_ts == index
            
            # 如果使用机器人底座，为动作序列添加底座控制 / If using robot base, add base control to action sequences
            if self.use_robot_base:
                base_actions = root['/base_action'][self.episode_begin:self.episode_end][index:]  # 底座动作序列 / Base action sequence
                action = np.concatenate((action, base_actions), axis=1)  # 拼接关节动作和底座动作 / Concatenate joint actions and base actions
                
                base_next_actions = root['/base_action'][self.episode_begin:self.episode_end][start_ts:index]  # 底座前瞻动作 / Base lookahead actions
                next_action = np.concatenate((next_action, base_next_actions), axis=1)  # 拼接前瞻动作 / Concatenate lookahead actions
            
            # 计算实际的动作序列长度 / Calculate actual action sequence lengths
            action_len = max_action_len - index  # 时间对齐后的动作序列长度 / Action sequence length after temporal alignment
            next_action_len = index - start_ts  # 前瞻动作序列长度 / Lookahead action sequence length
            # 基于数据类型的不同切片策略（已注释，现在使用统一策略） / Different slicing strategies based on data type (commented, now using unified strategy)
            # if is_sim:  # 如果是仿真数据 / If simulation data
            #     action = actions[start_action:]  # 直接从起始动作开始 / Start directly from start action
            #     next_action = actions[start_ts:start_action]  # 标准前瞻动作 / Standard lookahead actions
            #     action_len = max_action_len - start_action  # 标准动作长度计算 / Standard action length calculation
            #     next_action_len = start_action - start_ts  # 标准前瞻长度计算 / Standard lookahead length calculation
            # else:  # 如果是真实数据 / If real data
            #     index = max(0, start_action - 1)  # 真实数据需要1步延迟补偿 / Real data needs 1-step delay compensation
            #     action = actions[index:]  # 使用延迟补偿的动作切片 / Use delay-compensated action slicing
            #     next_action = actions[start_ts:index]  # 对应的前瞻动作 / Corresponding lookahead actions
            #     action_len = max_action_len - index  # 延迟补偿后的动作长度 / Action length after delay compensation
            #     next_action_len = index - start_ts  # 延迟补偿后的前瞻长度 / Lookahead length after delay compensation

        self.is_sim = is_sim  # 保存仿真数据标志供其他方法使用 / Save simulation data flag for use by other methods

        # 动作序列填充和掩码生成 / Action sequence padding and mask generation
        
        # 创建填充后的主动作序列 / Create padded main action sequence
        padded_action = np.zeros(original_action_shape, dtype=np.float32)  # 初始化零填充的动作数组 / Initialize zero-padded action array
        padded_action[:action_len] = action  # 填入实际动作数据 / Fill in actual action data
        
        # 创建主动作序列的填充掩码 / Create padding mask for main action sequence
        action_is_pad = np.zeros(max_action_len)  # 初始化填充掩码（0表示有效数据） / Initialize padding mask (0 for valid data)
        action_is_pad[action_len:] = 1  # 标记填充部分（1表示填充数据） / Mark padded portion (1 for padded data)
        
        # 创建填充后的前瞻动作序列 / Create padded lookahead action sequence
        padded_next_action = np.zeros((self.max_pos_lookahead, original_action_shape[1]), dtype=np.float32)  # 初始化前瞻动作数组 / Initialize lookahead action array

        # 创建前瞻动作序列的填充掩码 / Create padding mask for lookahead action sequence
        next_action_is_pad = np.zeros(self.max_pos_lookahead)  # 初始化前瞻填充掩码 / Initialize lookahead padding mask
        
        if next_action_len <= 0:  # 如果没有前瞻动作 / If no lookahead actions
            next_action_is_pad[:] = 1  # 全部标记为填充 / Mark all as padded
        else:  # 如果有前瞻动作 / If there are lookahead actions
            padded_next_action[:next_action_len] = next_action  # 填入前瞻动作数据 / Fill in lookahead action data
            next_action_is_pad[next_action_len:] = 1  # 标记剩余部分为填充 / Mark remaining portion as padded

        # 整合所有相机的图像数据 / Consolidate image data from all cameras
        all_cam_images = []  # 存储所有相机图像的列表 / List to store all camera images
        for cam_name in self.camera_names:  # 遍历所有相机名称 / Iterate through all camera names
            all_cam_images.append(image_dict[cam_name])  # 添加当前相机的图像 / Add current camera's image
        
        all_cam_images = np.stack(all_cam_images, axis=0)  # 沿第0轴堆叠图像数组 / Stack image arrays along axis 0
        
        # 构建观测数据的张量表示 / Construct tensor representation of observation data
        image_data = torch.from_numpy(all_cam_images)  # 将numpy数组转换为PyTorch张量 / Convert numpy array to PyTorch tensor
        image_data = torch.einsum('k h w c -> k c h w', image_data)  # 重排维度：(相机数, 高, 宽, 通道) -> (相机数, 通道, 高, 宽) / Rearrange dimensions: (cameras, height, width, channels) -> (cameras, channels, height, width)
        
        # 最终图像数据形状说明 / Final image data shape explanation
        # image_data: (cam_num, RGB3, 480, 640) - (相机数量, RGB通道数, 图像高度, 图像宽度) / (number of cameras, RGB channels, image height, image width)
        
        image_data = image_data / 255.0  # 将像素值从[0,255]范围归一化到[0,1] / Normalize pixel values from [0,255] range to [0,1]

        # 处理深度图像数据 / Process depth image data
        image_depth_data = np.zeros(1, dtype=np.float32)  # 默认初始化为占位符数组 / Default initialize as placeholder array
        
        if self.use_depth_image:  # 如果使用深度图像 / If using depth images
            # 整合所有相机的深度图像数据 / Consolidate depth image data from all cameras
            all_cam_images_depth = []  # 存储所有相机深度图像的列表 / List to store all camera depth images
            for cam_name in self.camera_names:  # 遍历所有相机名称 / Iterate through all camera names
                all_cam_images_depth.append(image_depth_dict[cam_name])  # 添加当前相机的深度图像 / Add current camera's depth image
            
            all_cam_images_depth = np.stack(all_cam_images_depth, axis=0)  # 沿第0轴堆叠深度图像数组 / Stack depth image arrays along axis 0
            
            # 构建深度观测数据的张量表示 / Construct tensor representation of depth observation data
            image_depth_data = torch.from_numpy(all_cam_images_depth)  # 将numpy数组转换为PyTorch张量 / Convert numpy array to PyTorch tensor
            # 深度图像通常不需要通道维度重排，因为只有1个通道 / Depth images usually don't need channel dimension rearrangement as they have only 1 channel
            # image_depth_data = torch.einsum('k h w c -> k c h w', image_depth_data)  # 如果需要可取消注释 / Uncomment if needed
            image_depth_data = image_depth_data / 255.0  # 将深度值归一化（假设深度存储为0-255范围） / Normalize depth values (assuming depth stored in 0-255 range)

        # 数据类型转换和归一化处理 / Data type conversion and normalization processing
        
        # 处理关节位置数据 / Process joint position data
        qpos_data = torch.from_numpy(qpos).float()  # 转换为浮点型张量 / Convert to float tensor
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]  # Z-score归一化 / Z-score normalization
        
        # 处理前瞻动作数据 / Process lookahead action data
        next_action_data = torch.from_numpy(padded_next_action).float()  # 转换为浮点型张量 / Convert to float tensor
        next_action_is_pad = torch.from_numpy(next_action_is_pad).bool()  # 转换为布尔型张量（填充掩码） / Convert to boolean tensor (padding mask)
        
        # 处理主动作数据 / Process main action data
        action_data = torch.from_numpy(padded_action).float()  # 转换为浮点型张量 / Convert to float tensor
        action_is_pad = torch.from_numpy(action_is_pad).bool()  # 转换为布尔型张量（填充掩码） / Convert to boolean tensor (padding mask)
        
        # 根据动作数据源类型进行不同的归一化 / Different normalization based on action data source type
        if self.use_dataset_action:  # 如果使用数据集中的动作标签 / If using action labels from dataset
            # 使用动作数据的统计信息进行归一化 / Normalize using action data statistics
            next_action_data = (next_action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
            action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        else:  # 如果使用关节位置差分作为动作 / If using joint position differences as actions
            # 使用关节位置数据的统计信息进行归一化 / Normalize using joint position data statistics
            next_action_data = (next_action_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
            action_data = (action_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # 调试打印选项设置（已注释） / Debug printing options (commented)
        # torch.set_printoptions(precision=10, sci_mode=False)  # 设置张量打印精度和格式 / Set tensor printing precision and format
        # torch.set_printoptions(threshold=float('inf'))  # 设置张量打印阈值 / Set tensor printing threshold
        # print("qpos_data:", qpos_data[7:])  # 打印关节位置数据的后几维 / Print later dimensions of joint position data
        # print("action_data:", action_data[:, 7:])  # 打印动作数据的后几维 / Print later dimensions of action data
        
        # 调试断点（已注释） / Debug breakpoint (commented)
        # import pdb; pdb.set_trace()  # Python调试器断点 / Python debugger breakpoint
        
        # 返回数据的形状说明 / Shape explanation for returned data
        # image_data.shape: (3, 3, 480, 640) - (相机数, RGB通道数, 图像高度, 图像宽度) / (cameras, RGB channels, image height, image width)
        # qpos_data.shape: (14) - 14维关节位置向量 / 14-dimensional joint position vector
        # action_data.shape: (~87, 14) - (动作序列长度, 14维动作向量) / (action sequence length, 14-dimensional action vector)
        
        # 返回完整的数据元组 / Return complete data tuple
        return image_data, image_depth_data, qpos_data, next_action_data, next_action_is_pad, action_data, action_is_pad

