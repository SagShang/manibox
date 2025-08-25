"""
HistoryEpisodicDataset.py

历史情节数据集类，用于加载和处理包含时间序列历史信息的机器人学习数据。
该数据集将多个连续时间步的观测数据组合成历史序列，支持基于历史状态的动作预测任务。

History Episodic Dataset class for loading and processing robotics learning data
with temporal historical information. This dataset combines observations from multiple
consecutive timesteps into historical sequences, supporting history-based action prediction tasks.
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

# 可视化库 / Visualization libraries
from matplotlib import pyplot as plt  # Matplotlib绘图库 / Matplotlib plotting library

# 调试工具 / Debugging tools
import IPython  # 交互式Python环境 / Interactive Python environment

# 项目特定导入 / Project-specific imports
import ManiBox  # ManiBox项目主模块 / ManiBox main project module
from ManiBox.utils import get_norm_stats_old  # 数据归一化统计工具 / Data normalization statistics utility
from ManiBox.dataloader.EpisodicDataset import EpisodicDataset  # 基础情节数据集类 / Base episodic dataset class

# 图像显示工具函数（调试用）/ Image display utility function (for debugging)
# from ManiBox.utils.try_yolo import output_RGB_image  # 原始导入（已注释）/ Original import (commented)

def output_RGB_image(img):
    """
    显示RGB图像的辅助函数（用于调试）
    Helper function to display RGB images (for debugging)
    
    Args:
        img: 要显示的RGB图像数组 / RGB image array to display
    """
    plt.imshow(img)  # 使用matplotlib显示图像 / Display image using matplotlib
    plt.show()  # 显示图像窗口 / Show image window

class HistoryEpisodicDataset(torch.utils.data.Dataset):
    """
    历史情节数据集类
    
    用于加载和处理包含时间序列历史信息的机器人学习数据集。
    与基础EpisodicDataset不同，该数据集将多个连续时间步的观测数据组合成历史序列，
    支持基于历史状态序列的动作预测，特别适用于需要考虑时序依赖的机器人控制任务。
    
    History Episodic Dataset Class
    
    Loads and processes robotics learning datasets with temporal historical information.
    Unlike the base EpisodicDataset, this dataset combines observations from multiple consecutive
    timesteps into historical sequences, supporting action prediction based on historical state sequences,
    particularly suitable for robot control tasks requiring temporal dependencies.
    
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
        self.norm_stats = norm_stats  # 归一化统计信息（均值和标准差）/ Normalization statistics (mean and std)
        self.is_sim = None  # 是否为仿真数据的标志（稍后初始化）/ Flag indicating if data is from simulation (initialized later)
        
        # 数据处理配置 / Data processing configuration
        self.max_pos_lookahead = max_pos_lookahead  # 最大前瞻位置步数 / Maximum lookahead position steps
        self.use_dataset_action = use_dataset_action  # 是否使用数据集中的动作数据 / Whether to use action data from dataset
        self.use_depth_image = use_depth_image  # 是否使用深度图像数据 / Whether to use depth image data
        self.arm_delay_time = arm_delay_time  # 机械臂响应延迟时间 / Arm response delay time
        self.use_robot_base = use_robot_base  # 是否包含机器人底座信息 / Whether to include robot base information
        
        # 情节切片配置 / Episode slicing configuration
        self.episode_begin = episode_begin  # 情节开始时间步（避免边界情况）/ Episode begin timestep (avoid edge cases)
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
        获取指定索引的历史序列数据样本
        Get historical sequence data sample at specified index
        
        该方法与基础EpisodicDataset的主要区别在于返回完整的历史序列而非单个时间步。
        The main difference from base EpisodicDataset is returning complete historical sequences instead of single timesteps.
        
        Args:
            index: 样本索引 / Sample index
        
        Returns:
            tuple: (image_data, image_depth_data, qpos_data, next_action_data, 
                   next_action_is_pad, action_data, action_is_pad)
        """
        # 调试信息（已注释）/ Debug info (commented out)
        # print('episode_ids', self.episode_ids)  # 打印情节ID列表 / Print episode IDs list
        
        episode_id = self.episode_ids[index]  # 获取当前索引对应的情节ID / Get episode ID for current index
        
        # 构建HDF5文件路径 / Build HDF5 file path
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        # print(f"episode_id {episode_id}")  # 调试：打印情节ID / Debug: print episode ID
        # print('load path:', dataset_path)  # 调试：打印加载路径 / Debug: print load path

        # 打开HDF5文件进行读取 / Open HDF5 file for reading
        with h5py.File(dataset_path, 'r') as root:
            # 读取数据集元属性 / Read dataset metadata attributes
            is_sim = root.attrs['sim']  # 是否为仿真数据标志 / Simulation data flag
            is_compress = root.attrs['compress']  # 图像数据是否压缩标志 / Image data compression flag
            
            # 计算动作序列的原始形状 / Calculate original shape of action sequence
            original_action_shape = root['/action'][self.episode_begin:self.episode_end].shape  # 获取切片后的动作数据形状 / Get sliced action data shape
            max_action_len = original_action_shape[0]  # 最大动作序列长度（时间步数）/ Maximum action sequence length (number of timesteps)
            
            # 如果使用机器人底座，调整动作维度 / If using robot base, adjust action dimensions
            if self.use_robot_base:
                # 添加2个底座控制维度（通常是线速度和角速度）/ Add 2 base control dimensions (typically linear and angular velocity)
                original_action_shape = (original_action_shape[0], original_action_shape[1] + 2)

            # 历史序列的时间窗口设置 / Historical sequence temporal window setup
            # max_action_len可能等于self.episode_end / max_action_len may equal self.episode_end
            start_ts = max_action_len  # 当前时间步设为序列末尾（使用完整历史）/ Current timestep set to end of sequence (use full history)
            history_start = 0  # 历史序列的起始时间步 / Starting timestep for historical sequence
            
            # 数据流说明 / Data flow explanation:
            # 输入：时间步[history_start, start_ts]的状态 / Input: states of timesteps [history_start, start_ts]
            # 输出：时间步[start_ts, start_ts + ?]的动作 / Output: actions of timesteps [start_ts, start_ts + ?]
            
            next_action_size = random.randint(0, self.max_pos_lookahead)  # 随机确定前瞻动作数量 / Randomly determine number of lookahead actions
            
            # 根据配置选择动作数据源 / Select action data source based on configuration
            if self.use_dataset_action:
                # 使用数据集中的动作标签 / Use action labels from dataset
                actions = root['/action'][self.episode_begin:self.episode_end]
            else:
                # 使用关节位置的差分作为动作（下一时刻位置）/ Use joint position differences as actions (next timestep positions)
                actions = root['/observations/qpos'][self.episode_begin:self.episode_end][1:]  # 从第二个时间步开始 / Start from second timestep
                actions = np.append(actions, actions[-1][np.newaxis, :], axis=0)  # 复制最后一帧以保持长度 / Duplicate last frame to maintain length
            
            # 计算前瞻动作的结束索引 / Calculate end index for lookahead actions
            end_next_action_index = min(start_ts + next_action_size, max_action_len - 1)
            
            # 状态数据：关节位置和图像 / State data: joint positions and images
            # TODO: 下面的额外分支尚未修改 / TODO: The extra branch below has not been modified
            
            # 获取历史关节位置序列 / Get historical joint position sequence
            qpos = root['/observations/qpos'][self.episode_begin:self.episode_end][history_start: start_ts + 1]  # 包含完整历史序列 / Include complete historical sequence
            
            # 如果使用机器人底座，添加底座状态信息 / If using robot base, add base state information
            if self.use_robot_base:
                base_action = root['/base_action'][self.episode_begin:self.episode_end][start_ts]  # 获取底座动作 / Get base action
                qpos = np.concatenate((qpos, base_action), axis=0)  # 拼接关节位置和底座状态 / Concatenate joint positions and base state
            # 初始化图像数据容器 / Initialize image data containers
            # image_dict = dict()  # 原始字典方案（已注释）/ Original dictionary approach (commented)
            all_cam_images = []  # 存储所有相机的历史图像序列 / Store historical image sequences from all cameras
            # 最终形状：(cam_num, context_len, 480, 640, RGB3) / Final shape: (cam_num, context_len, 480, 640, RGB3)
            
            # image_depth_dict = dict()  # 原始深度图像字典（已注释）/ Original depth image dictionary (commented)
            all_cam_images_depth = []  # 存储所有相机的深度图像序列 / Store depth image sequences from all cameras
            
            # 遍历所有相机，加载历史图像序列 / Iterate through all cameras to load historical image sequences
            for cam_name in self.camera_names:
                all_cam_images.append([])  # 为当前相机创建图像列表 / Create image list for current camera
                
                if is_compress:  # 如果图像数据是压缩格式 / If image data is compressed
                    # 读取压缩的历史图像序列 / Read compressed historical image sequence
                    decoded_image = root[f'/observations/images/{cam_name}'][self.episode_begin:self.episode_end][history_start: start_ts + 1]
                    
                    # 逐帧解压缩图像 / Decompress images frame by frame
                    for i in range(decoded_image.shape[0]):  # i: 每个时间步 / i: each timestep
                        # 解码压缩图像数据 / Decode compressed image data
                        compressed_data = np.frombuffer(decoded_image[i], np.uint8)  # 从缓冲区读取字节数据 / Read byte data from buffer
                        image = cv2.imdecode(compressed_data, 1)  # 解码为BGR格式 / Decode to BGR format
                        all_cam_images[-1].append(image)  # 添加到当前相机的图像列表 / Add to current camera's image list
                        # 注意：这里的图像是RGB格式，形状为(480, 640, 3)，值范围0~255 / Note: image here is RGB format, shape (480, 640, 3), values 0~255
                    
                    # 原始单帧解码方案（已注释）/ Original single-frame decoding scheme (commented)
                    # image_dict[cam_name] = cv2.cvtColor(cv2.imdecode(np.frombuffer(decoded_image, np.uint8), 1), cv2.COLOR_BGR2RGB)
                    
                    # 测试代码：图像解码验证（已注释）/ Test code: image decoding verification (commented)
                    # image = cv2.imdecode(np.frombuffer(decoded_image[10], np.uint8), 1)  # 解码第10帧用于测试 / Decode 10th frame for testing
                    # print("image.shape", image.shape)  # 打印图像形状 / Print image shape
                    # print("image value", image[0, 0, :])  # 打印左上角像素值 / Print top-left pixel values
                    # output_RGB_image(image)  # 显示图像 / Display image
                    # exit(0)  # 退出程序 / Exit program
                    # Image.fromarray(image, 'RGB').save(f'{cam_name}_first_image.jpg', 'JPEG')  # 保存为JPEG / Save as JPEG
                    # cv2.imwrite(f'~/Videos/ACT_reconstruction_image.jpg', image)  # 使用OpenCV保存 / Save using OpenCV
                    # print("save image!")  # 打印保存信息 / Print save info
                    # exit(0)  # 退出程序 / Exit program
                else:
                    # 直接读取未压缩的图像数据（注意：这里只读取了单帧而非历史序列）/ Directly read uncompressed image data (Note: only single frame, not historical sequence)
                    all_cam_images[-1].append(root[f'/observations/images/{cam_name}'][self.episode_begin:self.episode_end][start_ts])

                # 处理深度图像数据 / Process depth image data
                if self.use_depth_image:
                    all_cam_images_depth.append([])  # 为当前相机创建深度图像列表 / Create depth image list for current camera
                    # 读取深度图像数据（注意：这里也只读取了单帧）/ Read depth image data (Note: also only single frame)
                    depth_image = root[f'/observations/images_depth/{cam_name}'][self.episode_begin:self.episode_end][start_ts]
                    all_cam_images_depth[-1].append(depth_image)

            # 提取动作序列数据 / Extract action sequence data
            action = actions[history_start: start_ts + 1]  # 历史动作序列，包含完整的时间窗口 / Historical action sequence, includes complete time window
            next_action = actions[start_ts: start_ts]  # 前瞻动作序列（空列表，因为start_ts == start_ts）/ Lookahead action sequence (empty list since start_ts == start_ts)
            
            # 如果使用机器人底座，为动作序列添加底座控制 / If using robot base, add base control to action sequences
            if self.use_robot_base:
                # 注意：这里的index变量未定义，可能是代码错误 / Note: index variable is undefined here, might be a code error
                base_actions = root['/base_action'][self.episode_begin:self.episode_end][index:]  # 底座动作序列 / Base action sequence
                action = np.concatenate((action, base_actions), axis=1)  # 拼接关节动作和底座动作 / Concatenate joint actions and base actions
                
                base_next_actions = root['/base_action'][self.episode_begin:self.episode_end][start_ts:index]  # 底座前瞻动作 / Base lookahead actions
                next_action = np.concatenate((next_action, base_next_actions), axis=1)  # 拼接前瞻动作 / Concatenate lookahead actions

        self.is_sim = is_sim  # 保存仿真数据标志供其他方法使用 / Save simulation data flag for use by other methods
        
        # 创建前瞻动作的填充数组 / Create padded array for lookahead actions
        padded_next_action = np.zeros((self.max_pos_lookahead, original_action_shape[1]), dtype=np.float32)  # 零填充的前瞻动作数组 / Zero-padded lookahead action array

        # 创建前瞻动作的填充掩码 / Create padding mask for lookahead actions
        next_action_is_pad = np.zeros(self.max_pos_lookahead)  # 初始化前瞻填充掩码（0表示有效数据）/ Initialize lookahead padding mask (0 for valid data)

        # 原始图像整合方案（已注释）/ Original image consolidation scheme (commented)
        # all_cam_images = []  # 原始方案：从字典中收集图像 / Original scheme: collect images from dictionary
        # for cam_name in self.camera_names:  # 遍历相机名称 / Iterate camera names
        #     all_cam_images.append(image_dict[cam_name])  # 添加图像数据 / Add image data
        
        # 将嵌套列表转换为numpy数组 / Convert nested list to numpy array
        all_cam_images = np.stack(all_cam_images, axis=0)  # 沿第0轴堆叠所有相机的图像序列 / Stack image sequences from all cameras along axis 0
        
        # 构建观测数据的张量表示 / Construct tensor representation of observation data
        image_data = torch.from_numpy(all_cam_images)  # 将numpy数组转换为PyTorch张量 / Convert numpy array to PyTorch tensor
        
        # 当前形状说明 / Current shape explanation:
        # image_data: (cam_num, context_len, 480, 640, RGB3) - (相机数量, 上下文长度, 图像高度, 图像宽度, RGB通道数)
        
        # 重排维度以符合模型期望的格式 / Rearrange dimensions to match model expected format
        image_data = torch.einsum('n k h w c -> k n c h w', image_data)  # 维度重排：(相机数, 上下文长度, 高, 宽, 通道) -> (上下文长度, 相机数, 通道, 高, 宽)
        
        # 最终形状说明 / Final shape explanation:
        # image_data: (context_len, cam_num, RGB3, 480, 640) - (上下文长度, 相机数量, RGB通道数, 图像高度, 图像宽度)
        
        image_data = image_data / 255.0  # 将像素值从[0,255]范围归一化到[0,1] / Normalize pixel values from [0,255] range to [0,1]

        # 处理深度图像数据 / Process depth image data
        image_depth_data = np.zeros(1, dtype=np.float32)  # 默认初始化为占位符数组 / Default initialize as placeholder array
        
        if self.use_depth_image:  # 如果使用深度图像 / If using depth images
            # 原始深度图像整合方案（已注释）/ Original depth image consolidation scheme (commented)
            # all_cam_images_depth = []  # 从字典中收集深度图像 / Collect depth images from dictionary
            # for cam_name in self.camera_names:  # 遍历相机名称 / Iterate camera names
            #     all_cam_images_depth.append(image_depth_dict[cam_name])  # 添加深度图像数据 / Add depth image data
            
            # 将嵌套列表转换为numpy数组 / Convert nested list to numpy array
            all_cam_images_depth = np.stack(all_cam_images_depth, axis=0)  # 沿第0轴堆叠所有相机的深度图像 / Stack depth images from all cameras along axis 0
            
            # 构建深度观测数据的张量表示 / Construct tensor representation of depth observation data
            image_depth_data = torch.from_numpy(all_cam_images_depth)  # 将numpy数组转换为PyTorch张量 / Convert numpy array to PyTorch tensor
            
            # 深度图像的维度重排（已注释，通常深度图像只有1个通道）/ Depth image dimension rearrangement (commented, depth images usually have only 1 channel)
            # image_depth_data = torch.einsum('k h w c -> k c h w', image_depth_data)  # 如果需要可取消注释 / Uncomment if needed
            
            image_depth_data = image_depth_data / 255.0  # 将深度值归一化（假设深度存储为0-255范围）/ Normalize depth values (assuming depth stored in 0-255 range)

        # 数据类型转换和归一化处理 / Data type conversion and normalization processing
        
        # 处理关节位置历史序列数据 / Process joint position historical sequence data
        qpos_data = torch.from_numpy(qpos).float()  # 转换为浮点型张量 / Convert to float tensor
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]  # Z-score归一化 / Z-score normalization
        
        # 处理前瞻动作数据 / Process lookahead action data
        next_action_data = torch.from_numpy(padded_next_action).float()  # 转换为浮点型张量 / Convert to float tensor
        next_action_is_pad = torch.from_numpy(next_action_is_pad).bool()  # 转换为布尔型张量（填充掩码）/ Convert to boolean tensor (padding mask)
        
        # 处理历史动作序列数据 / Process historical action sequence data
        action_data = torch.from_numpy(action).float()  # 转换为浮点型张量 / Convert to float tensor
        action_is_pad = torch.zeros([action_data.shape[0]]).bool()  # 创建动作填充掩码（历史序列通常不需要填充）/ Create action padding mask (historical sequences usually don't need padding)
        
        # 根据动作数据源类型进行不同的归一化 / Different normalization based on action data source type
        if self.use_dataset_action:  # 如果使用数据集中的动作标签 / If using action labels from dataset
            # 使用动作数据的统计信息进行归一化 / Normalize using action data statistics
            next_action_data = (next_action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
            action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        else:  # 如果使用关节位置差分作为动作 / If using joint position differences as actions
            # 使用关节位置数据的统计信息进行归一化 / Normalize using joint position data statistics
            next_action_data = (next_action_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
            action_data = (action_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # 调试断点（已注释）/ Debug breakpoint (commented)
        # import pdb; pdb.set_trace()  # Python调试器断点 / Python debugger breakpoint
        
        # 返回数据的形状说明 / Shape explanation for returned data
        # image_data.shape: (episode_len, cam_num=3, RGB=3, 480, 640) - (情节长度, 相机数量=3, RGB通道=3, 图像高度, 图像宽度)
        # qpos_data.shape: (episode_len, state_dim=14) - (情节长度, 状态维度=14) - 历史关节位置序列 / Historical joint position sequence
        # action_data.shape: (episode_len, action_dim=14) - (情节长度, 动作维度=14) - 历史动作序列 / Historical action sequence
        
        # 注意：与基础EpisodicDataset的主要区别在于所有数据都包含时间序列维度 / Note: Main difference from base EpisodicDataset is that all data contains temporal sequence dimension
        
        # 返回完整的数据元组 / Return complete data tuple
        return image_data, image_depth_data, qpos_data, next_action_data, next_action_is_pad, action_data, action_is_pad
