"""
BBoxHistoryEpisodicDataset.py

边界框历史情节数据集类，用于加载包含YOLO预处理的边界框数据的机器人学习数据。
该数据集专门处理通过YOLO检测生成的边界框信息，用于空间抓取任务的训练。

Bounding Box History Episodic Dataset class for loading robotics learning data
with YOLO-preprocessed bounding box information. This dataset specifically handles
bounding box information generated through YOLO detection for spatial grasping tasks.
"""

# 数值计算和机器学习库 / Numerical computing and ML libraries
import numpy as np  # 数值计算库 / Numerical computing library
import torch  # PyTorch深度学习框架 / PyTorch deep learning framework
import torch.nn as nn  # PyTorch神经网络模块 / PyTorch neural network modules

# 文件和数据处理库 / File and data processing libraries
import os  # 操作系统接口 / Operating system interface
import h5py  # HDF5文件格式处理 / HDF5 file format handling
import random  # 随机数生成 / Random number generation
import glob  # 文件路径模式匹配 / File path pattern matching
import contextlib  # 上下文管理工具 / Context management utilities

# PyTorch数据处理 / PyTorch data handling
from torch.utils.data import TensorDataset, DataLoader, Dataset

# 图像处理库 / Image processing libraries
import cv2  # OpenCV计算机视觉库 / OpenCV computer vision library
from PIL import Image  # Python图像库 / Python Imaging Library

# 调试工具 / Debugging tools
import IPython  # 交互式Python环境 / Interactive Python environment

# 项目特定导入 / Project-specific imports
import ManiBox  # ManiBox项目主模块 / ManiBox main project module

# 禁用YOLO详细输出 / Disable YOLO verbose output
os.environ['YOLO_VERBOSE'] = str(False)  # 设置环境变量以禁用YOLO预测输出 / Set environment variable to disable YOLO prediction output
from ultralytics import YOLO  # YOLO目标检测库 / YOLO object detection library

# 项目内部数据加载器导入 / Internal dataloader imports
from ManiBox.dataloader.EpisodicDataset import EpisodicDataset  # 基础情节数据集类 / Base episodic dataset class
from ManiBox.yolo_process_data import YoloProcessDataByTimeStep, KalmanFilter  # YOLO数据处理和卡尔曼滤波器 / YOLO data processing and Kalman filter

class BBoxHistoryEpisodicDataset(Dataset):
    """
    边界框历史情节数据集类
    
    用于加载和处理包含YOLO检测边界框信息的机器人学习数据集。
    该数据集将多个时间步的边界框数据组合成历史序列，支持空间抓取任务的训练。
    
    Bounding Box History Episodic Dataset Class
    
    Loads and processes robotics learning datasets containing YOLO detection bounding box information.
    This dataset combines bounding box data from multiple timesteps into historical sequences,
    supporting training for spatial grasping tasks.
    
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
        random_mask_ratio: 随机遮蔽比例 / Random masking ratio
    """

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, arm_delay_time, max_pos_lookahead,
                 use_dataset_action, use_depth_image, use_robot_base, episode_begin, episode_end, random_mask_ratio=0):
        # 基础数据集配置 / Basic dataset configuration
        self.episode_ids = episode_ids  # 情节ID列表，通常包含1000个情节 / List of episode IDs, typically contains 1000 episodes
        self.dataset_dir = dataset_dir  # 数据集根目录路径 / Dataset root directory path
        self.camera_names = camera_names  # 相机名称列表，如['cam_high', 'cam_left_wrist', 'cam_right_wrist'] / Camera names list
        self.norm_stats = norm_stats  # 归一化统计信息（在此数据集中未使用） / Normalization statistics (unused in this dataset)
        self.is_sim = None  # 是否为仿真数据的标志 / Flag indicating if data is from simulation
        
        # 数据处理配置 / Data processing configuration
        self.max_pos_lookahead = max_pos_lookahead  # 最大前瞻位置步数 / Maximum lookahead position steps
        self.use_dataset_action = use_dataset_action  # 是否使用数据集中的动作数据 / Whether to use action data from dataset
        self.use_depth_image = use_depth_image  # 是否使用深度图像数据 / Whether to use depth image data
        self.arm_delay_time = arm_delay_time  # 机械臂响应延迟时间 / Arm response delay time
        self.use_robot_base = use_robot_base  # 是否包含机器人底座信息 / Whether to include robot base information
        
        # 情节切片配置 / Episode slicing configuration
        self.episode_begin = episode_begin  # 情节开始时间步（避免边界情况） / Episode begin timestep (avoid edge cases)
        self.episode_end = episode_end  # 情节结束时间步 / Episode end timestep
        self.random_mask_ratio = random_mask_ratio  # 随机遮蔽边界框的比例（数据增强） / Random bbox masking ratio (data augmentation)
        
        # 加载预处理的积分数据文件 / Load preprocessed integration data file
        integration_path = os.path.join(self.dataset_dir, "integration.pkl")  # 构建数据文件路径 / Build data file path
        self.data = torch.load(integration_path, map_location='cpu')  # 加载数据到CPU内存 / Load data to CPU memory
        # 数据格式详见yolo_process_data.py中的process_data()函数 / Data format details in process_data() function in yolo_process_data.py
        print("Load data from", integration_path, "Shape: ", self.data["image_data"].shape)  # 打印数据加载信息 / Print data loading information
        
        # 通过获取第一个样本来初始化和验证数据格式 / Initialize and validate data format by getting first sample
        image_data, image_depth_data, qpos_data, next_action_data, next_action_is_pad, action_data, action_is_pad = self.__getitem__(0)
        print(f"image_data.shape, qpos_data.shape, action_data.shape: ", image_data.shape, qpos_data.shape, action_data.shape)  # 打印数据维度信息 / Print data dimension information

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
     
    def get_bbox_idx(self, cam_idx, obj_idx):
        """
        计算边界框在扁平化数组中的起始索引
        Calculate the starting index of a bounding box in the flattened array
        
        Args:
            cam_idx: 相机索引 (0, 1, 2对应不同相机) / Camera index (0, 1, 2 for different cameras)
            obj_idx: 物体索引 (0对应目标物体如苹果) / Object index (0 for target object like apple)
        
        Returns:
            边界框起始索引，边界框占用4个连续位置 [x1, y1, x2, y2]
            Bounding box starting index, bbox occupies 4 consecutive positions [x1, y1, x2, y2]
        """
        # 计算公式：相机索引 * 物体数量 * 边界框大小 + 物体索引 * 边界框大小
        # Formula: camera_index * object_count * bbox_size + object_index * bbox_size
        return cam_idx * self.object_num * 4 + obj_idx * 4 
     
    def __getitem__(self, index):
        """
        获取指定索引的数据样本
        Get data sample at specified index
        
        Args:
            index: 样本索引 / Sample index
        
        Returns:
            tuple: (image_data, image_depth_data, qpos_data, next_action_data, 
                   next_action_is_pad, action_data, action_is_pad)
        
        注意：此处的qpos和action数据未进行归一化处理
        Note: qpos and action data here are not normalized
        """
        episode_id = self.episode_ids[index]  # 获取当前索引对应的情节ID / Get episode ID for current index
        
        # 提取边界框图像数据 / Extract bounding box image data
        # self.data["image_data"]: 预处理的边界框张量数据 / Preprocessed bounding box tensor data
        image_data = self.data["image_data"][episode_id][self.episode_begin:self.episode_end]  # 形状: (episode_end-episode_begin, 24) / Shape: (episode_end-episode_begin, 24)
        
        # 测试代码：复制图像数据用于调试 / Test code: copy image data for debugging
        # test_image = image_data.clone()  # 克隆数据以避免修改原始数据 / Clone data to avoid modifying original
        # test_image = test_image.reshape(image_data.shape[0], 3, 2, 4)[0:10, 0, :, :]  # 重塑为(时间步, 3相机, 2物体(苹果和桌子), 4坐标) / Reshape to (timesteps, 3cameras, 2objects(apple and table), 4coordinates)
        # print("test_image: ", test_image)  # 打印测试图像数据 / Print test image data
        
        epi_len = image_data.shape[0]  # 获取当前情节的时间步长度 / Get timestep length of current episode
        
        # 处理物体检测数据的维度适配 / Handle dimension adaptation for object detection data
        if len(YoloProcessDataByTimeStep.objects_names) == 1 and image_data.shape[1] == 24:  # 如果只检测一个物体但数据包含两个物体 / If detecting one object but data contains two objects
            # 丢弃桌子的边界框，只保留目标物体（如苹果） / Discard table bbox, keep only target object (like apple)
            image_data = image_data.reshape(epi_len, 3, 2, 4)  # 重塑为 (时间步长, 相机数=3, 物体数=2, 坐标数=4) / Reshape to (episode_len, cam_num=3, object_num=2, coords=4)
            image_data = image_data[:, :, 0, :]  # 只取第一个物体（索引0）的边界框 / Take only first object (index 0) bounding boxes
            image_data = image_data.reshape(epi_len, 12)   # 重塑为 (时间步长, 12=3相机*4坐标) / Reshape to (episode_len, 12=3cameras*4coords)
        # 否则image_data已经只包含目标物体，维度为12 / Otherwise image_data already contains only target object, dimension is 12
        
        # 初始化深度图像数据（此数据集中未使用，仅为批处理兼容性） / Initialize depth image data (unused in this dataset, for batch compatibility only)
        image_depth_data = torch.zeros(1, dtype=torch.float32)  # 占位符张量，用于批次迭代 / Placeholder tensor for batch iteration
        
        # 提取关节位置数据 / Extract joint position data
        qpos_data = self.data["qpos_data"][episode_id][self.episode_begin:self.episode_end]  # 机器人关节位置时间序列 / Robot joint position time series
        
        # 初始化未使用的动作相关数据（为接口兼容性） / Initialize unused action-related data (for interface compatibility)
        next_action_data = torch.zeros(1, dtype=torch.float32)  # 下一步动作数据占位符 / Next action data placeholder
        next_action_is_pad = torch.zeros(1, dtype=torch.float32)  # 下一步动作填充标记占位符 / Next action padding mask placeholder
        action_is_pad = torch.zeros(1, dtype=torch.float32)  # 动作填充标记占位符 / Action padding mask placeholder
        
        # 提取动作轨迹数据 / Extract action trajectory data
        action_data = self.data["action_data"][episode_id][self.episode_begin:self.episode_end]  # 机器人动作轨迹时间序列 / Robot action trajectory time series
        
        # 计算数据维度参数 / Calculate data dimension parameters
        cam_num = len(self.camera_names)  # 相机数量，通常为3（高视角、左腕、右腕） / Number of cameras, typically 3 (high, left_wrist, right_wrist)
        bbox_size = 4  # 边界框大小，包含4个坐标值 [x1, y1, x2, y2] / Bounding box size, contains 4 coordinate values [x1, y1, x2, y2]
        object_num = round(image_data.shape[1] / bbox_size / cam_num)  # 计算物体数量 / Calculate number of objects
        
        # 存储维度参数供其他方法使用 / Store dimension parameters for use by other methods
        self.cam_num = cam_num  # 相机数量 / Number of cameras
        self.object_num = object_num  # 物体数量 / Number of objects
        
        # 测试代码：将右侧相机数据设为0 / Test code: set right camera data to 0
        # image_data[:, 8:12] = 0.0  # 索引8-12对应右腕相机的边界框数据 / Indices 8-12 correspond to right wrist camera bbox data
        
        # 数据增强：随机遮蔽边界框 / Data augmentation: randomly mask bounding boxes
        # 在一定概率下将某些边界框设置为(0,0,0,0)以增强模型鲁棒性 / Set some bounding boxes to (0,0,0,0) with certain probability to enhance model robustness
        if self.random_mask_ratio != 0:  # 如果设置了随机遮蔽比例 / If random masking ratio is set
            mask_ratio = self.random_mask_ratio * np.random.rand()  # 动态计算当前遮蔽比例 / Dynamically calculate current masking ratio
            
            # 以下为替代的遮蔽方案（已注释） / Alternative masking schemes (commented out)
            # mask_timesteps = np.random.choice(epi_len, int(epi_len*mask_ratio), replace=False)  # 随机选择要遮蔽的时间步 / Randomly select timesteps to mask
            # mask_cam_idx = np.random.choice(cam_num, int(cam_num*epi_len))  # 随机选择相机索引 / Randomly select camera indices
            # mask_obj_idx = np.random.choice(object_num, int(epi_len*mask_ratio))  # 随机选择物体索引 / Randomly select object indices
            # for i, mask_timestep in enumerate(mask_timesteps):  # 遍历每个要遮蔽的时间步 / Iterate through each timestep to mask
            #     bbox_idx = self.get_bbox_idx(mask_cam_idx[i], mask_obj_idx[i])  # 计算边界框索引 / Calculate bbox index
            #     image_data[mask_timestep, bbox_idx:bbox_idx+4] = torch.tensor([0, 0, 0, 0], dtype=torch.float32)  # 设置为零 / Set to zero
            # breakpoint()  # 调试断点 / Debug breakpoint
            # 实现更细粒度的随机遮蔽策略 / Implement fine-grained random masking strategy
            total_len = cam_num * epi_len  # 计算总的相机-时间步组合数 / Calculate total camera-timestep combinations
            cam_idx = np.full(total_len, -1)  # 初始化相机索引数组，-1表示不遮蔽 / Initialize camera index array, -1 means no masking
            
            # 随机选择要遮蔽的位置 / Randomly select positions to mask
            random_indices = np.random.choice(total_len, int(mask_ratio*total_len), replace=False)  # 无重复随机选择索引 / Select indices randomly without replacement
            random_values = np.random.randint(0, 1, size = int(mask_ratio*total_len))  # 生成随机遮蔽值 / Generate random masking values
            cam_idx[random_indices] = random_values  # 将随机值分配给选中的索引 / Assign random values to selected indices
            
            # 应用遮蔽操作 / Apply masking operations
            for i, mask_cam in enumerate(cam_idx):  # 遍历所有相机索引 / Iterate through all camera indices
                if mask_cam != -1:  # 如果当前位置需要遮蔽 / If current position needs masking
                    bbox_idx = self.get_bbox_idx(i%cam_num, 0)  # 计算边界框起始索引 / Calculate bbox starting index
                    traj_idx = int(i / cam_num)  # 计算轨迹（时间步）索引 / Calculate trajectory (timestep) index
                    
                    if traj_idx == 0 and bbox_idx == 0:
                        continue  # 第一个时间步的高位相机必须保持有效 / First timestep's high camera must remain valid
                    
                    # 将选中的边界框设置为零向量 / Set selected bounding box to zero vector
                    image_data[traj_idx, bbox_idx:bbox_idx+4] = torch.tensor([0, 0, 0, 0], dtype=torch.float32)        
        
        # 相机ID独热编码方案（已注释，备用） / Camera ID one-hot encoding scheme (commented out, for reference)
        # cam_ids = torch.tensor([0, 1, 2])  # 定义相机ID / Define camera IDs
        #
        # # 相机ID独热编码 / Camera ID one-hot encoding
        # # 形状: (相机数, 相机ID维度) => (3, 3) / Shape: (num_cams, cam_id_dim) => (3, 3)
        # cam_one_hot = nn.functional.one_hot(cam_ids, num_classes=cam_num).float()  # 生成独热编码 / Generate one-hot encoding
        # # cam_one_hot[0] 类似 [1, 0, 0] / cam_one_hot[0] is like [1, 0, 0]
        # # image_data: (episode_len, 12) / image_data: (episode_len, 12)
        # # 对于每个轨迹i，将相机编码与边界框数据拼接 / For each trajectory i, concatenate camera encoding with bbox data
        # # image_data[i] <- concat(image_data[i, 0:4], cam_one_hot[0], image_data[i, 4:8], cam_one_hot[1], image_data[i, 8:12], cam_one_hot[2])
        # image_data = torch.cat([image_data[:, 0:4], cam_one_hot[0].repeat(epi_len, 1), image_data[:, 4:8],
        #                         cam_one_hot[1].repeat(epi_len, 1), image_data[:, 8:12], cam_one_hot[2].repeat(epi_len, 1)], dim=1)
        # # 最终形状: (episode_len, 12+3+3+3=21) / Final shape: (episode_len, 12+3+3+3=21)
                    
        # 卡尔曼滤波器平滑边界框方案（已注释，备用） / Kalman filter for bbox smoothing (commented out, for reference)
        # kalman_filter_objects = [[KalmanFilter() for _ in range(object_num)] for _ in range(cam_num)]  # 为每个相机的每个物体创建卡尔曼滤波器 / Create Kalman filter for each object in each camera
        # 
        # for i in range(cam_num):  # 遍历所有相机 / Iterate through all cameras
        #     for j in range(object_num):  # 遍历所有物体 / Iterate through all objects
        #         # image_data.shape: (episode_len, 12) / image_data.shape: (episode_len, 12)
        #         for k in range(epi_len):  # 遍历所有时间步 / Iterate through all timesteps
        #             # 计算当前边界框在扁平数组中的位置 / Calculate current bbox position in flattened array
        #             # image_data[k, i*bbox_size*object_num+j*bbox_size: i*bbox_size*object_num+(j+1)*bbox_size] 是相机i中物体j的边界框
        #             # image_data[k, i*bbox_size*object_num+j*bbox_size: i*bbox_size*object_num+(j+1)*bbox_size] is bbox of object j in camera i
        #             bbox = image_data[k, i*bbox_size*object_num+j*bbox_size: i*bbox_size*object_num+(j+1)*bbox_size]  # 提取当前边界框 / Extract current bbox
        #             # 使用卡尔曼滤波器填充缺失的边界框数据 / Use Kalman filter to fill missing bbox data
        #             image_data[k, i*bbox_size*object_num+j*bbox_size: i*bbox_size*object_num+(j+1)*bbox_size] = kalman_filter_objects[i][j].fill_missing_bbox_with_kalman(bbox)
        
        # 数据归一化方案（已注释，此数据集中数据未归一化） / Data normalization scheme (commented out, data not normalized in this dataset)
        # qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]  # 关节位置数据归一化 / Joint position data normalization
        # action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]  # 动作数据归一化 / Action data normalization
        
        # 调试断点（已注释） / Debug breakpoint (commented out)
        # import pdb; pdb.set_trace()  # Python调试器断点 / Python debugger breakpoint
        
        # 数据形状说明 / Data shape explanation
        # qpos_data.shape: (episode_len, state_dim=14) - 关节位置数据，14维状态空间 / Joint position data, 14-dimensional state space
        # action_data.shape: (episode_len, action_dim=14) - 动作数据，14维动作空间 / Action data, 14-dimensional action space
        
        # 返回完整的数据元组 / Return complete data tuple
        return image_data, image_depth_data, qpos_data, next_action_data, next_action_is_pad, action_data, action_is_pad
