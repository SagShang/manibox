
"""YOLO物体检测数据处理模块 - YOLO Object Detection Data Processing Module

本模块实现了基于YOLO模型的机器人视觉数据处理系统。
主要用于从机器人操作数据中提取物体边界框信息，将原始图像转换为结构化的边界框数据，
支持实时推理和批量数据预处理，为机器人策略提供物体位置信息。

This module implements YOLO-based robot vision data processing system.
Primarily used to extract object bounding box information from robot manipulation data,
converting raw images to structured bounding box data, supporting both real-time inference
and batch data preprocessing, providing object position information for robot policies.

核心功能 / Core Functions:
- YOLO物体检测：使用YOLOv8进行实时物体检测 / YOLO object detection: real-time detection using YOLOv8
- 边界框处理：提取、过滤和格式化边界框数据 / Bounding box processing: extract, filter and format bbox data  
- 卡尔曼滤波：时序数据平滑处理 / Kalman filtering: temporal data smoothing
- 批量数据处理：支持大规模数据集预处理 / Batch processing: support large-scale dataset preprocessing
- HDF5数据集集成：与机器人数据格式兼容 / HDF5 dataset integration: compatible with robot data formats

主要类 / Main Classes:
- ProcessDataFromHDF5: 从HDF5数据集提取YOLO边界框 / Extract YOLO bboxes from HDF5 datasets
- YoloProcessDataByTimeStep: 按时间步处理YOLO数据 / Process YOLO data by timestep
- KalmanFilter: 卡尔曼滤波器用于边界框平滑 / Kalman filter for bbox smoothing
- YoloCollectData: 收集和管理YOLO处理的数据 / Collect and manage YOLO-processed data

应用场景 / Use Cases:
- 训练数据预处理：离线处理大量机器人演示数据 / Training data preprocessing: offline processing of robot demonstration data
- 实时推理：在线处理机器人视觉输入 / Real-time inference: online processing of robot visual input
- 物体追踪：跨帧跟踪目标物体位置 / Object tracking: track target object positions across frames
"""

# 标准库导入 / Standard library imports
import os                 # 操作系统接口 / Operating system interface
import random             # 随机数生成 / Random number generation
import cv2                # OpenCV计算机视觉库 / OpenCV computer vision library
import h5py               # HDF5文件处理 / HDF5 file processing
import contextlib         # 上下文管理工具 / Context management utilities

# 数值计算和深度学习库 / Numerical computation and deep learning libraries
import numpy as np        # 数值计算库 / Numerical computation library
import torch              # PyTorch深度学习框架 / PyTorch deep learning framework
from torch.utils.data import Dataset  # PyTorch数据集基类 / PyTorch dataset base class
from tqdm import tqdm     # 进度条显示 / Progress bar display
from PIL import Image, ImageDraw, ImageFont  # Python图像处理库 / Python Image Library

# YOLO模型设置和导入 / YOLO model setup and import
os.environ['YOLO_VERBOSE'] = str(False)  # 禁用YOLO预测输出 / Disable YOLO prediction output
from ultralytics import YOLO  # YOLOv8模型库 / YOLOv8 model library

class ProcessDataFromHDF5:
    """HDF5数据集YOLO处理器 - HDF5 Dataset YOLO Processor
    
    从HDF5格式的机器人演示数据中提取视觉信息，使用YOLO模型进行物体检测，
    并将检测结果转换为适合机器学习训练的格式。支持批量处理多个回合的数据。
    
    Extract visual information from HDF5-formatted robot demonstration data, use YOLO model 
    for object detection, and convert detection results to formats suitable for machine learning training.
    Supports batch processing of multiple episode data.
    
    工作流程 / Workflow:
    1. 加载HDF5数据集中的图像和动作数据 / Load image and action data from HDF5 dataset
    2. 使用YOLO模型检测每帧图像中的物体 / Use YOLO model to detect objects in each frame
    3. 应用卡尔曼滤波器平滑边界框轨迹 / Apply Kalman filter to smooth bounding box trajectories
    4. 将结果保存为PyTorch张量格式 / Save results in PyTorch tensor format
    
    特点 / Features:
    - 支持压缩和非压缩的HDF5数据 / Support both compressed and uncompressed HDF5 data
    - 集成卡尔曼滤波器提高检测稳定性 / Integrate Kalman filter for improved detection stability
    - 批量处理提高效率 / Batch processing for improved efficiency
    """

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, arm_delay_time, max_pos_lookahead,
                 use_dataset_action, use_depth_image, use_robot_base, episode_begin, episode_end):
        """初始化HDF5数据处理器 / Initialize HDF5 data processor
        
        Args:
            episode_ids (list): 要处理的回合ID列表 / List of episode IDs to process
            dataset_dir (str): 数据集目录路径 / Dataset directory path
            camera_names (list): 相机名称列表 / Camera names list  
            norm_stats (dict): 数据正则化统计信息 / Data normalization statistics
            arm_delay_time (int): 机械臂延迟时间 / Robot arm delay time
            max_pos_lookahead (int): 最大位置前瞻 / Maximum position lookahead
            use_dataset_action (bool): 是否使用数据集动作 / Whether to use dataset actions
            use_depth_image (bool): 是否使用深度图像 / Whether to use depth images
            use_robot_base (bool): 是否使用机器人底座 / Whether to use robot base
            episode_begin (int): 回合开始索引 / Episode begin index
            episode_end (int): 回合结束索引 / Episode end index
        """
        # 回合相关配置 / Episode-related configuration
        self.episode_ids = episode_ids              # 回合ID列表 / Episode ID list
        self.num_episodes = len(episode_ids)        # 总回合数 / Total number of episodes
        self.episode_len = episode_end - episode_begin  # 每个回合长度 / Length of each episode
        
        # 数据集配置 / Dataset configuration
        self.dataset_dir = dataset_dir              # 数据集根目录 / Dataset root directory
        self.camera_names = camera_names            # 相机名称列表 / Camera names list
        self.norm_stats = norm_stats                # 数据正则化统计信息 / Data normalization statistics
        self.is_sim = None                          # 是否为仿真数据（运行时确定）/ Whether simulation data (determined at runtime)
        
        # 动作和状态配置 / Action and state configuration
        self.max_pos_lookahead = max_pos_lookahead  # 最大前瞻步数 / Maximum lookahead steps
        self.use_dataset_action = use_dataset_action  # 使用数据集动作还是状态 / Use dataset actions or states
        self.use_depth_image = use_depth_image      # 是否处理深度图像 / Whether to process depth images
        self.arm_delay_time = arm_delay_time        # 机械臂响应延迟 / Robot arm response delay
        self.use_robot_base = use_robot_base        # 是否包含移动底座 / Whether to include mobile base
        
        # 时间索引配置 / Time index configuration
        self.episode_begin = episode_begin          # 回合开始时间戳 / Episode begin timestamp
        self.episode_end = episode_end              # 回合结束时间戳 / Episode end timestamp
        
        # YOLO检测模型初始化 / YOLO detection model initialization
        self.detection_model = YOLO("yolov8l-world.pt")    # 加载YOLOv8大模型 / Load YOLOv8 large model
        self.detection_model.set_classes(["apple", "table"])  # 设置检测类别 / Set detection classes
        
        # self.process_item(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)
    
    def shuffle(self):
        np.random.shuffle(self.episode_ids)
    
    def process_data(self):
        data = {
            "image_data": torch.zeros(self.num_episodes, self.episode_len, 24),  # Tensor, shape: (num_episodes, episode_len, 24)
            "image_depth_data": None,
            "qpos_data": torch.zeros(self.num_episodes, self.episode_len, 14),  # Tensor, shape: (num_episodes, episode_len, 14)
            "next_action_data": None,
            "next_action_is_pad": None,
            "action_data": torch.zeros(self.num_episodes, self.episode_len, 14),   # Tensor, shape: (num_episodes, episode_len, 14)
            "action_is_pad": None,
        }
        for episode_id in tqdm(self.episode_ids):
            self.process_item(data, episode_id)
        # save the data to (dataset_dir, "integration.pkl")
        torch.save(data, os.path.join(self.dataset_dir, f"integration.pkl"))
        print(f"Successfully saved the {self.num_episodes} processed data to ", os.path.join(self.dataset_dir, f"integration.pkl"))
    
    def process_item(self, data, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            is_compress = root.attrs['compress']
            original_action_shape = root['/action'][self.episode_begin:self.episode_end].shape
            max_action_len = original_action_shape[0]
            if self.use_robot_base:
                original_action_shape = (original_action_shape[0], original_action_shape[1] + 2)

            start_ts = max_action_len
            history_start = 0
            next_action_size = random.randint(0, self.max_pos_lookahead)
            if self.use_dataset_action:
                actions = root['/action'][self.episode_begin:self.episode_end]
            else:
                actions = root['/observations/qpos'][self.episode_begin:self.episode_end][1:]
                actions = np.append(actions, actions[-1][np.newaxis, :], axis=0)
            end_next_action_index = min(start_ts + next_action_size, max_action_len - 1)
            
            qpos = root['/observations/qpos'][self.episode_begin:self.episode_end][history_start: start_ts + 1]
            if self.use_robot_base:
                qpos = np.concatenate((qpos, root['/base_action'][self.episode_begin:self.episode_end][start_ts]), axis=0)
            all_cam_images = []
            all_cam_images_depth = []
            all_cam_bbox_history_apple = []
            all_cam_bbox_history_table = []
            
            for cam_name in self.camera_names:
                all_cam_images.append([])
                all_cam_bbox_history_apple.append([])
                all_cam_bbox_history_table.append([])
                kalman_filter_apple = KalmanFilter()
                kalman_filter_table = KalmanFilter()
                if is_compress:
                    decoded_image = root[f'/observations/images/{cam_name}'][self.episode_begin:self.episode_end][history_start: start_ts + 1]
                    for i in range(decoded_image.shape[0]):
                        image = cv2.imdecode(np.frombuffer(decoded_image[i], np.uint8), 1)
                        bbox_apple, bbox_table = self.detect_bounding_boxes(image)
                        all_cam_bbox_history_apple[-1].append(kalman_filter_apple.fill_missing_bbox_with_kalman(bbox_apple))
                        all_cam_bbox_history_table[-1].append(kalman_filter_table.fill_missing_bbox_with_kalman(bbox_table))
                        # all_cam_images[-1].append(image)
                else:  # ERROR here
                    image = root[f'/observations/images/{cam_name}'][self.episode_begin:self.episode_end][start_ts]
                    bbox_apple, bbox_table = self.detect_bounding_boxes(image)
                    all_cam_bbox_history_apple[-1].append(kalman_filter_apple.fill_missing_bbox_with_kalman(bbox_apple))
                    all_cam_bbox_history_table[-1].append(kalman_filter_table.fill_missing_bbox_with_kalman(bbox_table))
                    # all_cam_images[-1].append(image)

                if self.use_depth_image:
                    all_cam_images_depth.append([])
                    all_cam_images_depth[-1].append(root[f'/observations/images_depth/{cam_name}'][self.episode_begin:self.episode_end][start_ts])

                # all_cam_bbox_history_apple[-1] = self.fill_missing_bboxes_with_kalman(all_cam_bbox_history_apple[-1])
                # all_cam_bbox_history_table[-1] = self.fill_missing_bboxes_with_kalman(all_cam_bbox_history_table[-1])

            action = actions[history_start: start_ts + 1] 
            next_action = actions[start_ts: start_ts]
            if self.use_robot_base:
                action = np.concatenate((action, root['/base_action'][self.episode_begin:self.episode_end][index:]), axis=1)
                next_action = np.concatenate((next_action, root['/base_action'][self.episode_begin:self.episode_end][start_ts:index]), axis=1)

        self.is_sim = is_sim
        padded_next_action = np.zeros((self.max_pos_lookahead, original_action_shape[1]), dtype=np.float32)
        next_action_is_pad = np.zeros(self.max_pos_lookahead)
        # all_cam_images = np.stack(all_cam_images, axis=0)
        # image_data = torch.from_numpy(all_cam_images)
        # image_data = torch.einsum('n k h w c -> k n c h w', image_data)
        # image_data = image_data / 255.0

        image_depth_data = np.zeros(1, dtype=np.float32)
        if self.use_depth_image:
            all_cam_images_depth = np.stack(all_cam_images_depth, axis=0)
            image_depth_data = torch.from_numpy(all_cam_images_depth)
            image_depth_data = image_depth_data / 255.0

        qpos_data = torch.from_numpy(qpos).float()
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        
        next_action_data = torch.from_numpy(padded_next_action).float()
        next_action_is_pad = torch.from_numpy(next_action_is_pad).bool()
        
        action_data = torch.from_numpy(action).float()
        action_is_pad = torch.zeros([action_data.shape[0]]).bool()
        
        if self.use_dataset_action:
            next_action_data = (next_action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
            action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        else:
            next_action_data = (next_action_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
            action_data = (action_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # image_data.shape: (episode_len, cam_num=3, RGB=3, 480, 640)
        
        all_cam_bbox_history_apple = torch.tensor(all_cam_bbox_history_apple).float()  # (3, episode_len, 4)
        all_cam_bbox_history_table = torch.tensor(all_cam_bbox_history_table).float()
        # make the episode_len dim to be the first dim
        all_cam_bbox_history_apple = all_cam_bbox_history_apple.permute(1, 0, 2)  # (episode_len, 3, 4)
        all_cam_bbox_history_table = all_cam_bbox_history_table.permute(1, 0, 2)
        image_data = torch.cat((all_cam_bbox_history_apple, all_cam_bbox_history_table), dim=-1)  # (episode_len, 3, 8)
        # flatten the image in each timestep
        image_data = image_data.view(image_data.shape[0], -1)  # (episode_len, 24)
        
        # import pdb; pdb.set_trace()
        # qpos.shape: (episode_len, state_dim=14)
        # action_data.shape: (episode_len, action_dim=14)
        data["image_data"][index] = image_data
        data["qpos_data"][index] = qpos_data
        data["action_data"][index] = action_data
        # return image_data, image_depth_data, qpos_data, next_action_data, next_action_is_pad, action_data, action_is_pad

    def detect_bounding_boxes(self, image):
        # with contextlib.redirect_stdout(None):
        with torch.no_grad():
            results = self.detection_model.predict(image)

        bbox_apple, bbox_table = None, None
        for box in results[0].boxes:
            if results[0].names[box.cls.item()] == "apple":
                bbox_apple = box.xyxyn.squeeze().cpu().numpy().tolist()
            elif results[0].names[box.cls.item()] == "table":
                bbox_table = box.xyxyn.squeeze().cpu().numpy().tolist()
        return bbox_apple, bbox_table


class YoloProcessDataByTimeStep:
    """按时间步的YOLO数据处理器 - YOLO Data Processor by Timestep
    
    实时机器人推理中的核心视觉处理组件。负责在每个时间步对多相机图像进行物体检测，
    提取目标物体的边界框信息，并转换为机器人策略可使用的结构化特征。
    支持动态目标切换和实时处理优化。
    
    Core visual processing component in real-time robot inference. Responsible for object detection
    on multi-camera images at each timestep, extracting bounding box information of target objects,
    and converting to structured features usable by robot policies. Supports dynamic target switching
    and real-time processing optimization.
    
    核心特性 / Core Features:
    - 实时物体检测：单步和批量检测支持 / Real-time object detection: single-step and batch detection support
    - 多相机处理：同时处理多视角图像 / Multi-camera processing: simultaneous multi-view image processing
    - 目标管理：动态切换检测目标 / Target management: dynamic detection target switching
    - 卡尔曼滤波：可选的轨迹平滑 / Kalman filtering: optional trajectory smoothing
    - 内存优化：批量处理避免内存溢出 / Memory optimization: batch processing to avoid overflow
    
    使用场景 / Use Cases:
    - 实时推理：机器人执行任务时的在线视觉处理 / Real-time inference: online visual processing during task execution
    - 数据收集：收集训练数据时的视觉特征提取 / Data collection: visual feature extraction during training data collection
    """
    
    # 默认检测目标物体列表（将目标物体放在第一位）/ Default detection target objects (put goal object first)
    objects_names = ["apple"]  # 苹果作为默认抓取目标 / Apple as default grasping target
    
    def __init__(self, detection_model=None, objects_names=None):
        """初始化按时间步的YOLO处理器 / Initialize timestep YOLO processor
        
        Args:
            detection_model (YOLO, optional): 预训练的YOLO模型，如果为None则加载默认模型 / Pre-trained YOLO model, load default if None
            objects_names (str/list, optional): 检测目标名称，可以是字符串或列表 / Detection target names, can be string or list
        """
        # YOLO检测模型初始化 / YOLO detection model initialization
        if detection_model == None:
            # 使用默认的YOLOv8 Large World模型 / Use default YOLOv8 Large World model
            self.detection_model = YOLO("yolov8l-world.pt")
        else:
            # 使用传入的预训练模型 / Use provided pre-trained model
            self.detection_model = detection_model
        
        # 检测目标配置 / Detection target configuration
        if objects_names:
            # 如果传入目标名称，进行格式转换和设置 / If target names provided, convert format and set
            if isinstance(objects_names, str):
                objects_names = [objects_names]  # 字符串转列表 / Convert string to list
            self.objects_names = objects_names  # 设置检测目标 / Set detection targets
            print("grasping target:", self.objects_names)  # 打印抓取目标 / Print grasping target
        
        # 相机配置 / Camera configuration
        self.cameras_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]  # 固定的三相机配置 / Fixed three-camera configuration
        
        # 物体索引映射 / Object index mapping
        self.object_to_idx = {obj: idx for idx, obj in enumerate(self.objects_names)}  # 物体名称到索引的映射 / Object name to index mapping
        
        # 设置YOLO模型的检测类别 / Set YOLO model detection classes
        self.detection_model.set_classes(self.objects_names)
        
        # 卡尔曼滤波开关 / Kalman filtering switch
        self.using_kalman_filter = False  # 默认关闭卡尔曼滤波 / Kalman filtering disabled by default
    
    def modify_objects_names(self, objects_names):
        """动态修改检测目标 / Dynamically modify detection targets
        
        在运行时更改YOLO模型的检测目标。这对于需要抓取不同物体的任务很有用，
        用户可以在实时推理中切换检测目标而无需重新初始化模型。
        
        Change YOLO model detection targets at runtime. This is useful for tasks requiring
        grasping different objects, allowing users to switch detection targets during
        real-time inference without model reinitialization.
        
        Args:
            objects_names (str/list): 新的检测目标名称 / New detection target names
        """
        # 格式转换：确保输入为列表格式 / Format conversion: ensure input is in list format
        if isinstance(objects_names, str):
            objects_names = [objects_names]  # 单个字符串转换为列表 / Convert single string to list
        
        # 更新检测目标配置 / Update detection target configuration
        self.objects_names = objects_names  # 设置新的目标列表 / Set new target list
        self.object_to_idx = {obj: idx for idx, obj in enumerate(self.objects_names)}  # 重建索引映射 / Rebuild index mapping
        self.detection_model.set_classes(self.objects_names)  # 更新YOLO模型的检测类别 / Update YOLO model detection classes
    
    def reset_new_episode(self):
        """重置新回合的状态 / Reset state for new episode
        
        在开始新的数据采集回合时调用，清理上一回合的状态数据。
        如果启用了卡尔曼滤波，会重新初始化所有的滤波器。
        
        Called at the beginning of a new data collection episode to clean up state
        data from the previous episode. If Kalman filtering is enabled, all filters
        will be reinitialized.
        """
        # 如果启用了卡尔曼滤波，重新初始化滤波器 / If Kalman filtering is enabled, reinitialize filters
        if self.using_kalman_filter:
            # 为每个相机和每个目标物体创建独立的卡尔曼滤泥器 / Create independent Kalman filters for each camera and each target object
            self.kalman_filter_objects = [[KalmanFilter() for _ in range(len(self.objects_names))] for _ in range(len(self.cameras_names))]
        # 形状: (3 cameras, object_num) / Shape: (3 cameras, object_num)
    
    def detect_bounding_boxes(self, image):
        """单张图像的边界框检测 / Bounding box detection for single image
        
        对单张图像执行YOLO检测，返回目标物体的边界框信息。
        支持NumPy数组和PyTorch张量两种输入格式。
        
        Performs YOLO detection on a single image and returns bounding box information
        for target objects. Supports both NumPy array and PyTorch tensor input formats.
        
        Args:
            image: 输入图像，可以是numpy.ndarray或torch.Tensor / Input image, can be numpy.ndarray or torch.Tensor
            
        Returns:
            检测结果 / Detection results
        """
        # 根据输入数据类型处理 / Process based on input data type
        if isinstance(image, np.ndarray):
            # NumPy数组输入：封装为列表进行批量处理 / NumPy array input: wrap as list for batch processing
            return self.parallel_detect_bounding_boxes([image])
        elif isinstance(image, torch.Tensor):
            # PyTorch张量输入：添加batch维度 / PyTorch tensor input: add batch dimension
            if len(image.shape) == 3:  # 单张图像，需要添加batch维度 / Single image, need to add batch dimension
                image = image.unsqueeze(0)
            return self.parallel_detect_bounding_boxes(image)

    def process(self, cam_high, cam_left_wrist, cam_right_wrist):
        """处理单时间步的多相机图像 / Process multi-camera images at single timestep
        
        这是实时推理中最常用的接口函数。接收三个相机的图像输入，
        进行YOLO物体检测，返回结构化的边界框特征用于机器人策略推理。
        
        This is the most commonly used interface function in real-time inference.
        Takes three camera images as input, performs YOLO object detection, 
        and returns structured bounding box features for robot policy inference.
        
        Args:
            cam_high: 高位相机图像 / High camera image
            cam_left_wrist: 左手腕相机图像 / Left wrist camera image  
            cam_right_wrist: 右手腕相机图像 / Right wrist camera image
            
        Returns:
            torch.Tensor: 边界框特征张量 / Bounding box feature tensor
        """
        # 根据输入数据类型选择处理方式 / Choose processing method based on input data type
        if isinstance(cam_high, np.ndarray):
            # NumPy数组输入：转换为列表格式进行批量处理 / NumPy array input: convert to list format for batch processing
                # NumPy数组输入：封装为列表格式进行批量处理 / NumPy array input: wrap as list format for batch processing
            return self.parallel_process_traj([cam_high], [cam_left_wrist], [cam_right_wrist])
        elif isinstance(cam_high, torch.Tensor):
            # PyTorch张量输入：添加批次维度进行批量处理 / PyTorch tensor input: add batch dimension for batch processing
            return self.parallel_process_traj(cam_high.unsqueeze(0), cam_left_wrist.unsqueeze(0), cam_right_wrist.unsqueeze(0))

    def parallel_detect_bounding_boxes(self, images):
        """并行批量边界框检测 / Parallel batch bounding box detection
        
        高效地对批量图像执行YOLO检测。支持两种输入格式：
        1. NumPy数组列表: 适用于不同尺寸的图像批量处理
        2. PyTorch张量: 适用于固定尺寸的GPU加速处理
        
        Efficiently performs YOLO detection on batched images. Supports two input formats:
        1. NumPy array list: suitable for batch processing of different-sized images
        2. PyTorch tensor: suitable for GPU-accelerated processing of fixed-size images
        
        Args:
            images: 批量图像输入 / Batch image input
            
        Returns:
            批量检测结果 / Batch detection results
        """
        batch_size = len(images)  # 获取批量大小 / Get batch size
        
        # NumPy数组输入格式验证和预处理 / NumPy array input format validation and preprocessing
        if isinstance(images[0], np.ndarray):
            assert len(images[0].shape) == 3, "图像必须为3维数组 / Images must be 3D arrays"  # 高度、宽度、通道 / Height, width, channels
            assert images[0].shape[-1] == 3, "图像必须为3通道RGB格式 / Images must be 3-channel RGB format"
            # 颜色空间转换：RGB转BGR（OpenCV默认格式）/ Color space conversion: RGB to BGR (OpenCV default format)
            for i in range(len(images)):
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        
        # PyTorch张量输入格式验证 / PyTorch tensor input format validation
        if isinstance(images, torch.Tensor):
            assert images.shape[1] == 3, "张量第二维必须为3通道 / Second dimension of tensor must be 3 channels"  # (batch, channels, height, width)
            assert len(images.shape) == 4, "张量必须为4维 / Tensor must be 4-dimensional"  # (batch, channels, height, width)
            
        # 执行YOLO检测 / Execute YOLO detection
        with torch.no_grad():  # 禁用梯度计算以节省内存 / Disable gradient computation to save memory
            results = self.detection_model.predict(images, batch=batch_size)  # 批量检测 / Batch detection
        
        # 处理检测结果 / Process detection results
        batched_bbox_list = []  # 存储所有图像的边界框结果 / Store bounding box results for all images
        for i in range(batch_size):  # 遍历每张图像 / Iterate through each image
            # 初始化边界框列表，默认为无边界框 / Initialize bbox list, default to no bbox
            bbox_list = [KalmanFilter.NO_BBOX for _ in range(len(self.objects_names))]
            
            # 处理每个检测到的边界框 / Process each detected bounding box
            for box in results[i].boxes:
                bbox = box.xyxyn.squeeze().cpu().numpy().tolist()  # 归一化边界框坐标 (x1,y1,x2,y2) / Normalized bbox coordinates (x1,y1,x2,y2)
                name = results[i].names[round(box.cls.item())]  # 获取检测物体名称 / Get detected object name
                # 如果检测到的物体在目标列表中，保存边界框 / If detected object is in target list, save bbox
                if name in self.object_to_idx:
                    bbox_list[self.object_to_idx[name]] = bbox  # 按物体索引存储边界框 / Store bbox by object index
            batched_bbox_list.append(bbox_list)  # 添加到批量结果中 / Add to batch results
        return batched_bbox_list  # 返回形状: (batch_size, objects_num, 4) / Return shape: (batch_size, objects_num, 4)
    
    def parallel_process_traj(self, cams_high, cams_left_wrist, cams_right_wrist):
        """并行处理整个轨迹的多相机图像 / Parallel processing of multi-camera images for entire trajectory
        
        这是批量数据处理的核心函数，高效地处理整个轨迹序列的多相机图像。
        支持卡尔曼滤波来平滑边界框轨迹，提高检测的时间一致性。
        
        This is the core function for batch data processing, efficiently processing
        multi-camera images for entire trajectory sequences. Supports Kalman filtering
        to smooth bounding box trajectories and improve temporal consistency.
        
        Args:
            cams_high: 高位相机图像序列 / High camera image sequence
            cams_left_wrist: 左腕相机图像序列 / Left wrist camera image sequence
            cams_right_wrist: 右腕相机图像序列 / Right wrist camera image sequence
            
        Returns:
            torch.Tensor: 特征化的边界框数据 / Featurized bounding box data
        """
        # 批处理参数设置 / Batch processing parameter setup
        batch_size = 96  # 防止内存溢出的批处理大小 / Batch size to avoid memory overflow
        epi_len = len(cams_high)  # 回合长度 / Episode length
        cam_num = len(self.cameras_names)  # 相机数量 = 3 / Number of cameras = 3
        objects_num = len(self.objects_names)  # 目标物体数量 / Number of target objects
        
        # 初始化边界框存储结构 / Initialize bounding box storage structure
        # 形状: (epi_len, cam_num, objects_num, xyxyn) / Shape: (epi_len, cam_num, objects_num, xyxyn)
        batched_cam_bbox_objects_list = [[[] for _ in range(cam_num)] for _ in range(epi_len)]
        
        # 根据输入数据类型进行相机数据拼接 / Concatenate camera data based on input data type
        if isinstance(cams_high, list):
            # 列表格式：直接拼接三个相机的图像列表 / List format: directly concatenate three camera image lists
            cams = cams_high + cams_left_wrist + cams_right_wrist  # 形状: (3 * epi_len, 480, 640, 3) / Shape: (3 * epi_len, 480, 640, 3)
        elif isinstance(cams_high, torch.Tensor):
            # PyTorch张量格式：沿批次维度拼接三个相机 / PyTorch tensor format: concatenate three cameras along batch dimension
            cams = torch.cat((cams_high, cams_left_wrist, cams_right_wrist), dim=0)  # 形状: (3 * epi_len, 3, 480, 640) / Shape: (3 * epi_len, 3, 480, 640)
        elif isinstance(cams_high, np.ndarray):
            # NumPy数组格式：沿第0轴拼接三个相机 / NumPy array format: concatenate three cameras along axis 0
            cams = np.concatenate((cams_high, cams_left_wrist, cams_right_wrist), axis=0)
        
        # 维度转换备用代码（已注释）/ Dimension conversion backup code (commented)
        # if cams[0].shape[0] == 3:  # 检查是否为通道第一的格式 / Check if format is channel-first
        #     for i in range(len(cams)):  # 适用于torch.Tensor / For torch.Tensor
        #         cams[i] = cams[i].permute(1, 2, 0).float()  # 转换为通道最后的格式 / Convert to channel-last format
        
        # 批量检测处理（原始方法，已注释）/ Batch detection processing (original method, commented)
        # batched_bbox_list = self.parallel_detect_bounding_boxes(cams)  # (3 * epi_len, objects_num, 4)
        
        # 分批处理图像以避免内存溢出 / Process images in batches to avoid memory overflow
        batched_bbox_list = []  # 存储所有检测结果 / Store all detection results
        for start_idx in range(0, len(cams), batch_size):  # 按批大小分割处理 / Process in chunks by batch size
            batch_images = cams[start_idx:start_idx + batch_size]  # 提取当前批次图像 / Extract current batch images
            batch_bbox_list = self.parallel_detect_bounding_boxes(batch_images)  # 对当前批次进行检测 / Detect current batch
            batched_bbox_list.extend(batch_bbox_list)  # 扩展结果列表 / Extend result list
        # batched_bbox_list: (3 * epi_len, objects_num, 4) - 所有相机和时间步的检测结果 / Detection results for all cameras and timesteps
        
        # 数据重整和滤波处理 / Data reorganization and filtering processing
        if self.using_kalman_filter:
            # 使用卡尔曼滤波模式 / Using Kalman filtering mode
            
            # 重整数据结构：从相机优先变为时间优先 / Reorganize data structure: from camera-first to time-first
            for i in range(epi_len):  # 遍历时间步 / Iterate through timesteps
                for j in range(cam_num):  # 遍历相机 / Iterate through cameras
                    # 重新排列数据索引 / Rearrange data indices
                    batched_cam_bbox_objects_list[i][j] = batched_bbox_list[j * epi_len + i]
            # batched_cam_bbox_objects_list: (epi_len, cam_num, objects_num, 4)
            
            # 沿时间维度应用卡尔曼滤波 / Apply Kalman filter along time dimension
            for j in range(cam_num):  # 对每个相机 / For each camera
                for k in range(len(self.objects_names)):  # 对每个目标物体 / For each target object
                    kalman_filter = KalmanFilter()  # 为当前相机-物体组合创建滤泥器 / Create filter for current camera-object combination
                    for i in range(epi_len):  # 沿时间序列进行滤波 / Filter along time sequence
                        # 对缺失的边界框进行卡尔曼滤波补全 / Fill missing bboxes with Kalman filtering
                        batched_cam_bbox_objects_list[i][j][k] = kalman_filter.fill_missing_bbox_with_kalman(batched_cam_bbox_objects_list[i][j][k])
                
            # 转换为PyTorch张量 / Convert to PyTorch tensor
            batched_cam_bbox_objects_list = torch.tensor(batched_cam_bbox_objects_list).float()  # (epi_len, cam_num, objects_num, 4)
        else:
            # 不使用卡尔曼滤波模式 / Not using Kalman filtering mode
            batched_cam_bbox_objects_list = torch.tensor(batched_bbox_list).float()  # (3 * epi_len, objects_num, 4)
            
            # 数据排列说明 / Data arrangement explanation:
            # 列表中的边界框排列为: [cam_high_0, cam_high_1, ..., cam_left_wrist_0, cam_left_wrist_1, ..., cam_right_wrist_0, cam_right_wrist_1, ...]
            # NOTE that the bbox in the list is: [cam_high_0, cam_high_1, ..., cam_left_wrist_0, cam_left_wrist_1, ..., cam_right_wrist_0, cam_right_wrist_1, ...]
            
            # 重塑数据维度 / Reshape data dimensions
            batched_cam_bbox_objects_list = batched_cam_bbox_objects_list.reshape(cam_num, epi_len, objects_num, -1)  # (cam_num, epi_len, objects_num, 4)
            # 转置维度顺序 / Permute dimension order
            batched_cam_bbox_objects_list = batched_cam_bbox_objects_list.permute(1, 0, 2, 3)  # (epi_len, cam_num, objects_num, 4)

        # 最终维度展平和返回 / Final dimension flattening and return
        batched_cam_bbox_objects_list = batched_cam_bbox_objects_list.reshape(epi_len, -1)  # (epi_len, cam_num * objects_num * 4)
        # 将最后三个维度展平为一维特征向量，方便机器人策略网络使用 / Flatten last three dimensions to 1D feature vector for robot policy network usage
        return batched_cam_bbox_objects_list

class KalmanFilter:
    """
    卡尔曼滤波器类 / Kalman Filter Class
    
    用于对边界框轨迹进行时间序列滤波，提高检测的稳定性和连续性。
    当YOLO检测在某些帧中失败或不稳定时，利用卡尔曼滤泥器的预测和更新机制
    来估计和平滑边界框的位置和尺寸。
    
    Used for temporal filtering of bounding box trajectories to improve detection
    stability and continuity. When YOLO detection fails or is unstable in some frames,
    utilizes Kalman filter's prediction and update mechanisms to estimate and smooth
    bounding box positions and sizes.
    
    特性 / Features:
    - 动态状态估计 / Dynamic state estimation
    - 缺失检测补全 / Missing detection interpolation
    - 噪声抑制 / Noise suppression
    - 轨迹平滑 / Trajectory smoothing
    
    TODO: 检查和优化算法参数 / Check and optimize algorithm parameters
    """
    # 无边界框的默认值 / Default value for no bounding box
    NO_BBOX = [0, 0, 0, 0]  # [x1, y1, x2, y2] 归一化坐标的空值 / Normalized coordinates null value
    def __init__(self):
        self.prev_bbox = KalmanFilter.NO_BBOX
        self.kalman = self.create_kalman_filter()
    
    def create_kalman_filter(self):
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        return kalman

    def fill_missing_bbox_with_kalman(self, bbox):
        # without for-loop here
        filled_bbox = KalmanFilter.NO_BBOX
        if bbox is KalmanFilter.NO_BBOX:
            if self.prev_bbox is not KalmanFilter.NO_BBOX:
                prediction = self.kalman.predict()
                bbox = [prediction[0][0], prediction[1][0], prediction[0][0] + self.prev_bbox[2], prediction[1][0] + self.prev_bbox[3]]
                filled_bbox = bbox
            else:
                filled_bbox = [0, 0, 0, 0]
        else:
            self.kalman.correct(np.array([[np.float32(bbox[0])], [np.float32(bbox[1])]], np.float32))
            filled_bbox = bbox
            self.prev_bbox = bbox
        return filled_bbox

import numpy as np
import cv2

import os
os.environ['YOLO_VERBOSE'] = str(False)  # disable the output of yolo predict
from ultralytics import YOLO
import h5py
import time


class AsyncYoloProcessDataFromHDF5:
    def __init__(self, num_envs, dataset_dir) -> None:
        self.dataset_dir = dataset_dir
        YoloCollectData.init(num_envs)  # no use, only [0] is used
        self.episode_begin = 0
        self.episode_end = 90
        self.use_dataset_action = True
        self.use_robot_base = False
    
    def process_item(self, episode_id):
        env_id = 0
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as data_file:
            is_sim = data_file.attrs['sim']
            is_compress = data_file.attrs['compress']
            original_action_shape = data_file['/action'][self.episode_begin:self.episode_end].shape
            max_action_len = original_action_shape[0]

            start_ts = max_action_len
            if self.use_dataset_action:
                actions = data_file['/action'][self.episode_begin:self.episode_end]  # (90, 14)
            else:
                actions = data_file['/observations/qpos'][self.episode_begin:self.episode_end][1:]
                actions = np.append(actions, actions[-1][np.newaxis, :], axis=0)
            
            qpos = data_file['/observations/qpos'][self.episode_begin:self.episode_end]  # (90, 14)
            if self.use_robot_base:
                qpos = np.concatenate((qpos, data_file['/base_action'][self.episode_begin:self.episode_end][start_ts]), axis=0)
            
            YoloCollectData.envs_process[env_id].reset_new_episode()
            
            # image_data = []
            # for i in range(self.episode_begin, self.episode_end):
            #     cams = []
            #     for cam_name in YoloCollectData.cameras_names:
            #         if is_compress:
            #             raw_image = data_file[f'/observations/images/{cam_name}'][i]
            #             image = cv2.imdecode(np.frombuffer(raw_image, np.uint8), 1)
            #         else:
            #             image = data_file[f'/observations/images/{cam_name}'][i]    
            #         image = torch.tensor(image).float().to('cuda')
            #         cams.append(image)
            #     image_data.append(YoloCollectData.envs_process[env_id].process(cams[0], cams[1], cams[2]))  # (1, 12)
            
            image_data = []
            cams = {
                "cam_high": [],
                "cam_left_wrist": [],
                "cam_right_wrist": []
            }
            for i in range(self.episode_begin, self.episode_end):
                for cam_name in YoloCollectData.cameras_names:
                    if is_compress:
                        raw_image = data_file[f'/observations/images/{cam_name}'][i]
                        image = cv2.imdecode(np.frombuffer(raw_image, np.uint8), 1)
                    else:
                        image = data_file[f'/observations/images/{cam_name}'][i]    
                    cams[cam_name].append(image)  # np.array, shape: (480, 640, 3)
            image_data = YoloCollectData.envs_process[env_id].parallel_process_traj(
                cams["cam_high"], cams["cam_left_wrist"], cams["cam_right_wrist"])  # (90, 12)
            
            # image_data = torch.cat(image_data, dim=0)  # (90, 12)
            YoloCollectData.save_data(image_data, qpos, actions, self.dataset_dir)
        print(f"AsyncYoloProcessDataFromHDF5 has processed {episode_id}th episode to integration.pkl")
    
    def run(self, episode_id=0):
        # make sure CollectEpsBuf.use_yolo_sync_process == False
        while True:
            file_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
            if os.path.exists(file_path):
                self.process_item(episode_id)
                if not ((2 <= episode_id < 19)):
                    os.remove(file_path)
                episode_id += 1
            else:
                time.sleep(0.1)  


class YoloCollectData:
    # num_episodes = 30000
    detection_model = YOLO("yolov8l-world.pt") # YOLO("yolov8m-world.pt")
    cameras_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    objects_names = ["apple"]  # , "table"
    episode_length = 90
    store_device = "cpu"
    data = {
        "image_data": torch.zeros((0, episode_length, 12 * len(objects_names)), device=store_device),  # Tensor, shape: (0, episode_len, 24)
        "image_depth_data": None,
        "qpos_data": torch.zeros((0, episode_length, 14), device=store_device),  # Tensor, shape: (0, episode_len, 14)
        "next_action_data": None,
        "next_action_is_pad": None,
        "action_data": torch.zeros((0, episode_length, 14), device=store_device),  # Tensor, shape: (0, episode_len, 14)
        "action_is_pad": None,
        "reward": torch.zeros((0, episode_length), device=store_device),
    }
    # TODO: save only one model
    @staticmethod
    def init(num_envs) -> None:
        YoloProcessDataByTimeStep.objects_names = YoloCollectData.objects_names  # modify the class to be detected
        YoloCollectData.object_to_idx = {obj: idx for idx, obj in enumerate(YoloCollectData.objects_names)}
        YoloCollectData.detection_model.set_classes(YoloCollectData.objects_names)
        print(f"YoloProcessDataByTimeStep.objects_names: {YoloProcessDataByTimeStep.objects_names}")
        YoloCollectData.envs_process = [YoloProcessDataByTimeStep(YoloCollectData.detection_model) for _ in range(num_envs)]
        YoloCollectData.num_envs = num_envs
    
    @staticmethod
    def save_data(image_data, qpos, action, reward, dataset_dir):
        if isinstance(image_data, list) or isinstance(qpos, list):
            image_data = torch.from_numpy(np.array(image_data)).float().unsqueeze(0).to(YoloCollectData.store_device)
            qpos = torch.from_numpy(np.array(qpos)).float().unsqueeze(0).to(YoloCollectData.store_device)
            action = torch.from_numpy(np.array(action)).float().unsqueeze(0).to(YoloCollectData.store_device)
            reward = torch.from_numpy(np.array(reward)).float().unsqueeze(0).to(YoloCollectData.store_device)
        else:
            image_data = torch.tensor(image_data).float().unsqueeze(0).to(YoloCollectData.store_device)  # Add batch dimension
            qpos = torch.tensor(qpos).float().unsqueeze(0).to(YoloCollectData.store_device)
            action = torch.tensor(action).float().unsqueeze(0).to(YoloCollectData.store_device)
            reward = torch.tensor(reward).float().unsqueeze(0).to(YoloCollectData.store_device)
        print(f"image_data, qpos, action, reward:", image_data.shape, qpos.shape, action.shape, reward.shape)
        # torch.Size([1, 90, 12]) torch.Size([1, 90, 14]) torch.Size([1, 90, 14])
        
        YoloCollectData.data["image_data"] = torch.cat([YoloCollectData.data["image_data"], image_data], dim=0)
        YoloCollectData.data["qpos_data"] = torch.cat([YoloCollectData.data["qpos_data"], qpos], dim=0)
        YoloCollectData.data["action_data"] = torch.cat([YoloCollectData.data["action_data"], action], dim=0)
        YoloCollectData.data["reward"] = torch.cat([YoloCollectData.data["reward"], reward], dim=0)
        
        # save the data to (dataset_dir, "integration.pkl")
        num_episodes = YoloCollectData.data["image_data"].shape[0]
        if num_episodes % 10 == 0:  # I checked it, it is correct because CollectEpsBuf.trajectory_num will be added by 1 first
            torch.save(YoloCollectData.data, os.path.join(dataset_dir, f"integration.pkl"))
            print(f"Save {num_episodes}th data to {os.path.join(dataset_dir, f'integration.pkl')}")
    
    
def plot_xyxyn_boxes_to_image(image, bboxes, cam_id=0, path="./"):
    '''
    image: np.array (H, W, 3), RGB, 0~255 
        or torch.tensor (3, H, W), RGB, 0~1
    bboxes: torch.Tensor, bounding box, each bounding box is [[x1, y1, x2, y2]] (x, y ∈ [0, 1]), shape: (num_boxes, 4)
    '''
    assert len(image.shape) == 3

    if isinstance(image, torch.Tensor) and image.shape[0] == 3:  # Check if the image is (3, W, H) and convert to (W, H, 3)
        image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
        image = (image * 255).astype(np.uint8)
    
    if len(bboxes.shape) == 1 and bboxes.shape[0] == 4:
        bboxes = bboxes.unsqueeze(0)
    
    # image: (H, W, 3)
    H, W, _ = image.shape
    image_Image = Image.fromarray(np.uint8(image), 'RGB')
    draw = ImageDraw.Draw(image_Image, 'RGB')
    
    for bbox in bboxes:
        bbox = bbox * torch.tensor([W, H, W, H], dtype=torch.float32, device=bboxes.device)
        draw.rectangle(bbox.tolist(), outline="red", width=5)
    
    # image_Image.show()
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    image_Image.save(os.path.join(path, f'{cam_id}_first_image.jpg'), 'JPEG')
    # print(f"Save the image to {cam_id}_first_image.jpg")