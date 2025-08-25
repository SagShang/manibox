#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""实时机器人推理模块 - Real-time Robot Inference Module

本模块实现了ManiBox框架的实时机器人控制推理系统。
通过ROS（Robot Operating System）与真实机器人硬件通信，
使用训练好的策略模型进行实时动作预测和执行。

This module implements the real-time robot control inference system for ManiBox framework.
Communicates with real robot hardware through ROS (Robot Operating System),
using trained policy models for real-time action prediction and execution.

核心功能 / Core Functions:
- 多策略支持：ACT、RNN、CNN-RNN等 / Multi-policy support: ACT, RNN, CNN-RNN, etc.
- 实时视觉处理：支持RGB+深度图像 / Real-time vision processing: supports RGB + depth images
- 多线程推理：异步推理提高响应速度 / Multi-threaded inference: asynchronous inference for better response
- ROS通信：与机器人硬件实时数据交换 / ROS communication: real-time data exchange with robot hardware
- 动作插值：平滑的动作执行轨迹 / Action interpolation: smooth action execution trajectories
- YOLO物体检测：基于物体检测的视觉预处理 / YOLO object detection: vision preprocessing based on object detection

支持的机器人配置 / Supported Robot Configurations:
- 双臂操作：左臂+右臂协同控制 / Dual-arm manipulation: left + right arm coordination  
- 移动底座：可选的移动机器人支持 / Mobile base: optional mobile robot support
- 多相机视觉：cam_high、cam_left_wrist、cam_right_wrist / Multi-camera vision: cam_high, cam_left_wrist, cam_right_wrist

架构特点 / Architecture Features:
- 模块化设计：策略、视觉、控制分离 / Modular design: policy, vision, control separation
- 容错机制：网络延迟和数据丢失处理 / Fault tolerance: network delay and data loss handling
- 可配置参数：通过命令行灵活配置 / Configurable parameters: flexible configuration via command line
"""

# 深度学习和数值计算库 / Deep learning and numerical computation libraries
import torch               # PyTorch深度学习框架 / PyTorch deep learning framework
import numpy as np         # 数值计算库 / Numerical computation library
import os                  # 操作系统接口 / Operating system interface
import pickle              # Python对象序列化 / Python object serialization
import json                # JSON数据处理 / JSON data processing
import argparse            # 命令行参数解析 / Command line argument parsing
from einops import rearrange  # 张量维度重排 / Tensor dimension rearrangement

# 注释掉的工具函数导入 / Commented utility function imports
# from utils import compute_dict_mean, set_seed, detach_dict # helper functions
# from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy

# Python标准库 / Python standard library
import collections        # 集合数据类型 / Collection data types
from collections import deque  # 双端队列 / Double-ended queue

# ROS（机器人操作系统）相关导入 / ROS (Robot Operating System) related imports
import rospy              # ROS Python客户端库 / ROS Python client library
from std_msgs.msg import Header       # ROS标准消息头 / ROS standard message header
from geometry_msgs.msg import Twist   # ROS几何消息-速度 / ROS geometry message - twist
from sensor_msgs.msg import JointState  # ROS传感器消息-关节状态 / ROS sensor message - joint state
from sensor_msgs.msg import Image as Image_msg  # ROS传感器消息-图像 / ROS sensor message - image
from nav_msgs.msg import Odometry     # ROS导航消息-里程计 / ROS navigation message - odometry
from cv_bridge import CvBridge        # ROS-OpenCV图像桥接 / ROS-OpenCV image bridge

# 系统和并发库 / System and concurrency libraries
import time               # 时间处理 / Time handling
import threading          # 多线程支持 / Multi-threading support
import math               # 数学函数库 / Mathematical functions library

# 图像处理库 / Image processing libraries
from PIL import Image, ImageDraw, ImageFont  # Python图像库 / Python Image Library

# ManiBox项目内部模块 / ManiBox internal modules
from ManiBox.yolo_process_data import YoloProcessDataByTimeStep, plot_xyxyn_boxes_to_image  # YOLO数据处理 / YOLO data processing
from ManiBox.train import make_policy  # 策略模型工厂函数 / Policy model factory function

# 系统路径配置 / System path configuration
import sys
sys.path.append("./")  # 添加当前目录到Python路径 / Add current directory to Python path

# 任务配置字典 / Task configuration dictionary
task_config = {'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']}  # 相机名称列表 / Camera name list

# 全局推理线程管理变量 / Global inference thread management variables
inference_thread = None              # 推理线程对象 / Inference thread object
inference_lock = threading.Lock()    # 线程锁，保护共享数据 / Thread lock to protect shared data
inference_actions = None             # 推理结果动作 / Inference result actions
inference_timestep = None            # 推理时间步 / Inference timestep

def set_seed(seed):
    """设置随机种子 / Set random seed
    
    为PyTorch和NumPy设置随机种子，确保推理过程的可重现性。
    在实时推理中，固定种子有助于调试和结果对比。
    
    Set random seed for PyTorch and NumPy to ensure reproducibility of inference process.
    In real-time inference, fixed seed helps with debugging and result comparison.
    
    Args:
        seed (int): 随机种子值 / Random seed value
    """
    torch.manual_seed(seed)      # 设置PyTorch随机种子 / Set PyTorch random seed
    np.random.seed(seed)         # 设置NumPy随机种子 / Set NumPy random seed

def interpolate_action(args, prev_action, cur_action):
    """动作插值函数 / Action interpolation function
    
    对连续的机器人动作进行线性插值，确保动作执行的平滑性。
    通过计算每个关节的最大变化量，生成平滑的中间动作序列，
    避免机器人动作突变造成的震动和不稳定。
    
    Perform linear interpolation between consecutive robot actions to ensure smooth execution.
    By calculating maximum change for each joint, generate smooth intermediate action sequences,
    avoiding vibration and instability caused by sudden action changes.
    
    Args:
        args: 包含arm_steps_length的配置参数 / Configuration parameters containing arm_steps_length
        prev_action (np.array): 前一个动作 / Previous action
        cur_action (np.array): 当前动作 / Current action
        
    Returns:
        np.array: 插值后的动作序列 / Interpolated action sequence
    """
    # 为左臂和右臂创建步长数组（双臂系统）/ Create step arrays for left and right arms (dual-arm system)
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    
    # 计算动作变化的绝对值 / Calculate absolute difference of action changes
    diff = np.abs(cur_action - prev_action)
    
    # 计算每个维度需要的插值步数 / Calculate interpolation steps needed for each dimension
    step = np.ceil(diff / steps).astype(int)
    
    # 使用最大步数确保所有关节同步运动 / Use maximum steps to ensure synchronized motion of all joints
    step = np.max(step)
    
    # 如果变化很小，直接返回目标动作 / If change is small, return target action directly
    if step <= 1:
        return cur_action[np.newaxis, :]  # 添加批次维度 / Add batch dimension
    
    # 生成线性插值序列 / Generate linear interpolation sequence
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    
    # 返回插值序列（排除起始点）/ Return interpolation sequence (excluding starting point)
    return new_actions[1:]


# def actions_interpolation(args, pre_action, actions, stats):
#     steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
#     if args.use_dataset_action:
#         pre_process = lambda next_action: (next_action - stats["action_mean"]) / stats["action_std"]
#         post_process = lambda a: a * stats['action_std'] + stats['action_mean']
#     else:
#         pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
#         post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']

#     # Original action sequence
#     action_seq = post_process(actions[0])
#     action_seq = np.concatenate((pre_action[np.newaxis, :], action_seq), axis=0)

#     # Interpolated action sequence
#     interp_action = []
#     # Interpolation
#     for i in range(1, action_seq.shape[0]):
#         # Difference between two adjacent actions
#         diff = np.abs(action_seq[i] - action_seq[i - 1])
#         # Number of steps to interpolate
#         step = np.ceil(diff / steps).astype(int)
#         # Interpolation by the maximum number of steps
#         step = np.max(step)
#         if step <= 1:
#             # No need to interpolate if the difference is smaller than the step size
#             interp_action.append(action_seq[i:i+1])
#         else:
#             new_actions = np.linspace(action_seq[i - 1], action_seq[i], step + 1)
#             interp_action.append(new_actions[1:])
    
#     # while len(result) < args.chunk_size+1:
#     #     result.append(result[-1])

#     result = np.concatenate(interp_action, axis=0)
#     result = result[:args.chunk_size]

#     result = pre_process(result)
#     result = result[np.newaxis, :]
#     return result


def get_model_config(args):
    """获取模型配置 / Get model configuration
    
    根据命令行参数构建模型推理所需的配置字典。
    加载训练时保存的配置文件，确保推理时使用与训练时完全相同的模型参数。
    这是保证模型正确加载和推理的关键步骤。
    
    Build model configuration dictionary required for inference based on command line arguments.
    Load configuration file saved during training to ensure identical model parameters are used
    for inference as during training. This is critical for proper model loading and inference.
    
    Args:
        args: 命令行参数对象 / Command line arguments object
        
    Returns:
        dict: 模型配置字典 / Model configuration dictionary
    """
    # 设置随机种子，确保在相同初始条件下每次运行的随机数序列相同 / Set random seed to ensure identical random sequences under same initial conditions
    set_seed(args.seed)

    # 配置加载逻辑 / Configuration loading logic
    if args.load_config:
        # 从训练保存的模型中加载策略配置 / Load policy config from trained model
        config_file = os.path.join(args.ckpt_dir, 'config.json')  # 配置文件路径 / Config file path
        with open(config_file, 'r') as f:
            config_json = f.read()  # 读取JSON配置 / Read JSON config
        policy_config = json.loads(config_json)['policy_config']  # 解析策略配置 / Parse policy config
        print(f"The training config {args.ckpt_dir} has been synced for inference!")  # 配置同步提示 / Config sync notification
    else:
        raise NotImplementedError  # 暂不支持其他配置加载方式 / Other config loading methods not implemented
    
    # 构建完整的推理配置字典 / Build complete inference configuration dictionary
    config = {
        'ckpt_dir': args.ckpt_dir,                    # 检查点目录 / Checkpoint directory
        'ckpt_name': args.ckpt_name,                  # 检查点文件名 / Checkpoint filename
        'ckpt_stats_name': args.ckpt_stats_name,      # 统计数据文件名 / Statistics filename
        'episode_len': args.max_publish_step,         # 推理步数限制 / Inference step limit
        'state_dim': args.state_dim,                  # 状态维度 / State dimension
        'policy_class': args.policy_class,            # 策略类名 / Policy class name
        'policy_config': policy_config,               # 策略详细配置 / Detailed policy config
        'temporal_agg': args.temporal_agg,            # 时序聚合配置 / Temporal aggregation config
        'camera_names': task_config['camera_names'],  # 相机名称列表 / Camera names list
    }
    return config  # 返回配置字典 / Return configuration dictionary

last_right_image = None  # 全局变量：保存上一帧右手相机图像 / Global variable: store last right camera image

def get_image(observation, camera_names):
    """获取和预处理相机图像 / Get and preprocess camera images
    
    从观测数据中提取多相机图像，进行格式转换和归一化处理。
    将图像从HWC格式转换为CHW格式，并转换为PyTorch张量送入GPU。
    
    Extract multi-camera images from observation data, perform format conversion and normalization.
    Convert images from HWC format to CHW format, and convert to PyTorch tensor on GPU.
    
    Args:
        observation (dict): 包含图像数据的观测字典 / Observation dictionary containing image data
        camera_names (list): 相机名称列表 / List of camera names
        
    Returns:
        torch.Tensor: 预处理后的图像张量，形状[1, num_cams, 3, H, W] / Preprocessed image tensor with shape [1, num_cams, 3, H, W]
    """
    global last_right_image  # 声明全局变量 / Declare global variable
    
    curr_images = []  # 当前帧图像列表 / Current frame image list
    
    # 遍历所有相机，提取图像数据 / Iterate through all cameras to extract image data
    for cam_name in camera_names:
        # 将图像从HWC格式重排为CHW格式 / Rearrange image from HWC to CHW format
        curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
        
        # 特殊处理右手相机图像（用于调试）/ Special handling for right wrist camera (for debugging)
        if cam_name == 'cam_right_wrist':
            # 保存当前右手相机图像用于调试 / Save current right camera image for debugging
            last_right_image = curr_image
            
        curr_images.append(curr_image)  # 添加到图像列表 / Add to image list
    
    # 将多个相机图像堆叠为四维数组 / Stack multiple camera images into 4D array
    curr_image = np.stack(curr_images, axis=0)  # 形状: [num_cams, 3, H, W] / Shape: [num_cams, 3, H, W]
    
    # 转换为PyTorch张量，归一化到[0,1]范围，移至GPU并添加批次维度 / Convert to PyTorch tensor, normalize to [0,1], move to GPU and add batch dimension
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)  # 形状: [1, num_cams, 3, H, W] / Shape: [1, num_cams, 3, H, W]
    
    return curr_image  # 返回预处理后的图像张量 / Return preprocessed image tensor


def get_depth_image(observation, camera_names):
    """获取和预处理深度图像 / Get and preprocess depth images
    
    从观测数据中提取多相机深度图像，进行归一化处理。
    深度图像通常用于提供3D空间信息，增强机器人的空间感知能力。
    
    Extract multi-camera depth images from observation data and perform normalization.
    Depth images are typically used to provide 3D spatial information, enhancing robot spatial perception.
    
    Args:
        observation (dict): 包含深度图像数据的观测字典 / Observation dictionary containing depth image data
        camera_names (list): 相机名称列表 / List of camera names
        
    Returns:
        torch.Tensor: 预处理后的深度图像张量 / Preprocessed depth image tensor
    """
    curr_images = []  # 当前帧深度图像列表 / Current frame depth image list
    
    # 遍历所有相机，提取深度图像数据 / Iterate through all cameras to extract depth image data
    for cam_name in camera_names:
        curr_images.append(observation['images_depth'][cam_name])  # 添加深度图像 / Add depth image
    
    # 堆叠多相机深度图像 / Stack multi-camera depth images
    curr_image = np.stack(curr_images, axis=0)
    
    # 转换为PyTorch张量，归一化并移至GPU / Convert to PyTorch tensor, normalize and move to GPU
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    
    return curr_image  # 返回预处理后的深度图像张量 / Return preprocessed depth image tensor


def inference_thread_fn(args, config, ros_operator, policy, next_actions, stats, t, pre_action, yolo_process_data=None):
    global inference_lock
    global inference_actions
    global inference_timestep
    
    if config['policy_class'] in ["RNN", "FCNet", "DiffusionState", "CEPPolicy"]:
        is_qpos_normalized = False
        yolo_preprocess = True
        yolo_process_data: YoloProcessDataByTimeStep
    else:
        is_qpos_normalized = True
        yolo_preprocess = False
    
    print_flag = True
    pre_pos_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    pre_action_process = lambda next_action: (next_action - stats["action_mean"]) / stats["action_std"]
    rate = rospy.Rate(args.publish_rate)
    while True and not rospy.is_shutdown():
        # import pdb; pdb.set_trace()
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result
        obs = collections.OrderedDict()
        image_dict = dict()

        image_dict[config['camera_names'][0]] = img_front
        image_dict[config['camera_names'][1]] = img_left
        image_dict[config['camera_names'][2]] = img_right


        obs['images'] = image_dict

        if args.use_depth_image:
            image_depth_dict = dict()
            image_depth_dict[config['camera_names'][0]] = img_front_depth
            image_depth_dict[config['camera_names'][1]] = img_left_depth
            image_depth_dict[config['camera_names'][2]] = img_right_depth
            obs['images_depth'] = image_depth_dict

        obs['qpos'] = np.concatenate(
            (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
        obs['qvel'] = np.concatenate(
            (np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
        obs['effort'] = np.concatenate(
            (np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
        if args.use_robot_base:
            obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
        else:
            obs['base_vel'] = [0.0, 0.0]
        # 取obs的位姿qpos
        if args.max_pos_lookahead != 0:
            padded_next_action = np.zeros((args.max_pos_lookahead, obs['qpos'].shape[0]), dtype=np.float32)
            next_action_is_pad = np.zeros(args.max_pos_lookahead)
            if next_actions is not None:
                padded_next_action[0:next_actions.shape[0]] = next_actions
                next_action_is_pad[next_actions.shape[0]:] = 1
        else:
            padded_next_action = None
            next_action_is_pad = None

        # qpos_numpy = np.array(obs['qpos'])

        # 归一化处理qpos 并转到cuda
        # TODO: you should consider whether to preprocess the qpos data
        if is_qpos_normalized:
            qpos = pre_pos_process(obs['qpos'])
        else:
            qpos = obs['qpos']

        qpos[7:] = 0
        # qpos = np.zeros(0)
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        if args.max_pos_lookahead != 0:
            next_actions = padded_next_action  # pre_action_process(padded_next_action)
            next_actions = torch.from_numpy(next_actions).float().cuda().unsqueeze(0)
            next_action_is_pad = torch.from_numpy(next_action_is_pad).float().cuda().unsqueeze(0)
        else:
            next_actions = None
            next_action_is_pad = None
        # 当前图像curr_image获取图像
        # print(img_front.shape)
        curr_image = get_image(obs, config['camera_names'])
        curr_depth_image = None
        if args.use_depth_image:
            curr_depth_image = get_depth_image(obs, config['camera_names'])
        
        if yolo_preprocess:
            # print(f"curr_image shape {curr_image.shape}")
            # curr_image shape torch.Size([1, 3, 3, 480, 640])
            # cams = [curr_image[0][0], curr_image[0][1], curr_image[0][2]]
            cams = [torch.tensor(img_front), torch.tensor(img_left), torch.tensor(img_right)]
            # img_*: RGB, shape: (480, 640, 3), 0~255
            for i, cam in enumerate(cams):
                cams[i] = cams[i].permute(2, 0, 1) / 255.0
            # img_*: RGB, shape: (3, 480, 640), 0~1
            image_data = yolo_process_data.process(cams[0].cuda(), cams[1].cuda(), cams[2].cuda())
            
            bboxes = image_data.reshape(3, len(YoloProcessDataByTimeStep.objects_names), 4)  # (3 cameras, apple and table, xyxyn)
            
            # show the high, left, right images with bounding box:
            if t <= 40:
                for i in range(len(cams)):
                    plot_xyxyn_boxes_to_image(cams[i], bboxes[i], config['camera_names'][i] + "_" + str(t))
            #     W, H = image.shape[1:3]
            #     bbox = bboxes[i]
            #     bbox = bbox * torch.Tensor([W, H, W, H])
            #     draw = ImageDraw.Draw(image)
            #     draw.rectangle(bbox.tolist(), outline="red", width=5)
            #     image.show()
            
            print(f"bbox {image_data}")
            if t == 1 and image_data[0][0].sum() == 0.0:
                print("No object detected!!!!!!!!!!!!!!!!")
            image_data = image_data.cuda()
            qpos = qpos.cuda()
        
        time1 = time.time() 
        # TODO: policy inference, maybe you need to modify the code here
        if yolo_preprocess:
            all_actions = policy(image_data, curr_depth_image, qpos, next_actions, next_action_is_pad, None, None)
        else:
            all_actions = policy(curr_image, curr_depth_image, qpos, next_actions, next_action_is_pad)
        print("model_time: ", time.time() - time1)
        inference_lock.acquire()
        # print('--------------', all_actions.shape)  # -------------- torch.Size([1, 30, 14])
        inference_actions = all_actions.cpu().detach().numpy()
        print("left finger", inference_actions[0, 6])
        # if pre_action is None:
        #     pre_action = obs['qpos']
        # print("obs['qpos']:", obs['qpos'][7:])
        # if args.use_actions_interpolation:
        #     inference_actions = actions_interpolation(args, pre_action, inference_actions, stats)
        inference_timestep = t
        inference_lock.release()
        break


def detect_object(prompt, ros_operator, yolo_process_data: YoloProcessDataByTimeStep = None):
    # detect whether there is specified object in the camera
    if yolo_process_data is None:
        return True
    result = ros_operator.get_frame()
    (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result
    cams = [torch.tensor(img_front), torch.tensor(img_left), torch.tensor(img_right)]
    # img_*: RGB, shape: (480, 640, 3), 0~255
    for i, cam in enumerate(cams):
        cams[i] = cams[i].permute(2, 0, 1) / 255.0
    # img_*: RGB, shape: (3, 480, 640), 0~1
    image_data = yolo_process_data.process(cams[0].cuda(), cams[1].cuda(), cams[2].cuda()) 
    if image_data[0][0].sum() == 0.0:
        print(f"No {prompt} detected in cam_high!!!!!!!!!!!!!!!!")
        return False
    else:
        print(f"{prompt} detected in cam_high")
        return True


def model_inference(args, config, ros_operator, save_episode=True):
    global inference_lock
    global inference_actions
    global inference_timestep
    global inference_thread
    # Get the prediction from the inference thread
    def get_inference_result(inference_thread, inference_lock, inference_actions,
                            inference_timestep):
        if inference_thread is not None:
            inference_lock.acquire()
            action_chunk = inference_actions
            inference_actions = None
            action_t = inference_timestep
            inference_timestep = None
            inference_lock.release()
            return action_chunk.squeeze(0), action_t
        return None, None
    # Update the action to the action buffer
    def update_action_buffer(action_buffer, action_chunk, action_t):
        start_idx = action_t % chunk_size
        end_idx = (start_idx + chunk_size) % chunk_size
        action_buffer[start_idx:] = action_chunk[:chunk_size - start_idx]
        action_buffer[:end_idx] = action_chunk[chunk_size - start_idx:]
        return action_buffer
    

    # 1 创建模型数据  继承nn.Module
    # policy = make_policy(config['policy_class'], config['policy_config'])
    policy = make_policy(config['policy_class'], config['policy_config'], config['ckpt_dir']).to(device=args.device)
    # print("model structure\n", policy.model)
    
    # 2 加载模型权重 (in make_policy)
    # ckpt_path = os.path.join(config['ckpt_dir'], config['ckpt_name'])
    # state_dict = torch.load(ckpt_path)
    # 状态字典过滤（已注释）/ State dict filtering (commented)
    # new_state_dict = {}
    # for key, value in state_dict.items():
    #     if key in ["model.is_pad_head.weight", "model.is_pad_head.bias"]:
    #         continue  # 跳过填充头权重 / Skip padding head weights
    #     if args.max_pos_lookahead == 0 and key in ["model.input_proj_next_action.weight", "model.input_proj_next_action.bias"]:
    #         continue  # 跳过下一动作投影权重 / Skip next action projection weights
    #     new_state_dict[key] = value
    # loading_status = policy.deserialize(new_state_dict)  # 反序列化模型状态 / Deserialize model state
    # if not loading_status:
    #     print("ckpt path not exist")  # 检查点路径不存在 / Checkpoint path does not exist
    #     return False

    # 3. 模型设备和模式设置 / Model device and mode setup
    policy = policy.cuda()  # 将模型移至GPU / Move model to GPU
    policy.eval()  # 设置模型为评估模式，禁用dropout和batch normalization训练行为 / Set model to evaluation mode, disable dropout and batch normalization training behavior

    # 4. 加载数据统计信息 / Load data statistics
    stats_path = os.path.join(config['ckpt_dir'], config['ckpt_stats_name'])
    # 加载统计数据：action_mean, action_std, qpos_mean, qpos_std（14维）/ Load statistics: action_mean, action_std, qpos_mean, qpos_std (14-dim)
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)  # 反序列化统计数据 / Deserialize statistics data

    # 数据预处理和后处理函数定义 / Data preprocessing and postprocessing function definition
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']  # 关节位置标准化 / Joint position normalization
    if args.use_dataset_action:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']  # 动作反标准化 / Action denormalization
    else:
        post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']  # 关节位置反标准化 / Joint position denormalization

    # 推理参数设置 / Inference parameter setup
    max_publish_step = config['episode_len']         # 最大发布步数 / Maximum publish steps
    chunk_size = config['policy_config']['chunk_size']  # 动作块大小 / Action chunk size

    # 机械臂初始位置设定 / Robot arm initial position setup
    # 原始测试位置（已注释）/ Original test positions (commented)
    # left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    # right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
    # left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
    # right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
    
    # 左臂位置：6个关节角度 + 夹爪开合度 / Left arm position: 6 joint angles + gripper openness
    left1 = [0] * 6 + [-0.1350]  # 夹爪开合状态 / Gripper open state
    # left1 = [-0.0050,  0.0520, -0.0080, -0.0000,  0.0000,  0.0000, -0.1350]  # 备选左臂位置 / Alternative left arm position
    right1 = [0] * 6 + [-0.1350]  # 右臂夹爪开合状态 / Right arm gripper open state
    left0 = left1[:6] + [3.557830810546875]   # 左臂夹爪关闭状态 / Left arm gripper closed state
    right0 = right1[:6] + [3.557830810546875]  # 右臂夹爪关闭状态 / Right arm gripper closed state
    # 备选初始位置（已注释）/ Alternative initial positions (commented)
    # left0 = [0] * 6 + [3.557830810546875]
    # left1 = [0] * 6 + [-0.3393220901489258]
    # right0 = [0] * 6 + [3.557830810546875]
    # right1 = [0] * 6 + [-0.3397035598754883]
    
    
    # 发布初始恢复位置 / Publish initial recovery position
    ros_operator.puppet_arm_publish_continuous(left0, right0)  # 恢复到初始位置 / Recover to initial position
    
    # 策略类型判断和预处理设置 / Policy type determination and preprocessing setup
    if config['policy_class'] in ["RNN", "FCNet", "DiffusionState", "CEPPolicy"]:
        is_qpos_normalized = False  # 不需要关节位置标准化 / No joint position normalization needed
        yolo_preprocess = True      # 需要YOLO预处理 / YOLO preprocessing required
    else:
        is_qpos_normalized = True   # 需要关节位置标准化 / Joint position normalization required
        yolo_preprocess = False     # 不需要YOLO预处理 / No YOLO preprocessing needed
    
    # YOLO数据处理器初始化 / YOLO data processor initialization
    if yolo_preprocess:
        yolo_process_data = YoloProcessDataByTimeStep()  # 创建YOLO处理器 / Create YOLO processor
        yolo_process_data.reset_new_episode()            # 重置新回合 / Reset new episode
        policy.reset_recur(1, "cuda:0")                  # 重置策略递归状态 / Reset policy recurrent state
    else:
        yolo_process_data = None  # 不使用YOLO处理器 / No YOLO processor
    # input("Press any key to continue")  # 等待用户输入（已注释）/ Wait for user input (commented)
    # 目标物体检测和用户交互 / Target object detection and user interaction
    if yolo_preprocess:
        while True:  # 循环直到成功检测到目标物体 / Loop until target object is successfully detected
            try:
                prompt = input("Please input the object you wish to grab, the default is 'apple', then press enter to continue: ")  # 获取用户输入的目标物体 / Get user input for target object
            except KeyboardInterrupt:
                print("Interrupted")  # 用户中断 / User interruption
                exit(0)
            if prompt == "":
                prompt = "apple"  # 默认目标物体 / Default target object
            print("your prompt:", prompt)  # 显示用户选择的目标 / Display user's target selection
            yolo_process_data.modify_objects_names(prompt)  # 修改YOLO检测器的目标物体名称 / Modify YOLO detector's target object name
            if detect_object(prompt, ros_operator, yolo_process_data):  # 尝试检测目标物体 / Attempt to detect target object
                break  # 检测成功，退出循环 / Detection successful, exit loop
    else:        
        input("Press any key to continue")  # 等待用户输入继续 / Wait for user input to continue
        
    # 设置机械臂到工作位置 / Set robot arm to working position
    ros_operator.puppet_arm_publish_continuous(left1, right1)  # 发布工作位置（夹爪开启）/ Publish working position (gripper open)
    
    # 初始化前一动作为机器人初始状态 / Initialize previous action as initial robot state
    pre_action = np.zeros(config['state_dim'])  # 创建零动作向量 / Create zero action vector
    pre_action[:14] = np.array(
        left1 +   # 左臂7维（6个关节+夹爪）/ Left arm 7-dim (6 joints + gripper)
        right1    # 右臂7维（6个关节+夹爪）/ Right arm 7-dim (6 joints + gripper)
    )
    action = None  # 初始化动作为空 / Initialize action as None
    
    
    
    # 推理主循环 / Main inference loop
    with torch.inference_mode():  # 推理模式，禁用梯度计算 / Inference mode, disable gradient computation
        while True and not rospy.is_shutdown():  # 持续运行直到ROS关闭 / Continue running until ROS shutdown
            # 当前时间步 / Current time step
            t = 0
            # max_t = 0  # 最大时间步（已注释）/ Maximum time step (commented)
            rate = rospy.Rate(args.publish_rate)  # 设置发布频率 / Set publish rate
            # 时序聚合动作缓冲区初始化（已注释）/ Temporal aggregation action buffer initialization (commented)
            # if config['temporal_agg']:  
            #     all_time_actions = np.zeros([max_publish_step, max_publish_step + chunk_size, config['state_dim']])  # 所有时间步的动作矩阵 / Action matrix for all timesteps
            
            # 启动第一个推理线程并初始化动作缓冲区 / Start the first thread and initialize the action buffer
            inference_thread = threading.Thread(target=inference_thread_fn,  # 创建推理线程 / Create inference thread
                                                args=(args, config, ros_operator,  # 传递参数 / Pass arguments
                                                    policy, None, stats, t, pre_action, yolo_process_data))
            inference_thread.start()  # 启动推理线程 / Start inference thread
            action_buffer = np.zeros([chunk_size, config['state_dim']])  # 初始化动作缓冲区 / Initialize action buffer
            
            # 推理执行循环 / Inference execution loop
            while t < max_publish_step and not rospy.is_shutdown():  # 在最大步数内且ROS未关闭时循环 / Loop within max steps and ROS not shutdown
                start_time = time.time()  # 记录开始时间 / Record start time
                # 查询策略 / Query policy
                # ACT策略的位置前瞻处理（已注释）/ ACT policy position lookahead processing (commented)
                # if config['policy_class'] == "ACT":
                    # if args.max_pos_lookahead != 0:  # 如果使用位置前瞻 / If using position lookahead
                    #     if t == 0:  # 第一个时间步 / First timestep
                    #         pre_action = None
                    #         inference_thread = threading.Thread(target=inference_process,
                    #                                             args=(args, config, ros_operator,
                    #                                                   policy, None, stats, t, pre_action))
                    #         inference_thread.start()
                    #     if t >= max_t:  # 达到最大时间步时 / When reaching max timestep
                    #         if inference_thread is not None:
                    #             inference_thread.join()  # 等待推理线程完成 / Wait for inference thread completion
                    #             inference_lock.acquire()  # 获取推理锁 / Acquire inference lock
                    #             if inference_actions is not None:
                    #                 inference_thread = None
                    #                 all_actions = inference_actions  # 获取推理结果 / Get inference results
                    #                 inference_actions = None
                    #                 max_t = t + args.pos_lookahead_step  # 更新最大时间步 / Update max timestep
                    #                 if config['temporal_agg']:  # 时序聚合处理 / Temporal aggregation processing
                    #                     all_time_actions[[t], t:t + chunk_size] = all_actions
                    #             inference_lock.release()  # 释放推理锁 / Release inference lock
                    #             pre_action = post_process(all_actions[0][args.pos_lookahead_step-1])  # 后处理前一动作 / Post-process previous action
                    #             inference_thread = threading.Thread(target=inference_process,  # 启动新的推理线程 / Start new inference thread
                    #                                                 args=(args, config, ros_operator,
                    #                                                       policy, all_actions[0][:args.pos_lookahead_step], stats, t, pre_action))
                    #             inference_thread.start()
                    #             print("inference:t=", t)  # 打印推理时间步 / Print inference timestep
                    # else:  # 不使用位置前瞻 / Not using position lookahead
                        # if t >= max_t:  # 条件判断（已注释）/ Condition check (commented)

                # 当到达动作块的中间或末尾时 / When coming to the middle or the end of the action chunk
                
                # 单步推理策略处理 / Single-step inference policy processing
                if config['policy_class'] in ["FCNet", "RNN", "DiffusionState", "CEPPolicy"]:
                    inference_thread.join()  # 等待推理线程完成 / Wait for inference thread completion
                    action, action_t = get_inference_result(inference_thread, inference_lock,   # 获取推理结果 / Get inference result
                                                inference_actions, inference_timestep)
                    # 模型输出动作形状: (1, 14) / Model output action shape: (1, 14)
                    # 处理后动作: (14) / Processed action: (14)
                    
                    # print(f"inference one-step action {action}")  # 打印单步推理动作（已注释）/ Print one-step inference action (commented)
                    raw_action = action  # 保存原始动作 / Save raw action
                    
                    # 后续推理线程启动（已注释）/ Subsequent inference thread start (commented)
                    # inference_thread = threading.Thread(target=inference_thread_fn,
                    #                                     args=(args, config, ros_operator,
                    #                                             policy, None, stats, t, pre_action, yolo_process_data))
                    # inference_thread.start()
                
                # ACT策略的块推理处理 / ACT policy chunk inference processing
                if config['policy_class'] == "ACTPolicy" and t % (chunk_size // 2) == 0:  # 每半个chunk执行一次推理 / Execute inference every half chunk
                    # 等待前一个推理线程完成 / Wait for the previous inference thread to finish
                    inference_thread.join()
                    action_chunk, action_t = get_inference_result(inference_thread, inference_lock,   # 获取动作块推理结果 / Get action chunk inference result
                                                inference_actions, inference_timestep)
                    action_buffer = update_action_buffer(action_buffer, action_chunk, action_t)  # 更新动作缓冲区 / Update action buffer

                    # 启动新的推理线程 / Start a new inference thread
                    inference_thread = threading.Thread(target=inference_thread_fn,
                                                        args=(args, config, ros_operator,
                                                                policy, None, stats, t, pre_action, yolo_process_data))
                    inference_thread.start()  # 启动线程 / Start thread
                
                # 推理结果处理和时序聚合（已注释）/ Inference result processing and temporal aggregation (commented)
                # inference_thread.join()  # 等待推理线程 / Wait for inference thread
                # inference_lock.acquire()  # 获取推理锁 / Acquire inference lock
                # if inference_actions is not None:
                #     inference_thread = None
                #     all_actions = inference_actions  # 获取所有动作 / Get all actions
                #     inference_actions = None
                    # max_t = t + args.pos_lookahead_step  # 更新最大时间步 / Update max timestep
                    # if config['temporal_agg']:  # 时序聚合处理 / Temporal aggregation processing
                    #     all_time_actions[[t], t:t + chunk_size] = all_actions
                # inference_lock.release()  # 释放推理锁 / Release inference lock

                    # 时序聚合权重计算（已注释）/ Temporal aggregation weight computation (commented)
                    # if config['temporal_agg']:
                    #     actions_for_curr_step = all_time_actions[:, t]  # 获取当前步的所有动作 / Get all actions for current step
                    #     actions_populated = np.all(actions_for_curr_step != 0, axis=1)  # 检查非零动作 / Check non-zero actions
                    #     actions_for_curr_step = actions_for_curr_step[actions_populated]  # 过滤有效动作 / Filter valid actions
                    #     k = 0.01  # 指数衰减系数 / Exponential decay coefficient
                    #     exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))  # 计算指数权重 / Compute exponential weights
                    #     exp_weights = exp_weights / exp_weights.sum()  # 归一化权重 / Normalize weights
                    #     exp_weights = exp_weights[:, np.newaxis]  # 扩展维度 / Expand dimensions
                    #     raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)  # 加权求和 / Weighted sum
                    # else:  # 不使用时序聚合 / Not using temporal aggregation
                    #     if args.pos_lookahead_step != 0:  # 使用位置前瞻 / Using position lookahead
                    #         raw_action = all_actions[:, t % args.pos_lookahead_step]  # 获取前瞻动作 / Get lookahead action
                    #     else:
                    #         raw_action = all_actions[:, t % chunk_size]  # 获取当前块动作 / Get current chunk action
                # else:  # 未实现的策略类型 / Unimplemented policy type
                #     raise NotImplementedError

                # 从动作缓冲区获取当前动作 / Get current action from action buffer
                raw_action = action_buffer[t % chunk_size]  # 使用循环索引获取动作 / Use circular index to get action
                
                # 动作后处理 / Action post-processing
                if is_qpos_normalized:
                    action = post_process(raw_action)  # 反标准化动作 / Denormalize action
                else:
                    action = raw_action  # 直接使用原始动作 / Use raw action directly
                
                # 直接发布动作（已注释），效果不如下面的插值代码 / Direct action publish (commented), worse than the interpolation code below
                # ros_operator.puppet_arm_publish_continuous(action[:7], action[7:])
                
                # 对原始动作序列进行插值 / Interpolate the original action sequence
                if args.use_actions_interpolation:
                    interp_actions = interpolate_action(args, pre_action, action)  # 生成插值动作序列 / Generate interpolated action sequence
                else:
                    interp_actions = action[np.newaxis, :]  # 不进行插值，直接使用单个动作 / No interpolation, use single action directly
                
                # 逐一执行插值后的动作 / Execute the interpolated actions one by one
                for act in interp_actions:
                    left_action = act[:7]   # 取左臂7维度 shape:(7) / Take left arm 7 dimensions
                    # left_action[0:6] = 0  # 设置关节角度为0（已注释）/ Set joint angles to 0 (commented)
                    # if t <=4 :  # 前几步的特殊处理（已注释）/ Special handling for first few steps (commented)
                    #     left_action[6] = 4.3  # 设置夹爪位置 / Set gripper position
                    # print("----------------------left_action[6]", left_action[6])  # 打印夹爪位置 / Print gripper position
                    # left_action[7] =   # 第8维设置（未完成）/ 8th dimension setting (incomplete)
                    right_action = act[7:14]  # 取右臂7维度 / Take right arm 7 dimensions
                    
                    # 发布机械臂动作指令 / Publish robot arm action commands
                    ros_operator.puppet_arm_publish(left_action, right_action)  # 发布左右臂动作 / Publish left and right arm actions
                
                    # 机器人底盘控制（可选）/ Robot base control (optional)
                    if args.use_robot_base:
                        vel_action = act[14:16]  # 获取底盘速度动作 / Get base velocity action
                        ros_operator.robot_base_publish(vel_action)  # 发布底盘速度 / Publish base velocity
                    rate.sleep()  # 等待按照设定频率 / Wait according to set frequency
                t += 1  # 时间步递增 / Increment time step
                
                # 单步策略的后续推理线程启动 / Subsequent inference thread start for single-step policies
                if config['policy_class'] in ["FCNet", "RNN", "DiffusionState", "CEPPolicy"]:
                    # 备用推理线程管理（已注释）/ Alternative inference thread management (commented)
                    # inference_thread.join()  # 等待线程完成 / Wait for thread completion
                    # action, action_t = get_inference_result(inference_thread, inference_lock, 
                    #                             inference_actions, inference_timestep)  # 获取结果 / Get results
                    # # 模型输出动作形状: (1, 14) / Model output action shape: (1, 14)
                    # # 处理后动作: (14) / Processed action: (14)
                    
                    # print(f"inference one-step action {action}")  # 打印单步动作 / Print single-step action
                    # raw_action = action  # 保存原始动作 / Save raw action
                    
                    # 为下一步启动新的推理线程 / Start new inference thread for next step
                    inference_thread = threading.Thread(target=inference_thread_fn,
                                                        args=(args, config, ros_operator,
                                                                policy, None, stats, t, pre_action, yolo_process_data))
                    inference_thread.start()  # 启动线程 / Start thread
                end_time = time.time()  # 记录结束时间 / Record end time
                
                # 打印执行信息和更新状态 / Print execution info and update state
                print("Published Step", t)  # 打印已发布的步数 / Print published step number
                # print("time:", end_time - start_time)  # 打印执行时间（已注释）/ Print execution time (commented)
                # print("left_action:", left_action)  # 打印左臂动作（已注释）/ Print left arm action (commented)
                # print("right_action:", right_action)  # 打印右臂动作（已注释）/ Print right arm action (commented)
                # rate.sleep()  # 等待频率控制（已注释）/ Wait for rate control (commented)
                pre_action = action  # 更新前一动作 / Update previous action


class RosOperator:
    """
    ROS通信操作类 / ROS Communication Operator Class
    
    负责处理与机器人硬件的ROS通信，包括：/ Handles ROS communication with robot hardware, including:
    - 机械臂控制 / Robot arm control
    - 底盘移动 / Base movement  
    - 相机数据获取 / Camera data acquisition
    - 数据缓冲管理 / Data buffer management
    
    Attributes:
        robot_base_deque: 机器人底盘数据队列 / Robot base data queue
        puppet_arm_*_deque: 机械臂数据队列 / Robot arm data queues
        img_*_deque: 图像数据队列 / Image data queues
        bridge: CV桥接器 / CV bridge
        *_publisher: ROS发布者 / ROS publishers
    """
    def __init__(self, args):
        """
        初始化ROS操作器 / Initialize ROS operator
        
        Args:
            args: 命令行参数对象 / Command line arguments object
        """
        # 数据队列初始化 / Data queue initialization
        self.robot_base_deque = None              # 机器人底盘数据队列 / Robot base data queue
        self.puppet_arm_right_deque = None        # 右臂数据队列 / Right arm data queue
        self.puppet_arm_left_deque = None         # 左臂数据队列 / Left arm data queue
        self.img_front_deque = None               # 前置相机图像队列 / Front camera image queue
        self.img_right_deque = None               # 右相机图像队列 / Right camera image queue
        self.img_left_deque = None                # 左相机图像队列 / Left camera image queue
        self.img_front_depth_deque = None         # 前置深度相机队列 / Front depth camera queue
        self.img_right_depth_deque = None         # 右深度相机队列 / Right depth camera queue
        self.img_left_depth_deque = None          # 左深度相机队列 / Left depth camera queue
        
        # ROS通信组件初始化 / ROS communication components initialization
        self.bridge = None                        # CV桥接器 / CV bridge
        self.puppet_arm_left_publisher = None     # 左臂发布者 / Left arm publisher
        self.puppet_arm_right_publisher = None    # 右臂发布者 / Right arm publisher
        self.robot_base_publisher = None          # 机器人底盘发布者 / Robot base publisher
        
        # 线程和同步控制 / Thread and synchronization control
        self.puppet_arm_publish_thread = None     # 机械臂发布线程 / Robot arm publish thread
        self.puppet_arm_publish_lock = None       # 机械臂发布锁 / Robot arm publish lock
        
        self.args = args                          # 保存参数 / Save arguments
        self.init()                               # 基础初始化 / Basic initialization
        self.init_ros()                           # ROS初始化 / ROS initialization

    def init(self):
        """
        基础组件初始化 / Basic components initialization
        
        初始化CV桥接器和各类数据队列 / Initialize CV bridge and various data queues
        """
        self.bridge = CvBridge()                  # 初始化OpenCV-ROS桥接器 / Initialize OpenCV-ROS bridge
        
        # 初始化图像数据队列 / Initialize image data queues
        self.img_left_deque = deque()             # 左相机图像队列 / Left camera image queue
        self.img_right_deque = deque()            # 右相机图像队列 / Right camera image queue
        self.img_front_deque = deque()            # 前置相机图像队列 / Front camera image queue
        self.img_left_depth_deque = deque()       # 左深度相机队列 / Left depth camera queue
        self.img_right_depth_deque = deque()      # 右深度相机队列 / Right depth camera queue
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            # print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        """获取同步的多传感器帧数据 / Get synchronized multi-sensor frame data
        
        从各个传感器队列中提取时间同步的数据帧，确保所有传感器数据来自同一时刻。
        这是实现多模态传感器融合的关键函数，处理传感器之间的时间对齐问题。
        
        Extract time-synchronized data frames from various sensor queues, ensuring all sensor
        data comes from the same moment. This is a key function for multi-modal sensor fusion,
        handling time alignment issues between sensors.
        
        Returns:
            tuple/bool: 同步的传感器数据元组，如果无法同步则返回False / 
                       Synchronized sensor data tuple, or False if synchronization fails
        """
        # 检查必需传感器队列是否有数据 / Check if required sensor queues have data
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False  # 数据不完整，无法获取帧 / Incomplete data, cannot get frame
        # 计算帧同步时间戳：找到所有传感器最新数据的最早时间 / Calculate frame sync timestamp: find earliest time among latest data from all sensors
        if self.args.use_depth_image:
            # 包含深度图像的同步：需要同步RGB和深度传感器 / Sync with depth images: need to sync RGB and depth sensors
            frame_time = min([
                self.img_left_deque[-1].header.stamp.to_sec(),       # 左相机时间戳 / Left camera timestamp
                self.img_right_deque[-1].header.stamp.to_sec(),      # 右相机时间戳 / Right camera timestamp  
                self.img_front_deque[-1].header.stamp.to_sec(),      # 前相机时间戳 / Front camera timestamp
                self.img_left_depth_deque[-1].header.stamp.to_sec(), # 左深度相机时间戳 / Left depth camera timestamp
                self.img_right_depth_deque[-1].header.stamp.to_sec(),# 右深度相机时间戳 / Right depth camera timestamp
                self.img_front_depth_deque[-1].header.stamp.to_sec() # 前深度相机时间戳 / Front depth camera timestamp
            ])
        else:
            # 仅RGB相机同步：只需同步三个RGB传感器 / RGB-only sync: only need to sync three RGB sensors
            frame_time = min([
                self.img_left_deque[-1].header.stamp.to_sec(),   # 左相机时间戳 / Left camera timestamp
                self.img_right_deque[-1].header.stamp.to_sec(),  # 右相机时间戳 / Right camera timestamp
                self.img_front_deque[-1].header.stamp.to_sec()   # 前相机时间戳 / Front camera timestamp
            ])

        # 验证所有传感器是否都有该时间戳之后的数据 / Verify all sensors have data after this timestamp
        # 如果任何传感器缺少所需时间的数据，返回False / If any sensor lacks data for required time, return False
        
        # 检查RGB相机数据可用性 / Check RGB camera data availability
        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False  # 左相机数据不足 / Insufficient left camera data
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False  # 右相机数据不足 / Insufficient right camera data
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False  # 前相机数据不足 / Insufficient front camera data
            
        # 检查机械臂状态数据可用性 / Check robot arm state data availability
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False  # 左臂状态数据不足 / Insufficient left arm state data
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False  # 右臂状态数据不足 / Insufficient right arm state data
            
        # 检查深度相机数据可用性（如果启用）/ Check depth camera data availability (if enabled)
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False  # 左深度相机数据不足 / Insufficient left depth camera data
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False  # 右深度相机数据不足 / Insufficient right depth camera data
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False  # 前深度相机数据不足 / Insufficient front depth camera data
            
        # 检查机器人底座数据可用性（如果启用）/ Check robot base data availability (if enabled)
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False  # 机器人底座数据不足 / Insufficient robot base data

        # === 数据提取阶段：从队列中提取同步时间的数据 / Data extraction phase: extract synchronized data from queues ===
        
        # 提取左相机图像 / Extract left camera image
        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()  # 移除过时的数据帧 / Remove outdated data frames
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')  # 转换ROS图像消息为OpenCV格式 / Convert ROS image message to OpenCV format

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image_msg, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image_msg, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image_msg, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image_msg, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image_msg, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image_msg, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='aloha_mobile_dummy', required=False)
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default='ACT', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5, required=False)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)", required=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features", required=False)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=False, required=False)

    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=1e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer", required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=30, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    parser.add_argument('--max_pos_lookahead', action='store', type=int, help='max_pos_lookahead',
                        default=0, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=30, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)
                        # default=[0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.1], required=False)

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation',
                        default=True, required=False)
    parser.add_argument('--use_dataset_action', action='store', type=bool, help='use_dataset_action',
                        default=True, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)

    # for Diffusion
    parser.add_argument('--observation_horizon', action='store', type=int, help='observation_horizon', default=1, required=False)
    parser.add_argument('--action_horizon', action='store', type=int, help='action_horizon', default=8, required=False)
    parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=10, required=False)
    parser.add_argument('--ema_power', action='store', type=int, help='ema_power', default=0.75, required=False)
    
    parser.add_argument('--pretrain_timestamp', action='store', type=str, help='pretrain_timestamp, like 2024-03-27_16-52-32', default='', required=False)
    parser.add_argument('--load_config', action='store', type=int, help='load_config', default=1, required=False)
    parser.add_argument('--device', type=str, help='device', default='cuda:0')
    args = parser.parse_args()
    
    # args.arm_steps_length = [x * 2 for x in args.arm_steps_length]
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    config = get_model_config(args)
    model_inference(args, config, ros_operator, save_episode=True)


if __name__ == '__main__':
    main()
