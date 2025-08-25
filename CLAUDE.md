# CLAUDE.md

此文件为Claude Code (claude.ai/code)在使用此仓库代码时提供指导。  
This file provides guidance for Claude Code (claude.ai/code) when working with this repository.

## 安装和配置 / Installation and Configuration

```bash
conda deactivate
conda create -n manibox python=3.9
conda activate manibox

pip install "setuptools<60" numpy==1.22.4
pip install -e .

# 对于isaac lab中的学生推理代码 / For student inference code in isaac lab:
pip install einops
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
```

## 训练命令 / Training Commands

### 学生训练 / Student Training
```bash
# BBOX RNN 训练 / BBOX RNN Training
python ManiBox/train.py --policy_class RNN --batch_size 128 --dataset ../ --num_episodes 38000 --loss_function l1 --rnn_layers 3 --rnn_hidden_dim 512 --actor_hidden_dim 512 --num_epochs 50 --lr 2e-3 --gradient_accumulation_steps 1
```

### 部署/推理 / Deployment/Inference
```bash
python ManiBox/inference_real_world.py --ckpt_dir /PATH/TO/ManiBox/ckpt/2024-xx-xx_xx-xx-xxRNN --policy_class RNN --ckpt_name policy_best.ckpt
```

## 数据集 / Dataset

代码库需要在数据集目录中有一个`integration.pkl`文件，包含：  
The codebase requires an `integration.pkl` file in the dataset directory containing:
- `image_data`: (num_episodes, episode_len, dim) - 相机观测数据 / Camera observation data
- `qpos_data`: (num_episodes, episode_len, dim) - 关节位置数据 / Joint position data 
- `action_data`: (num_episodes, episode_len, dim) - 动作轨迹数据 / Action trajectory data

## 代码架构 / Code Architecture

### 核心组件 / Core Components

**ManiBox/train.py**: 主训练脚本，功能包括：  
Main training script with functionality including:
- 使用自定义数据加载器加载情节数据集 / Load episodic datasets using custom data loaders
- 支持多种策略类型（ACT、RNN、CNN-MLP、Diffusion）/ Support multiple policy types (ACT, RNN, CNN-MLP, Diffusion)
- 处理归一化统计和数据预处理 / Handle normalization statistics and data preprocessing
- 使用Accelerate进行分布式训练 / Distributed training using Accelerate

**ManiBox/inference_real_world.py**: 真实世界部署脚本，功能包括：  
Real-world deployment script with functionality including:
- 与ROS接口进行机器人控制和传感器数据处理 / Interface with ROS for robot control and sensor data handling
- 支持基于YOLO的物体检测进行空间抓取 / Support YOLO-based object detection for spatial grasping
- 处理动作插值和机器人手臂控制 / Handle action interpolation and robot arm control
- 管理相机流和关节状态同步 / Manage camera streams and joint state synchronization

**策略架构 / Policy Architecture** (ManiBox/policy/):
- `ACTPolicy.py`: 动作分块变换器实现 / Action Chunking Transformer implementation
- `RNNPolicy.py`: 基于RNN的序列预测策略 / RNN-based sequential prediction policy
- `backbone.py`: 共享视觉编码器（ResNet变体）/ Shared visual encoders (ResNet variants)
- 策略使用不同的分块策略和时间建模 / Policies use different chunking strategies and temporal modeling

**数据加载 / Data Loading** (ManiBox/dataloader/):
- `EpisodicDataset.py`: 单时间步基础数据集 / Single-timestep base dataset
- `HistoryEpisodicDataset.py`: 多时间步上下文窗口 / Multi-timestep context windows
- `BBoxHistoryEpisodicDataset.py`: 边界框预处理数据 / Bounding box preprocessed data
- 处理带压缩支持的HDF5数据格式 / Handle HDF5 data formats with compression support

### 关键模式 / Key Patterns

**策略工厂 / Policy Factory**: train.py中的`make_policy()`函数根据字符串名称和配置实例化不同的策略类型。  
The `make_policy()` function in train.py instantiates different policy types based on string names and configurations.

**归一化 / Normalization**: 所有数据集计算并应用qpos和动作数据的归一化统计信息以稳定训练。  
All datasets compute and apply normalization statistics for qpos and action data to stabilize training.

**多相机设置 / Multi-Camera Setup**: 标准相机配置使用`['cam_high', 'cam_left_wrist', 'cam_right_wrist']`提供不同视角。  
Standard camera configuration uses `['cam_high', 'cam_left_wrist', 'cam_right_wrist']` for different viewpoints.

**情节窗口 / Episode Windows**: 训练使用可配置的情节窗口（如episode_begin=3, episode_end=90）来避免边界情况。  
Training uses configurable episode windows (e.g. episode_begin=3, episode_end=90) to avoid boundary cases.

### 训练配置 / Training Configuration

模型通过具有策略特定参数的广泛参数解析进行配置。关键配置包括：  
Models are configured through extensive argument parsing with policy-specific parameters. Key configurations include:
- 动作预测视野的分块大小 / Chunk size for action prediction horizon
- RNN层数和隐藏维度 / RNN layer count and hidden dimensions
- 学习率（骨干网络学习率单独配置）/ Learning rates (backbone lr configured separately)
- 损失函数（L1、L2、组合）/ Loss functions (L1, L2, combinations)
- 数据增强选项 / Data augmentation options

代码库自动保存模型检查点、训练曲线和数据集统计信息，以实现可重现的推理。  
The codebase automatically saves model checkpoints, training curves, and dataset statistics for reproducible inference.

## 重要技术细节 / Important Technical Details

### 机器人控制 / Robot Control
- **双臂协调 / Dual-Arm Coordination**: 支持左右臂独立控制，每臂7自由度（6关节+夹爪）  
  Supports independent control of left and right arms, 7 DOF each (6 joints + gripper)
- **动作插值 / Action Interpolation**: 平滑的动作执行以减少机械振动  
  Smooth action execution to reduce mechanical vibrations
- **ROS集成 / ROS Integration**: 完整的机器人操作系统集成用于实时控制  
  Complete Robot Operating System integration for real-time control

### 视觉处理 / Vision Processing
- **YOLO物体检测 / YOLO Object Detection**: 实时目标检测和边界框提取  
  Real-time object detection and bounding box extraction
- **卡尔曼滤波 / Kalman Filtering**: 边界框轨迹平滑以提高检测稳定性  
  Bounding box trajectory smoothing for improved detection stability
- **多视角融合 / Multi-View Fusion**: 三相机系统提供全方位空间感知  
  Three-camera system provides comprehensive spatial awareness

### 训练优化 / Training Optimization
- **分布式训练 / Distributed Training**: 使用Accelerate库实现多GPU训练加速  
  Multi-GPU training acceleration using Accelerate library
- **梯度累积 / Gradient Accumulation**: 支持大批次训练在有限显存下  
  Support for large batch training with limited GPU memory
- **自适应学习率 / Adaptive Learning Rate**: 骨干网络和策略网络独立学习率调度  
  Independent learning rate scheduling for backbone and policy networks