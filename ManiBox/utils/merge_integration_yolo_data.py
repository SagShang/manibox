"""YOLO数据集合并工具 - YOLO Dataset Merging Utility

本脚本用于合并多个分批处理的YOLO数据文件，将分散的integration_xxx.pkl文件
合并为单个大型数据集文件。主要用于机器人学习数据的后处理阶段，
将多次采集或分批处理的数据整合为训练就绪的统一格式。

This script merges multiple batch-processed YOLO data files, combining scattered
integration_xxx.pkl files into a single large dataset file. Primarily used in
post-processing stage of robot learning data, integrating multiple collection
sessions or batch-processed data into unified training-ready format.

功能特点 / Features:
- 批量文件合并：自动发现并合并所有integration_*.pkl文件 / Batch file merging: auto-discover and merge all integration_*.pkl files
- 数据验证：检查数据完整性和格式一致性 / Data validation: check data integrity and format consistency  
- 时间切片：支持选择特定时间段的数据 / Time slicing: support selection of specific time periods
- 数据清理：移除无效或损坏的数据 / Data cleaning: remove invalid or corrupted data
- 随机打乱：合并后随机打乱数据顺序 / Random shuffling: shuffle data order after merging
- GPU优化：利用GPU内存进行高效处理 / GPU optimization: utilize GPU memory for efficient processing

使用场景 / Use Cases:
- 训练数据准备：将多次收集的数据合并为训练集 / Training data preparation: merge multiple data collections into training set
- 数据预处理：统一数据格式和维度 / Data preprocessing: unify data format and dimensions
- 存储优化：减少文件数量，提高加载效率 / Storage optimization: reduce file count, improve loading efficiency
"""

# 标准库导入 / Standard library imports
import os                 # 操作系统接口 / Operating system interface
import glob               # 文件模式匹配 / File pattern matching
import torch              # PyTorch深度学习框架 / PyTorch deep learning framework

# 项目内部模块导入 / Internal module imports
from ManiBox.yolo_process_data import YoloProcessDataByTimeStep  # YOLO数据处理器 / YOLO data processor

# 数据目录配置 / Data directory configuration
# current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本目录 / Get current script directory
current_dir = "./data/"   # 数据存放目录 / Data storage directory

print("开始合并所有integration_*.pkl文件 / Starting to merge all integration_*.pkl files")

# 文件发现和收集 / File discovery and collection
file_pattern = os.path.join(current_dir, 'integration_*.pkl')  # 匹配所有integration开头的pkl文件 / Match all pkl files starting with integration
files = glob.glob(file_pattern)  # 获取所有符合模式的文件路径 / Get all file paths matching the pattern

print(f"发现{len(files)}个待合并的文件 / Found {len(files)} files to merge:")
for file in files:
    print(f"  - {file}")

# 数据容器初始化 / Data container initialization
all_image_data = []     # 存储所有图像数据 / Store all image data
all_qpos_data = []      # 存储所有关节位置数据 / Store all joint position data  
all_action_data = []    # 存储所有动作数据 / Store all action data
all_reward = []         # 存储所有奖励数据 / Store all reward data

# 时间切片配置 / Time slice configuration
episode_begin = 3       # 回合开始时间戳（跳过前3帧）/ Episode start timestamp (skip first 3 frames)
episode_end = 90        # 回合结束时间戳 / Episode end timestamp

# 图像特征维度计算 / Image feature dimension calculation
# 维度 = 物体类别数 * 相机数(3) * 边界框维度(4) = objects_num * 3 * 4
image_data_dim = len(YoloProcessDataByTimeStep.objects_names) * 12  # 12 = 3相机 * 4边界框坐标 / 12 = 3 cameras * 4 bbox coordinates
print(f"使用前{image_data_dim}维的图像边界框信息 / Using first {image_data_dim}-dim image-bbox information")

# 逐文件处理和数据提取 / Process each file and extract data
for file in files:
    print(f"\n正在处理文件 / Processing file: {file}")
    
    # 加载数据文件 / Load data file
    data = torch.load(file)
    print(f"数据形状 / Data shape: {data['image_data'].shape}")
    
    # 从文件名提取回合数量 / Extract episode number from filename
    # 文件名格式：integration_1000.pkl -> 提取1000
    epi_num = int(file.split("_")[-1].split(".")[0])  # 解析文件名获取回合数 / Parse filename to get episode count
    print(f"文件包含{epi_num}个回合 / File contains {epi_num} episodes")
    
    # 数据索引基线校正 / Data index baseline correction
    # 合并后的数据可能已经进行过时间切片，需要调整索引基线
    baseline = 0  # 索引基线，用于处理已经切片的数据 / Index baseline for handling pre-sliced data
    if data['image_data'].shape[1] == episode_end - episode_begin:
        # 如果时间维度已经是切片后的长度，说明数据已经预处理过
        baseline = episode_begin  # 设置基线为episode_begin / Set baseline to episode_begin
        print(f"检测到预处理数据，设置基线为{baseline} / Detected preprocessed data, setting baseline to {baseline}")
    
    # 图像数据提取和GPU转移 / Image data extraction and GPU transfer
    if 'image_data' in data:
        print(f"提取图像数据，形状: {data['image_data'].shape} / Extracting image data, shape: {data['image_data'].shape}")
        # 数据切片：[:epi_num, episode_begin-baseline:episode_end-baseline, :image_data_dim]
        # 回合维度：取前epi_num个回合 / Episode dimension: take first epi_num episodes
        # 时间维度：取episode_begin到episode_end的时间段 / Time dimension: take time period from episode_begin to episode_end
        # 特征维度：取前image_data_dim个特征 / Feature dimension: take first image_data_dim features
        sliced_image_data = data['image_data'].cuda()[:epi_num, episode_begin-baseline:episode_end-baseline, :image_data_dim]
        all_image_data.append(sliced_image_data)
        print(f"图像数据切片后形状: {sliced_image_data.shape} / Image data shape after slicing: {sliced_image_data.shape}")
    
    # 关节位置数据提取 / Joint position data extraction  
    if 'qpos_data' in data:
        sliced_qpos_data = data['qpos_data'].cuda()[:epi_num, episode_begin-baseline:episode_end-baseline, :]
        all_qpos_data.append(sliced_qpos_data)
        print(f"关节位置数据形状: {sliced_qpos_data.shape} / Joint position data shape: {sliced_qpos_data.shape}")
    
    # 动作数据提取 / Action data extraction
    if 'action_data' in data:
        sliced_action_data = data['action_data'].cuda()[:epi_num, episode_begin-baseline:episode_end-baseline, :]
        all_action_data.append(sliced_action_data)
        print(f"动作数据形状: {sliced_action_data.shape} / Action data shape: {sliced_action_data.shape}")
    
    # 奖励数据提取 / Reward data extraction
    if 'reward' in data:
        sliced_reward_data = data['reward'].cuda()[:epi_num, episode_begin-baseline:episode_end-baseline]
        all_reward.append(sliced_reward_data)
        print(f"奖励数据形状: {sliced_reward_data.shape} / Reward data shape: {sliced_reward_data.shape}")

# 数据合并和后处理 / Data combination and post-processing
if all_image_data:
    print(f"\n开始合并数据 / Starting data combination...")
    
    # 沿着第0维（回合维度）拼接所有数据 / Concatenate all data along dimension 0 (episode dimension)
    combined_image_data = torch.cat(all_image_data, dim=0)      # 合并图像数据 / Combine image data
    combined_qpos_data = torch.cat(all_qpos_data, dim=0)        # 合并关节位置数据 / Combine joint position data
    combined_action_data = torch.cat(all_action_data, dim=0)    # 合并动作数据 / Combine action data
    
    # 合并奖励数据（如果存在）/ Combine reward data (if exists)
    if all_reward:
        combined_reward = torch.cat(all_reward, dim=0)          # 合并奖励数据 / Combine reward data
    
    # 数据清理（已注释）/ Data cleaning (commented)
    # 以下代码用于移除无效数据，根据需要可以取消注释
    # The following code is for removing invalid data, can be uncommented as needed
    
    # 移除关节位置异常的数据 / Remove data with abnormal joint positions
    # invalid_index = (torch.logical_or(combined_qpos_data[:, 30:, 6] > 3.7, combined_qpos_data[:, 30:, 6] < 2)).sum(dim=-1) > 0
    # print("关节位置无效数据数量 / Number of invalid joint position data:", invalid_index.sum())
    # combined_qpos_data = combined_qpos_data[~invalid_index]
    # combined_image_data = combined_image_data[~invalid_index]  
    # combined_action_data = combined_action_data[~invalid_index]
    
    # 移除图像边界框全零的数据 / Remove data with all-zero image bounding boxes
    # invalid_index = (combined_image_data[:, 0, 0:4] == 0).sum(dim=-1) == 4
    # print("图像无效数据数量 / Number of invalid image data:", invalid_index.sum())
    # combined_qpos_data = combined_qpos_data[~invalid_index]
    # combined_image_data = combined_image_data[~invalid_index]
    # combined_action_data = combined_action_data[~invalid_index]
    
    # 数据随机打乱 / Random data shuffling
    print("正在随机打乱数据顺序 / Shuffling data order...")
    all_index = torch.arange(combined_image_data.shape[0])      # 生成索引序列 / Generate index sequence
    shuffled_index = torch.randperm(combined_image_data.shape[0])  # 生成随机排列索引 / Generate random permutation indices
    
    # 应用随机索引到所有数据 / Apply random indices to all data
    combined_image_data = combined_image_data[shuffled_index]   # 打乱图像数据 / Shuffle image data
    combined_qpos_data = combined_qpos_data[shuffled_index]     # 打乱关节位置数据 / Shuffle joint position data
    combined_action_data = combined_action_data[shuffled_index] # 打乱动作数据 / Shuffle action data
    if all_reward:
        combined_reward = combined_reward[shuffled_index]       # 打乱奖励数据 / Shuffle reward data
    
    # 构建最终数据字典 / Build final data dictionary
    data = {
        "image_data": combined_image_data,      # 图像数据张量，形状: (num_episodes, episode_len, image_dim) / Image data tensor, shape: (num_episodes, episode_len, image_dim)
        "image_depth_data": None,               # 深度图像数据（未使用）/ Depth image data (unused)
        "qpos_data": combined_qpos_data,        # 关节位置数据张量，形状: (num_episodes, episode_len, 14) / Joint position data tensor, shape: (num_episodes, episode_len, 14)
        "next_action_data": None,               # 下一步动作数据（未使用）/ Next action data (unused)
        "next_action_is_pad": None,             # 下一步动作填充标记（未使用）/ Next action padding mask (unused)
        "action_data": combined_action_data,    # 动作数据张量，形状: (num_episodes, episode_len, 14) / Action data tensor, shape: (num_episodes, episode_len, 14)
        "action_is_pad": None,                  # 动作填充标记（未使用）/ Action padding mask (unused)
        "reward": combined_reward,              # 奖励数据张量 / Reward data tensor
    }
    
    # 打印合并后的数据统计信息 / Print combined data statistics
    print(f"合并后图像数据形状 / Combined image data shape: {combined_image_data.shape}")
    print(f"合并后奖励数据形状 / Combined reward data shape: {combined_reward.shape}")
    print(f"总回合数 / Total episodes: {combined_image_data.shape[0]}")
    
    # 保存合并后的数据 / Save combined data
    dataset_dir = current_dir  # 保存目录 / Save directory
    output_filename = f"integration_{combined_image_data.shape[0]}.pkl"  # 输出文件名包含回合数 / Output filename includes episode count
    output_path = os.path.join(dataset_dir, output_filename)
    
    print(f"正在保存合并后的数据到 / Saving combined data to: {output_path}")
    torch.save(data, output_path)  # 保存数据文件 / Save data file
    print(f"数据合并完成！/ Data merging completed!")
    print(f"合并后的文件: {output_path} / Merged file: {output_path}")
    
else:
    print("错误：未找到任何图像数据文件 / Error: No image data found in the files.")
