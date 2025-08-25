# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""杂项功能模块 - Miscellaneous Functions Module

本模块包含了各种工具函数和辅助类，主要用于分布式训练、指标记录和数据处理。
大部分代码从 TorchVision 参考实现中复制而来，经过适配和优化。

Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references, adapted and optimized.

主要组件 / Main Components:
- SmoothedValue: 平滑数值记录器，用于训练指标跟踪 / Smoothed value recorder for training metrics tracking
- MetricLogger: 指标日志记录器，统一管理多个指标 / Metric logger for unified multi-metric management
- NestedTensor: 嵌套张量类，处理不同尺寸的图像批次 / Nested tensor class for handling variable-sized image batches
- 分布式训练工具函数 / Distributed training utility functions
- Git版本管理工具 / Git version management tools

功能特点 / Features:
- 支持多 GPU 分布式训练 / Multi-GPU distributed training support
- 灵活的指标统计和显示 / Flexible metric statistics and display
- 高效的数据批处理 / Efficient data batch processing
- ONNX 导出兼容性 / ONNX export compatibility
"""
# 标准库导入 / Standard library imports
import os                      # 操作系统接口 / Operating system interface
import subprocess             # 子进程管理 / Subprocess management
import time                   # 时间处理工具 / Time processing utilities
import datetime               # 日期时间处理 / Date and time processing
import pickle                 # 对象序列化 / Object serialization
from collections import defaultdict, deque  # 默认字典和双端队列 / Default dict and double-ended queue
from packaging import version # 版本解析工具 / Version parsing utilities
from typing import Optional, List  # 类型提示 / Type hints

# PyTorch相关导入 / PyTorch related imports
import torch                  # PyTorch核心库 / PyTorch core library
import torch.distributed as dist  # 分布式训练 / Distributed training
from torch import Tensor      # 张量类型 / Tensor type

# TorchVision导入和版本兼容性处理 / TorchVision imports and version compatibility handling
# 由于pytorch和torchvision 0.5中的空张量bug而需要 / Needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision            # 计算机视觉库 / Computer vision library
if version.parse(torchvision.__version__) < version.parse('0.7'):  # 版本兼容性检查 / Version compatibility check
    from torchvision.ops import _new_empty_tensor    # 空张量创建函数 / Empty tensor creation function
    from torchvision.ops.misc import _output_size    # 输出尺寸计算函数 / Output size calculation function


class SmoothedValue(object):
    """平滑数值记录器 - Smoothed Value Recorder
    
    跟踪一系列数值，并提供在滑动窗口或全局序列上的平滑数值访问。
    常用于训练过程中的损失、准确率等指标的平滑显示。
    
    Track a series of values and provide access to smoothed values over a
    window or the global series average. Commonly used for smoothed display
    of loss, accuracy and other metrics during training.
    
    提供的统计量 / Provided Statistics:
    - median: 中位数 / Median value
    - avg: 窗口平均值 / Window average
    - global_avg: 全局平均值 / Global average  
    - max: 最大值 / Maximum value
    - value: 最新值 / Latest value
    """

    def __init__(self, window_size=20, fmt=None):
        """初始化平滑数值记录器 / Initialize smoothed value recorder
        
        Args:
            window_size (int): 滑动窗口大小，默认20 / Sliding window size, default 20
            fmt (str): 显示格式字符串 / Display format string
        """
        if fmt is None:  # 如果没有指定显示格式 / If no display format specified
            fmt = "{median:.4f} ({global_avg:.4f})"  # 默认格式：中位数(全局平均) / Default format: median (global_avg)
            
        # 初始化内部数据结构 / Initialize internal data structures
        self.deque = deque(maxlen=window_size)  # 固定大小的双端队列，用于滑动窗口 / Fixed-size deque for sliding window
        self.total = 0.0   # 累计总和 / Cumulative total
        self.count = 0     # 累计计数 / Cumulative count
        self.fmt = fmt     # 显示格式 / Display format

    def update(self, value, n=1):
        """更新数值记录 / Update value record
        
        Args:
            value (float): 新数值 / New value
            n (int): 数值数量（用于批量更新）/ Value count (for batch updates)
        """
        self.deque.append(value)  # 添加到滑动窗口 / Add to sliding window
        self.count += n           # 更新总计数 / Update total count
        self.total += value * n   # 更新加权总和 / Update weighted total sum

    def synchronize_between_processes(self):
        """在多进程之间同步数值 / Synchronize values between processes
        
        警告：不会同步deque窗口数据！
        Warning: does not synchronize the deque window data!
        """
        if not is_dist_avail_and_initialized():  # 检查分布式环境 / Check distributed environment
            return  # 非分布式环境直接返回 / Return directly in non-distributed environment
            
        # 创建包含计数和总和的张量 / Create tensor containing count and total
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()      # 同步障障，等待所有进程 / Synchronization barrier, wait for all processes
        dist.all_reduce(t)  # 所有进程的值相加 / Sum values across all processes
        t = t.tolist()      # 转换为Python列表 / Convert to Python list
        
        # 更新同步后的值 / Update synchronized values
        self.count = int(t[0])  # 更新计数 / Update count
        self.total = t[1]       # 更新总和 / Update total

    @property
    def median(self):
        """计算窗口内数值的中位数 / Calculate median of values in window"""
        d = torch.tensor(list(self.deque))  # 转换为张量 / Convert to tensor
        return d.median().item()  # 返回中位数 / Return median value

    @property
    def avg(self):
        """计算窗口内数值的平均值 / Calculate average of values in window"""
        d = torch.tensor(list(self.deque), dtype=torch.float32)  # 转换为浮点张量 / Convert to float tensor
        return d.mean().item()  # 返回平均值 / Return mean value

    @property
    def global_avg(self):
        """计算全局平均值 / Calculate global average value"""
        return self.total / self.count  # 总和除以总数 / Total sum divided by total count

    @property
    def max(self):
        """获取窗口内最大值 / Get maximum value in window"""
        return max(self.deque)  # 返回队列中最大值 / Return maximum value in deque

    @property
    def value(self):
        """获取最新的数值 / Get latest value"""
        return self.deque[-1]  # 返回队列最后一个元素 / Return last element in deque

    def __str__(self):
        """格式化显示所有统计信息 / Format display all statistical information"""
        return self.fmt.format(  # 使用指定格式显示 / Display using specified format
            median=self.median,        # 中位数 / Median
            avg=self.avg,             # 窗口平均 / Window average
            global_avg=self.global_avg, # 全局平均 / Global average
            max=self.max,             # 最大值 / Maximum
            value=self.value          # 最新值 / Latest value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    """嵌套张量类 - Nested Tensor Class
    
    用于处理不同尺寸的图像批次，通过填充和掩码机制实现统一处理。
    在Transformer模型中特别有用，可以高效处理变长序列数据。
    
    Used for handling image batches of different sizes through padding and masking.
    Particularly useful in Transformer models for efficient variable-length sequence processing.
    
    组件 / Components:
    - tensors: 填充后的张量数据 / Padded tensor data
    - mask: 掩码张量，标记有效区域 / Mask tensor marking valid regions
    """
    
    def __init__(self, tensors, mask: Optional[Tensor]):
        """初始化嵌套张量 / Initialize nested tensor
        
        Args:
            tensors: 主张量数据 / Main tensor data
            mask: 掩码张量（可选）/ Mask tensor (optional)
        """
        self.tensors = tensors  # 保存张量数据 / Store tensor data
        self.mask = mask        # 保存掩码信息 / Store mask information

    def to(self, device):
        """将嵌套张量移动到指定设备 / Move nested tensor to specified device
        
        Args:
            device: 目标设备（CPU或GPU）/ Target device (CPU or GPU)
            
        Returns:
            NestedTensor: 新的嵌套张量对象 / New nested tensor object
        """
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)  # 移动主张量到目标设备 / Move main tensor to target device
        mask = self.mask  # 获取掩码 / Get mask
        
        if mask is not None:  # 如果存在掩码 / If mask exists
            assert mask is not None  # 断言检查 / Assertion check
            cast_mask = mask.to(device)  # 移动掩码到目标设备 / Move mask to target device
        else:  # 无掩码情况 / No mask case
            cast_mask = None  # 设置为None / Set to None
            
        return NestedTensor(cast_tensor, cast_mask)  # 返回新的嵌套张量 / Return new nested tensor

    def decompose(self):
        """分解嵌套张量为组成部分 / Decompose nested tensor into components
        
        Returns:
            tuple: (张量, 掩码) / (tensor, mask)
        """
        return self.tensors, self.mask  # 返回张量和掩码 / Return tensor and mask

    def __repr__(self):
        """返回嵌套张量的字符串表示 / Return string representation of nested tensor"""
        return str(self.tensors)  # 返回主张量的字符串表示 / Return string representation of main tensor


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)
