import os
import numpy as np
import collections

from typing import Any, Optional, Tuple
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
import shutil
import torchvision.transforms as transforms
import json

from timm.utils.model import unwrap_model

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def distribute_bn(model, world_size, reduce=False):
    # ensure every node has the same running bn stats
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                # average bn stats across whole group
                torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                bn_buf /= float(world_size)
            else:
                # broadcast bn stats from rank 0 to whole group
                torch.distributed.broadcast(bn_buf, 0)


def reduce_xla_tensor(tens, world_size):
    reduced = xm.all_reduce(xm.REDUCE_SUM, tens, scale=1/world_size)
    return reduced


def distribute_bn_tpu(model, world_size, reduce=True):
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                bn_buf = reduce_xla_tensor(bn_buf, world_size)
            else:
                raise NotImplementedError


def is_distributed_env():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break

    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break

    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0

    dist_backend = getattr(args, 'dist_backend', 'nccl')
    dist_url = getattr(args, 'dist_url', 'env://')
    if is_distributed_env():
        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            args.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
            )
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
        args.distributed = True

    if torch.cuda.is_available():
        if args.distributed:
            device = 'cuda:%d' % args.local_rank
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    args.device = device
    device = torch.device(device)
    return device

def softmax(x, axis=0):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)


def config_json_to_list_repr(arch_json):
    with open(arch_json, "r") as fh:
        data = json.load(fh)

    # Initialize empty lists for each key
    values_e = []
    values_k = []
    values_l = []
    values_se = []

    # Iterate through the dictionary and append values to the appropriate list
    for i in range(7):
        key = f"block{i}"
        values_e.append(data[f"{key}_e"])
        values_k.append(data[f"{key}_k"])
        values_l.append(data[f"{key}_l"])
        values_se.append(True if data[f"{key}_se"] else False)

    # Make a list of lists with the values
    value_lists = [
        list(map(int, values_e)),
        list(map(int, values_k)),
        list(map(int, values_l)),
        values_se
    ]

    return value_lists


def _restore_weights(old_dict, new_dict, param_storage):
    for k in old_dict:
        if k in new_dict and not (
            k.endswith("alphas_normal") or k.endswith("alphas_reduce")
        ):
            new_dict[k] = old_dict[k]
        elif k not in new_dict and not (
            k.endswith("alphas_normal") or k.endswith("alphas_reduce")
        ):
            # Logic to store parameters absent from new dict
            # print('Storing entry missing in new_dict: %s' % k)
            param_storage[k] = old_dict[k]

    for k in new_dict:
        if k not in old_dict and not (
            k.endswith("alphas_normal") or k.endswith("alphas_reduce")
        ):
            # Parameters in new_dict that are absent in old dict (i.e. newly added operations)
            if k in param_storage:
                # print('Restoring key found in param storage: %s', k)
                new_dict[k] = param_storage[k]
                # Remove k from param_storage when restored
                try:
                    param_storage.pop(k)
                except KeyError:
                    print("Could not pop key... This should never happen... %s" % k)
            # else:
            # print('New key could not be found in param storage: %s', k)
    # print(len(old_dict), len(new_dict))
    return new_dict, param_storage


def restore_weights(
    old_dict, new_dict, layer_reinit=0, stages=[3, 3, 9, 3], distributed=True
):
    for k in old_dict:
        # print(k.split('.'), len(k.split('.')))
        # print(k in new_dict)
        layer = None
        if "stage" in k:
            stage = int(k.split(".")[0 + distributed][-1])
            cell = int(k.split(".")[2 + distributed])
            layer = sum(stages[:stage]) + cell
            # print(k, ' ', layer)
        if (
            k in new_dict
            and old_dict[k].shape == new_dict[k].shape
            and layer_reinit != layer
        ):
            # print('Found key in both new and old dict with same shape: %s' % k)
            new_dict[k] = old_dict[k]
        elif (
            k not in new_dict
            or old_dict[k].shape != new_dict[k].shape
            or layer_reinit == layer
        ):
            # print('Key missing from new_dict or size mismatch or reinit layer: %s' % k)
            continue
            # Logic to store parameters absent from new dict
            # param_storage[k] = old_dict[k]

    return new_dict


def restore_weights2(
    old_dict,
    new_dict,
    layer_disc=None,
    restore_edge=None,
    stages=[3, 3, 9, 3],
    distributed=True,
):
    for k in old_dict:
        # print(k.split('.'), len(k.split('.')))
        # print(k in new_dict)
        layer = None
        if "stage" in k:
            stage = int(k.split(".")[0 + distributed][-1])
            cell = int(k.split(".")[2 + distributed])
            layer = sum(stages[:stage]) + cell
            branch = int(k.split(".")[6 + distributed])
            # print(k, ' ', layer)
        if (
            k in new_dict
            and old_dict[k].shape == new_dict[k].shape
            and layer != layer_disc
        ):
            # print('Found key in both new and old dict with same shape: %s, %s' % (k, old_dict[k].shape))
            new_dict[k] = old_dict[k]
        elif (
            k not in new_dict or old_dict[k].shape != new_dict[k].shape
        ) and layer != layer_disc:
            # print('Key missing from new_dict or size mismatch or reinit layer: %s, %s, %d, %d' % (k, old_dict[k].shape, layer, layer_disc))
            continue
    for k in new_dict:
        layer = None
        if "stage" in k:
            stage = int(k.split(".")[0 + distributed][-1])
            cell = int(k.split(".")[2 + distributed])
            layer = sum(stages[:stage]) + cell
            branch = int(k.split(".")[6 + distributed])
            # print(k, ' ', layer)
        if layer == layer_disc and restore_edge is not None:
            ss = k.split(".")
            ss[6 + distributed] = str(restore_edge)
            k_new = ".".join(ss)
            # print(k, k_new)
            if k_new in old_dict and old_dict[k_new].shape == new_dict[k].shape:
                print("Restoring edge %s, %s" % (k_new, new_dict[k].shape))
                new_dict[k] = old_dict[k_new]
            elif k_new not in old_dict:
                # print('Did not find key in old dict: %s' % (k_new))
                continue
            elif old_dict[k_new].shape != new_dict[k].shape:
                # print('Shape mismatch: %s, %s, %s' % (k_new, old_dict[k_new].shape, new_dict[k].shape))
                continue

            # Logic to store parameters absent from new dict
            # param_storage[k] = old_dict[k]

    return new_dict


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def count_parameters_in_MB(model):
    return (
        np.sum(
            np.prod(v.size())
            for name, v in model.named_parameters()
            if "auxiliary" not in name
        )
        / 1e6
    )


def save_checkpoint(state, save):
    filename = os.path.join(save, "f_model.pth")
    torch.save(state, filename, _use_new_zipfile_serialization=False)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def create_optimizer(model, lr, weight_decay, epochs):
    parameter_group_names = {}
    parameter_group_vars = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if group_name not in parameter_group_names:
            scale = 1.0

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    parameters = list(parameter_group_vars.values())
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    return optimizer, scheduler


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


"""
def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x
"""


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


class LayerNorm2d(nn.LayerNorm):
    r"""LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1),
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            ).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print("Experiment log dir : {}".format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            if os.path.dirname(script) == "":
                dst_file = os.path.join(path, "scripts", os.path.basename(script))
            else:
                pth = os.path.join(path, "scripts", os.path.dirname(script))
                if not os.path.exists(pth):
                    os.makedirs(pth)
                dst_file = os.path.join(path, "scripts", script)
            shutil.copyfile(script, dst_file)


def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]


def adjust_lr(optimizer, epoch, argslr, epochs):
    # Smaller slope for the last 5 epochs
    # because lr * 1/250 is relatively large
    if epochs - epoch > 5:
        lr = argslr * (epochs - 5 - epoch) / (epochs - 5)
    else:
        lr = argslr * (epochs - epoch) / ((epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8
    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


