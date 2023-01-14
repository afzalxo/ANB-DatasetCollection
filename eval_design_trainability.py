import sys
import time
import numpy as np
import torch
import gc
import logging
import torch.utils
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
import random
import warnings

from models.accelbenchnet import AccelNet as Network

warnings.filterwarnings("ignore")


def profile_memory(design, platform, mode, gpu):
    after_backward, before_model = 0, 0
    try:
        criterion = torch.nn.CrossEntropyLoss()
        before_model = torch.cuda.max_memory_allocated()
        model_temp = Network(design=design, platform=platform, mode=mode).to(f'cuda:{gpu}')
        after_model = torch.cuda.max_memory_allocated()
        input_mem = torch.randn((512, 3, 192, 192), dtype=torch.float32).to(f'cuda:{gpu}')
        target = torch.randn((512, 1000), dtype=torch.float32).to(f'cuda:{gpu}')
        with autocast():
            output = model_temp(input_mem.contiguous(memory_format=torch.channels_last))
        after_forward = torch.cuda.max_memory_allocated()
        loss = criterion(output, target)
        loss.backward()
        after_backward = torch.cuda.max_memory_allocated()
        print(f'Forward Only: {(after_forward-after_model)/10**9}, Forward-Backward Total: {(after_backward-after_model)/10**9}, Including Model: {(after_backward-before_model)/10**9}')
        del model_temp, target, input_mem, output, loss
        torch.cuda.empty_cache()
        gc.collect()
        trainable = True
    except RuntimeError:
        print('Ran out of GPU memory... Untrainable...')
        trainable = False
    return after_backward - before_model, trainable


def eval_trainability(architecture_json, gpu):
    if not torch.cuda.is_available():
        logging.info("No GPU device available")
        sys.exit(1)
    # trainval
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.set_printoptions(precision=4)
    np.set_printoptions(precision=4)

    design = architecture_json
    # design_list = utils.config_json_to_list_repr(args.architecture_json)
    # args.design = searchables.CustomSearchable(
    #    e=design_list[0], k=design_list[1], la=design_list[2], se=design_list[3]
    # )
    '''
    logging.info(
        "Design: \n%s",
        np.array(design),
    )
    '''
    platform, mode = "fpga", "train"
    print('Checking design trainability...')
    mem, trainable = profile_memory(design, platform, mode, gpu)
    if not trainable:
        logging.info(
            "Design not trainable due to GPU mem overflows...\n"
        )
    else:
        print('Design trainable...')

    torch.cuda.empty_cache()
    gc.collect()
    return trainable
