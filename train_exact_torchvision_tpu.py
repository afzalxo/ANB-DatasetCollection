import os
import sys
import time
import glob
import numpy as np
import torch
import gc
import logging
import argparse
import configparser
import torch.utils
import torch.backends.cudnn as cudnn
import random
import warnings
from dataclasses import replace
from thop import profile

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import auxiliary.utils as utils
from trainval.trainval_tpu import train_x_epochs_tpu
from dataloader.torchvision_dataloader import build_loader_timm_tpu, build_torchvision_loader_tpu
from models.accelbenchnet import AccelNet as Network
from searchables import searchables

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler#, scheduler_kwargs
from timm.utils import ModelEmaV2, random_seed
from timm.bits import initialize_device, TrainCfg, setup_model_and_optimizer

from timm.utils import setup_default_logging

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

def profile_model(input_size, design, activation_fn, mode):
    model_temp = Network(design=design, activation_fn=activation_fn, mode=mode)
    input = torch.randn(input_size)
    macs, params = profile(model_temp, inputs=(input,))
    macs, params = macs / 1000000, params / 1000000
    del model_temp
    logging.info("MFLOPS %f, MPARAMS: %f", macs, params)
    gc.collect()
    return macs, params


def map_fn(index, args):
    dev_env = initialize_device(force_cpu=False, amp=False, channels_last=False)
    args.world_size = world_size = dev_env.world_size
    args.local_rank = local_rank = dev_env.local_rank
    args.global_rank = global_rank = dev_env.global_rank
    args.use_wandb = True if global_rank == 0 and args.use_wandb else False

    args.save = "{}tpuv2Train-{}-{}-{}".format(
        args.save, args.job_id, args.note, time.strftime("%Y%m%d-%H%M")
    )
    print(
        f"Global Rank {global_rank}, Local Rank {local_rank},\
        World Size {world_size}, Job ID {args.job_id}"
    )
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_printoptions(precision=4)
    np.set_printoptions(precision=4)
    if global_rank == 0:
        utils.create_exp_dir(
            args.save, scripts_to_save=glob.glob("**/*.py", recursive=True)
        )
        setup_default_logging(log_path=os.path.join(args.save, f"log_rank{args.global_rank}.txt"))
        '''
        log_format = f"%(asctime)s - Rank {args.global_rank} - %(message)s"
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format=log_format,
            datefmt="%m/%d %I:%M:%S %p",
        )
        fh = logging.FileHandler(
            os.path.join(args.save, f"log_rank{args.global_rank}.txt")
        )
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        '''
        logging.info("Initialized logging...")

    wandb_con = None
    wandb_art = None
    wandb_metadata_dir = None
    if args.use_wandb and global_rank == 0:
        wandb_metadata_dir = args.save
        import wandb

        os.environ["WANDB_API_KEY"] = "166a45fa2ad2b2db9ec555119b273a3a9bdacc41"
        os.environ["WANDB_ENTITY"] = "europa1610"
        os.environ["WANDB_PROJECT"] = "NASBenchFPGA"
        wandb_con = wandb.init(
            project="NASBenchFPGA",
            entity="europa1610",
            name=args.note + f"_{args.job_id}",
            settings=wandb.Settings(code_dir="."),
            dir=wandb_metadata_dir,
            group="deployment_models_tpu",
        )
        wandb_art = wandb.Artifact(
            name=f"train-code-deploytpuv3-jobid{args.job_id}", type="code"
        )
        wandb_art.add_dir(os.path.join(args.save, "scripts"))
        wandb_con.log_artifact(wandb_art)
        wandb_con.config.update(args)
        logging.info("Saving py files to wandb...")
        wandb_con.save("./*.py")
        wandb_con.save("./trainval/*.py")
        wandb_con.save("./configs/*.cfg")
        wandb_con.save("./dataloader/*.py")
        wandb_con.save("./auxiliary/*.py")
        wandb_con.save("./models/*.py")
        wandb_con.save("./models/ops/*.py")
        wandb_con.save("./searchables/*.py")
        logging.info("Saved py files to wandb...")
        args.wandb_con = wandb_con
    else:
        wandb_con = None
        wandb_art = None

    logging.info("args = %s", args)

    args.in_memory = True
    m = 0
    models_to_eval = 1
    device = xm.xla_device()
    # Dataloader here
    random_seed(args.seed, args.global_rank)
    train_queue, valid_queue = build_loader_timm_tpu(args)
    # train_queue, valid_queue = build_torchvision_loader_tpu(args)
    random_seed(args.seed, 0)
    criterion_train = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing).to(device)
    criterion_val = torch.nn.CrossEntropyLoss().to(device)
    args.writer = None
    if xm.is_master_ordinal():
        import torch_xla.test.test_utils as test_utils
        args.writer = test_utils.get_summary_writer(args.save)
    mixup_fn = None
    if args.use_mixup:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.label_smoothing,
            num_classes=args.CLASSES,
        )
    while m < models_to_eval:
        args.model_num = m
        args.design = searchables.CustomSearchable([4,6,6,6,6,6,6], [3,5,5,5,5,5,5], [2,1,3,3,3,4,3], [False, False, False, False, False, False, True])
        logging.info(
            "Job ID: %d, Model Number: %d, Design: \n%s",
            args.job_id,
            args.model_num,
            np.array(args.design),
        )
        activation_fn, mode = "relu", "train"
        args.macs, args.params = None, None
        '''
        if args.global_rank == 0:
            args.macs, args.params = profile_model(
                (1, 3, 224, 224),
                design=args.design,
                activation_fn=activation_fn,
                mode=mode,
            )
        '''
        args.opt = "rmsproptf"
        args.weight_decay = 1e-5
        args.momentum = 0.9
        args.opt_eps = 0.001

        args.epochs = 450
        args.decay_epochs = 2.4
        args.decay_rate = 0.97
        args.sched = "step"
        args.warmup_lr = 1e-6
        args.warmup_epochs = 5
        args.lr_cycle_decay = 0.5
        #args.decay_milestones = [90, 180, 270]
        args.decay_milestones = [30, 60]

        model = Network(design=args.design, activation_fn=activation_fn, mode=mode)
        model = model.to(device)

        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

        model_ema = None
        if args.model_ema:
            model_ema = ModelEmaV2(
                model,
                decay=args.model_ema_decay,
                device="cpu" if args.model_ema_force_cpu else None,
            )

        lr_scheduler, num_epochs = create_scheduler(args, optimizer)

        train_acc, train_loss, valid_t1, valid_t5, valid_loss = train_x_epochs_tpu(
            args.epochs,
            lr_scheduler,
            train_queue,
            valid_queue,
            model,
            model_ema,
            criterion_train,
            criterion_val,
            optimizer,
            mixup_fn,
            args,
        )
        logging.info(
            "Job ID: %d, Model Number: %d, Model Acc: %f",
            args.job_id,
            args.model_num,
            valid_t1,
        )
        del model, optimizer
        gc.collect()
        m += 1
    xm.rendezvous('checking_out')


# Map Function


def main():
    CLASSES = 1000
    DEFAULT_CROP_RATIO = 224 / 256

    parser = argparse.ArgumentParser("")
    parser.add_argument("--cfg_path")
    args = parser.parse_args()

    cfg_path = args.cfg_path

    config = configparser.ConfigParser()
    config.read(cfg_path)

    # Logging
    args.save = config["logging"]["save"]
    args.note = config["logging"]["note"]
    args.report_freq = config["logging"].getint("report_freq")
    args.fast = config["logging"].getboolean("fast")
    args.use_wandb = config["logging"].getboolean("use_wandb")
    # model
    args.label_smoothing = config["model"].getfloat("label_smoothing")
    args.design = config["model"]["design"]
    # dataloaders
    args.train_dataset = config["dataloader"]["train_dataset"]
    args.num_workers = config["dataloader"].getint("num_workers")
    args.in_memory = config["dataloader"].getboolean("in_memory")
    # trainval
    args.epochs = config["trainval"].getint("epochs")
    args.train_batch_size = config["trainval"].getint("train_batch_size")
    args.val_batch_size = config["trainval"].getint("val_batch_size")
    args.seed = config["trainval"].getint("seed")
    # optimizer
    args.lr = config["optimizer"].getfloat("lr")
    args.weight_decay = config["optimizer"].getfloat("weight_decay")
    args.min_lr = config["optimizer"].getfloat("min_lr")
    args.warmup_epochs = config["optimizer"].getint("warmup_epochs")
    # distributed
    args.distributed = config["distributed"].getboolean("distributed")
    args.cluster = config["distributed"]["cluster"]
    args.port = config["distributed"]["port"]

    args.CLASSES = CLASSES
    args.DEFAULT_CROP_RATIO = DEFAULT_CROP_RATIO
    args.opt_eps = 1e-8

    args.use_mixup = False
    args.mixup = 0.0
    args.mixup_mode = "batch"
    args.mixup_prob = 1
    args.mixup_switch_prob = 0.5
    args.cutmix = 0
    args.cutmix_minmax = None

    args.model_ema = True
    args.model_ema_decay = 0.99995
    args.model_ema_force_cpu = False

    # os.environ["XLA_USE_BF16"] = "1"
    job_id = os.getpid()
    ip = "127.0.0.1"

    args.ip = ip
    args.job_id = job_id
    args.update_freq = 1
    flags = args
    xmp.spawn(map_fn, args=(flags,), nprocs=8)#, start_method="fork")
    exit(0)


if __name__ == "__main__":
    start_time = time.time()
    main()
    duration = time.time() - start_time
    logging.info("Total eval time: %ds", duration)
