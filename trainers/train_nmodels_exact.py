import os
import sys
import subprocess
import time
import glob
import numpy as np
import torch
import logging
import argparse
import configparser
import torch.utils
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import random
import warnings
from thop import profile

import auxiliary.utils as utils
from trainval.trainval import train_x_epochs
from dataloader.ffcv_dataloader import get_ffcv_loaders
from auxiliary.utils import CrossEntropyLabelSmooth
from models.efficientnet import efficientnet_b0 as Network
from searchables import searchables

warnings.filterwarnings("ignore")


def setup_distributed(rank, local_rank, address, port, world_size, cluster):
    os.environ["MASTER_ADDR"] = address
    os.environ["MASTER_PORT"] = port
    # os.environ['NCCL_IB_DISABLE'] = '1'
    # os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    print("Setting up dist training rank %d" % rank)
    if cluster == "local":
        init_method = "file:///home/aahmadaa/newf1"
    elif cluster == "tacc":
        init_method = "env://"

    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size
    )
    torch.cuda.set_device(local_rank)


def cleanup_distributed():
    dist.destroy_process_group()


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


def profile_model(input_size, design, platform, mode):
    model_temp = Network(design=design, platform=platform, mode=mode)
    input = torch.randn(input_size)
    macs, params = profile(model_temp, inputs=(input,))
    macs, params = macs / 1000000, params / 1000000
    del model_temp
    print(f"MFLOPS: {macs}, MPARAMS: {params}")
    return macs, params


def main():
    if not torch.cuda.is_available():
        logging.info("No GPU device available")
        sys.exit(1)
    CLASSES = 1000
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
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
    args.val_dataset = config["dataloader"]["val_dataset"]
    args.num_workers = config["dataloader"].getint("num_workers")
    args.in_memory = config["dataloader"].getboolean("in_memory")
    # trainval
    args.epochs = config["trainval"].getint("epochs")
    args.train_batch_size = config["trainval"].getint("train_batch_size")
    args.val_batch_size = config["trainval"].getint("val_batch_size")
    args.val_resolution = config["trainval"].getint("val_resolution")
    args.lr_tta = config["trainval"].getboolean("lr_tta")
    args.min_res = config["trainval"].getint("min_res")
    args.max_res = config["trainval"].getint("max_res")
    args.start_ramp = config["trainval"].getint("start_ramp")
    args.end_ramp = config["trainval"].getint("end_ramp")
    args.seed = config["trainval"].getint("seed")
    # optimizer
    args.lr = config["optimizer"].getfloat("lr")
    args.weight_decay = config["optimizer"].getfloat("weight_decay")
    args.lr_peak_epoch = config["optimizer"].getint("lr_peak_epoch")
    # distributed
    args.distributed = config["distributed"].getboolean("distributed")
    args.cluster = config["distributed"]["cluster"]
    args.port = config["distributed"]["port"]

    args.CLASSES = CLASSES
    args.IMAGENET_MEAN = IMAGENET_MEAN
    args.IMAGENET_STD = IMAGENET_STD
    args.DEFAULT_CROP_RATIO = DEFAULT_CROP_RATIO

    if args.distributed and args.cluster == "tacc":
        global_rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        iplist = os.environ["SLURM_JOB_NODELIST"]
        job_id = os.environ['SLURM_JOB_ID']
        ip = subprocess.getoutput(f"scontrol show hostname {iplist} | head -n1")
        setup_for_distributed(global_rank == 0)
    elif args.distributed and args.cluster == "local":
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        job_id = int(os.getpid())
        ip = "127.0.0.1"
        setup_for_distributed(global_rank == 0)
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        job_id = 0
        ip = "localhost"

    args.world_size = world_size
    args.local_rank = local_rank
    args.global_rank = global_rank
    args.ip = ip
    args.job_id = job_id

    USE_WANDB = args.use_wandb if global_rank == 0 else False

    args.save = "{}exp-{}-{}".format(
        args.save, args.note, time.strftime("%Y%m%d-%H%M%S")
    )
    print(
        f"Global Rank {global_rank}, Local Rank {local_rank},\
        World Size {world_size}, Job ID {args.job_id}"
    )
    if global_rank == 0:
        #utils.create_exp_dir(args.save, scripts_to_save=glob.glob("**/*.py", recursive=True))
        utils.create_exp_dir(args.save, scripts_to_save=None)
    wandb_con = None
    wandb_art = None
    wandb_metadata_dir = None
    if USE_WANDB:
        wandb_metadata_dir = args.save
        import wandb
        os.environ["WANDB_API_KEY"] =\
            "166a45fa2ad2b2db9ec555119b273a3a9bdacc41"
        os.environ["WANDB_ENTITY"] = "europa1610"
        os.environ["WANDB_PROJECT"] = "NASBenchFPGA"
        wandb_con = wandb.init(
            project="NASBenchFPGA",
            entity="europa1610",
            name=args.note,
            settings=wandb.Settings(code_dir="."),
            dir=wandb_metadata_dir,
        )
        #wandb_art = wandb.Artifact(name=f'train-code-nasfpga-jobid{job_id}', type='code')
        #wandb_art.add_dir(os.path.join(args.save, 'scripts'))
        #wandb_con.log_artifact(wandb_art)
    else:
        wandb_con = None
        wandb_art = None

    if global_rank == 0:
        log_format = "%(asctime)s %(message)s"
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format=log_format,
            datefmt="%m/%d %I:%M:%S %p",
        )
        fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    logging.info("args = %s", args)
    torch.set_printoptions(precision=4)
    np.set_printoptions(precision=4)
    if USE_WANDB:
        wandb_con.config.update(args)
        wandb_con.save("./*.py")
        wandb_con.save("./trainval/*.py")
        wandb_con.save("./dataloader/*.py")
        wandb_con.save("./auxiliary/*.py")
        wandb_con.save("./models/*.py")
        wandb_con.save("./models/ops/*.py")
        wandb_con.save("./searchables/*.py")
        args.wandb_con = wandb_con

    # FFCV loader here
    args.in_memory = True

    if args.distributed:
        setup_distributed(
            global_rank, local_rank, ip, args.port, world_size, args.cluster
        )

    train_queue, valid_queue, dl = get_ffcv_loaders(local_rank, args)
    criterion = CrossEntropyLabelSmooth(args.CLASSES, args.label_smoothing).to(
        f"cuda:{local_rank}"
    )
    for m in range(2, 9):
        args.model_num = m
        dist.barrier()
        not_loaded = True
        while not_loaded:
            try:
                if local_rank == 0:
                    artifact = wandb_con.use_artifact(f'europa1610/NASBenchFPGA/models-nasfpga-try1-random-all-jobid10204:v{m}', type='model')
                    artifact.download(args.save)
                dist.barrier()
                sd = torch.load(os.path.join(args.save, "f_model.pth"), map_location="cpu")
                not_loaded = False
            except ValueError:
                not_loaded = True
                raise ValueError('Artifact failed to successfully load... Retrying in 10 seconds...')
                time.sleep(10)

        args.design = sd['model_metadata']['architecture']#searchables.RandomSearchable()#EfficientNetB0Conf(d=1)#.RandomSearchable()
        logging.info('Design = \n%s', str(np.array(args.design)))
        platform, mode = 'fpga', 'train'
        model = Network(design=args.design, platform=platform, mode=mode)
        args.macs, args.params = profile_model((1, 3, 224, 224),
                                               design=args.design,
                                               platform=platform,
                                               mode=mode)
        model = model.to(memory_format=torch.channels_last)
        model = model.to(f"cuda:{local_rank}")

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model)
        optimizer, scheduler = create_optimizer(model, args.lr,
                                                args.weight_decay, args)
        dist.barrier()
        train_acc, train_loss, valid_t1, _, valid_loss = train_x_epochs(
            args.epochs,
            scheduler,
            dl,
            train_queue,
            valid_queue,
            model,
            criterion,
            optimizer,
            args.global_rank,
            args.local_rank,
            args.world_size,
            wandb_con,
            args,
        )
        logging.info("Model Acc: %f", valid_t1)


def create_optimizer(model, lr, weight_decay, args):
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
    optimizer = torch.optim.SGD(parameters, lr=1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs)

    return optimizer, scheduler


if __name__ == "__main__":
    start_time = time.time()
    main()
    duration = time.time() - start_time
    logging.info("Total eval time: %ds", duration)
