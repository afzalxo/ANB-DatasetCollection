import os
import sys
import subprocess
import time
import numpy as np
import torch
import gc
import logging
import argparse
import configparser
import torch.utils
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import random
import warnings
import pickle
from thop import profile

import auxiliary.utils as utils
from trainval.trainval_satellite import train_x_epochs
from dataloader.satellite_dataloaders import SatelliteFFCVDataLoader
from auxiliary.utils import CrossEntropyLabelSmooth, create_optimizer, setup_for_distributed
from models.accelbenchnet_1d import AccelNet1d as Network
from searchables import searchables

warnings.filterwarnings("ignore")


def setup_distributed(rank, local_rank, address, port, world_size, cluster):
    os.environ["MASTER_ADDR"] = address
    os.environ["MASTER_PORT"] = port
    # os.environ['NCCL_IB_DISABLE'] = '1'
    # os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    print("Setting up dist training rank %d" % rank)
    if cluster == "local":
        home_dir = os.getcwd()
        file_path = os.path.join(home_dir, "dist_pg")
        init_method = f"file://{file_path}"
    elif cluster == "tacc":
        init_method = "env://"

    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size
    )
    torch.cuda.set_device(local_rank)


def profile_model(input_size, design, activation_fn, mode, local_rank):
    model_temp = Network(design=design, activation_fn=activation_fn, mode=mode)
    input = torch.randn(input_size)  # .to(f'cuda:{local_rank}')
    macs, params = profile(model_temp, inputs=(input,), verbose=False)
    macs, params = macs / 1000000, params / 1000000
    del model_temp
    logging.info("MFLOPS %f, MPARAMS: %f", macs, params)
    torch.cuda.empty_cache()
    gc.collect()
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
    parser.add_argument("--wandb-api-key", type=str)
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--seed", type=int, default=2)
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
    args.activation_fn = config["model"]["activation_fn"]
    # dataloaders
    args.train_dataset = config["dataloader"]["train_dataset"]
    args.val_dataset = config["dataloader"]["val_dataset"]
    args.num_workers = config["dataloader"].getint("num_workers")
    args.in_memory = config["dataloader"].getboolean("in_memory")
    # trainval
    if not hasattr(args, "epochs"):
        args.epochs = config["trainval"].getint("epochs")
    if not hasattr(args, "train_batch_size"):
        args.train_batch_size = config["trainval"].getint("train_batch_size")
    args.val_batch_size = config["trainval"].getint("val_batch_size")
    args.val_resolution = config["trainval"].getint("val_resolution")
    if not hasattr(args, "min_res"):
        args.min_res = config["trainval"].getint("min_res")
    if not hasattr(args, "max_res"):
        args.max_res = config["trainval"].getint("max_res")
    if not hasattr(args, "start_ramp"):
        args.start_ramp = config["trainval"].getint("start_ramp")
    if not hasattr(args, "end_ramp"):
        args.end_ramp = config["trainval"].getint("end_ramp")
    if not hasattr(args, "seed"):
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

    if args.cluster == "tacc":
        global_rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        iplist = os.environ["SLURM_JOB_NODELIST"]
        job_id = int(os.environ["SLURM_JOB_ID"])
        ip = subprocess.getoutput(f"scontrol show hostname {iplist} | head -n1")
        # setup_for_distributed(global_rank == 0)
    elif args.cluster == "local":
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        job_id = 10001  # int(os.getpid())
        ip = "127.0.0.1"
        # setup_for_distributed(global_rank == 0)
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

    args.use_wandb = True if global_rank == 0 and args.use_wandb else False

    args.save = "{}exp-{}-{}-{}".format(
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
    # if args.distributed:
    #    setup_distributed(
    #        global_rank, local_rank, ip, args.port, world_size, args.cluster
    #    )
    #    dist.barrier()
    torch.set_printoptions(precision=4)
    np.set_printoptions(precision=4)
    # if global_rank == 0:
    #    utils.create_exp_dir(
    #        args.save, scripts_to_save=None  # glob.glob("**/*.py", recursive=True)
    #    )
    if args.distributed:
        dist.barrier()
    # if args.global_rank == 0:
    log_format = f"%(asctime)s - Rank {args.global_rank} - %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
        # fh = logging.FileHandler(os.path.join(args.save, f"log_rank{args.global_rank}.txt"))
        #fh.setFormatter(logging.Formatter(log_format))
        #logging.getLogger().addHandler(fh)

    wandb_con = None
    wandb_metadata_dir = None
    if args.use_wandb and global_rank == 0:
        wandb_metadata_dir = args.save
        import wandb
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        os.environ["WANDB_ENTITY"] = args.wandb_entity
        os.environ["WANDB_PROJECT"] = args.wandb_project
        wandb_con = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.note + f"_{args.job_id}",
            settings=wandb.Settings(code_dir="."),
            dir=wandb_metadata_dir,
        )
        wandb_con.config.update(args)
        logging.info("Saving py files to wandb...")
        wandb_con.save("./*.py")
        wandb_con.save("./trainval/*.py")
        wandb_con.save("./dataloader/*.py")
        wandb_con.save("./auxiliary/*.py")
        wandb_con.save("./models/*.py")
        wandb_con.save("./models/ops/*.py")
        wandb_con.save("./searchables/*.py")
        logging.info("Saved py files to wandb...")
        args.wandb_con = wandb_con
    else:
        wandb_con = None

    logging.info("args = %s", args)

    # FFCV loader here
    args.in_memory = True
    num_models = 40
    job_idx = 0
    job_offset = job_idx * num_models * 4
    m = job_offset + args.local_rank * num_models
    starting_index = m

    dl = SatelliteFFCVDataLoader(local_rank, args.num_workers)
    train_queue = dl.create_train_loader(args.train_dataset, args.train_batch_size)
    valid_queue = dl.create_val_loader(args.val_dataset, args.val_batch_size)

    # Load archs here
    archs = pickle.load(open("models_rand_satellite_shallow_160.tor", "rb"))
    archs = archs["arch_dict"]

    criterion = CrossEntropyLabelSmooth(args.CLASSES, args.label_smoothing).to(
        f"cuda:{local_rank}"
    )
    while m < starting_index + num_models:# len(archs):
        args.model_num = m
        if args.distributed:
            dist.barrier()
        args.design = archs[m][0]
        logging.info(
            "Job ID: %d, Model Number: %d, Design: \n%s",
            args.job_id,
            args.model_num,
            np.array(args.design),
        )
        activation_fn, mode = args.activation_fn, "train"
        args.macs, args.params = archs[m][2], archs[m][1]
        '''
        if args.global_rank == 0:
            args.macs, args.params = profile_model(
                (1, 1, 46),
                design=args.design,
                activation_fn=activation_fn,
                mode=mode,
                local_rank=args.local_rank,
            )
        if args.distributed:
            dist.barrier()
        '''
        model = Network(design=args.design, activation_fn=activation_fn, mode=mode)
        # model = model.to(memory_format=torch.channels_last)
        # model = MLPBaseline(46, 40, 30, 24)
        model = model.to(f"cuda:{local_rank}")

        if args.distributed:
            dist.barrier()
            model = torch.nn.parallel.DistributedDataParallel(model)
            dist.barrier()
        _, scheduler = create_optimizer(model, args.lr, args.weight_decay, args.epochs)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_acc, train_loss, valid_t1, _, valid_loss, train_time = train_x_epochs(
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
        logging.info(
            "Job ID: %d, Model Number: %f, Train Time: %f, Model Acc: %f",
            args.job_id,
            args.model_num,
            train_time,
            valid_t1.detach().cpu().item(),
        )
        archs[m].append(train_time)
        archs[m].append(valid_t1.detach().cpu().item())
        with open(f'models_rand_satellite_accs_shallow_160_proxified_start{starting_index}_num{num_models}_seed{args.seed}.tor', 'wb') as fh:
            arch_d = {"arch_dict": archs}
            pickle.dump(arch_d, fh)
        if args.distributed:
            dist.barrier()
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()
        m += 1


if __name__ == "__main__":
    start_time = time.time()
    main()
    duration = time.time() - start_time
    logging.info("Total eval time: %ds", duration)
