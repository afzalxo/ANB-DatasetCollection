import os
import sys
import subprocess
import time
import datetime
import glob
import numpy as np
import torch
import csv
import gc
import logging
import argparse
import configparser
import torch.utils
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import random
import warnings
import pandas as pd
from thop import profile

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu


import auxiliary.utils as utils
from trainval.trainval_tpu import train_x_epochs_tpu
from trainval.trainval_tpu import throughput_tpu
from dataloader.torchvision_dataloader import build_torchvision_loader_tpu
from auxiliary.utils import CrossEntropyLabelSmooth
from auxiliary.utils import create_optimizer
from models.accelbenchnet import AccelNet as Network
from searchables import searchables

warnings.filterwarnings("ignore")


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
    # print(f"MFLOPS: {macs}, MPARAMS: {params}")
    logging.info('MFLOPS %f, MPARAMS: %f', macs, params)
    gc.collect()
    return macs, params


def map_fn(index, args):
    local_rank = global_rank = args.local_rank = args.global_rank = index#xm.get_ordinal()
    args.world_size = world_size = 8#xm.xrt_world_size()
    DEV_NAME = 'TPUv3'
    args.use_wandb = True if global_rank == 0 and args.use_wandb else False

    args.save = "{}exp-throughput-{}-{}-{}-{}".format(
        args.save, DEV_NAME, args.job_id, args.note, time.strftime("%Y%m%d-%H%M%S")
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
        log_format = f"%(asctime)s - Rank {args.global_rank} - %(message)s"
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format=log_format,
            datefmt="%m/%d %I:%M:%S %p",
        )
        fh = logging.FileHandler(os.path.join(args.save, f"log_rank{args.global_rank}.txt"))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

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
            group='throughput-TPUv3-dataset',
        )
        wandb_art = wandb.Artifact(name=f"throughput-tpu-code-jobid{args.job_id}", type="code")
        wandb_art.add_dir(os.path.join(args.save, "scripts"))
        wandb_con.log_artifact(wandb_art)
        wandb_con.config.update(args)
        logging.info('Saving py files to wandb...')
        wandb_con.save("./*.py")
        wandb_con.save("./trainval/*.py")
        wandb_con.save("./dataloader/*.py")
        wandb_con.save("./auxiliary/*.py")
        wandb_con.save("./models/*.py")
        wandb_con.save("./models/ops/*.py")
        wandb_con.save("./searchables/*.py")
        logging.info('Saved py files to wandb...')
        args.wandb_con = wandb_con
    else:
        wandb_con = None
        wandb_art = None

    logging.info("args = %s", args)
    os.makedirs(os.path.join(args.save, "csv"))

    device = xm.xla_device()
    args.in_memory = True
    m = 0
    version = 0
    missing_counter = 0
    # xm.rendezvous('Finishing...')
    # return

    job_ids = [10237, 10239, 10240, 10241, 10265, 10266, 10268, 10269, 10270, 10272,
               10277, 10279, 10280, 10286, 10298, 10325, 10338, 10342, 10344, 10345,
               10347, 10348, 10350, 2358639, 10355, 10354, 10356, 10357, 10359, 10360,
               10361, 10362, 10363, 10372, 10373, 10376, 10377, 10379, 10380, 10381,
               10382, 10383, 10384, 10386, 10387, 10402, 10403, 10404, 10406, 10407,
               10432, 10433, 10435, 10436, 10457, 10458, 10459, 10461, 10463, 10465,
               10466, 10467, 10468]
    job_ids = [10474, 10475, 10476, 10477, 10478, 10479, 10000, 10481, 10001, 10523, 10522, 10525]

    table_rows = []

    train_queue, valid_queue = build_torchvision_loader_tpu(args)
    train_queue = pl.MpDeviceLoader(train_queue, device)
    valid_queue = pl.MpDeviceLoader(valid_queue, device)
    args.writer = None
    platform, mode = "fpga", "train"
    if xm.is_master_ordinal():
        import torch_xla.test.test_utils as test_utils
        args.writer = test_utils.get_summary_writer(args.save)
    with open(os.path.join(args.save, "csv", f"throughput_{DEV_NAME}.csv"), "a+") as fh:
        writer = csv.writer(fh)
        for jid in job_ids:
            finished = False
            while not finished:
                try:
                    finished = False
                    artifact = wandb_con.use_artifact(
                            f"europa1610/NASBenchFPGA/models-random-jobid{jid}-model{version}:v0",
                            type="model",
                    )
                    md = artifact.metadata["model_metadata"]
                    args.design = md["architecture"]
                    blocks = np.array(args.design)[:, 0]
                    print(f'Job ID: {jid}, Version {version}')
                    if 'FMB' in blocks:
                        version += 1
                        continue
                    if len(args.design[0]) == 7:
                        for p in range(7):
                            args.design[p].append(False)
                    # args.design = (searchables.RandomSearchable())
                    logging.info(
                        "Job ID: %d, Model Number: %d, MACS: %f, MParams %f, Design: \n%s",
                        jid,
                        m,
                        md['macs'],
                        md['params'],
                        np.array(args.design),
                    )
                    model = Network(design=args.design, platform=platform, mode=mode)
                    # model = model.to(memory_format=torch.channels_last)
                    model = model.to(device)
                    mean_thr, std_thr = throughput_tpu(train_queue, model, args, 0)
                    m += 1
                    row = [
                        m,
                        jid,
                        version,
                        float(md["best_acc_top1"]),
                        float(md["best_acc_top5"]),
                        float(md["macs"]),
                        float(md["params"]),
                        abs(float(md["train_time"])),
                        mean_thr,
                        std_thr,
                    ]
                    table_rows.append(row)
                    writer.writerow(row)
                    fh.flush()
                    version += 1
                    m += 1
                    missing_counter = 0
                    del model
                except KeyboardInterrupt:
                    print('Abort...')
                    args.wandb_con.finish()
                    exit(0)
                except:
                    print(
                        f"Cant find job {jid}, model {version}, missing counter {missing_counter}..."
                    )
                    version += 1
                    missing_counter += 1
                    if missing_counter == 10:
                        print(f"Finished throughput measurements for job {jid}...")
                        finished = True
                        version = 0
                        missing_counter = 0
    columns = [
        "Model Num",
        "Job ID",
        "Model Rank",
        "Top-1",
        "Top-5",
        "MACs",
        "MParams",
        "Train Time",
        "Throughput Mean",
        "Throughput Std",
    ]
    df = pd.DataFrame(table_rows, columns=columns)
    print(df.head())
    table = wandb.Table(dataframe=df)
    artifact = wandb.Artifact(f"throughput_dataset_{DEV_NAME}", "dataset")
    artifact.add(table, f"throughput_dataset_{DEV_NAME}")
    args.wandb_con.log_artifact(artifact)

    print(f'Returning from Rank {args.local_rank}')
    xm.rendezvous('checking_out')

#### Map Function

def main():
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
        job_id = int(os.environ["SLURM_JOB_ID"])
        ip = subprocess.getoutput(f"scontrol show hostname {iplist} | head -n1")
        setup_for_distributed(global_rank == 0)
    elif args.distributed and args.cluster == "local":
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        job_id = 10001# int(os.getpid())
        ip = "127.0.0.1"
        setup_for_distributed(global_rank == 0)
    elif args.distributed and args.cluster == "tpu":
        os.environ['XLA_USE_BF16'] = '1' # Enable bfloat16
        job_id = 20000
        ip = "127.0.0.1"
    else:
        global_rank = 0
        job_id = 0
        ip = "localhost"

    args.ip = ip
    args.job_id = job_id

    flags = args
    xmp.spawn(map_fn, args=(flags,), nprocs=1, start_method='fork')
    print('Finished...')
    exit(0)


    
if __name__ == "__main__":
    start_time = time.time()
    main()
    duration = time.time() - start_time
    logging.info("Total eval time: %ds", duration)
