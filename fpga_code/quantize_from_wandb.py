import os
import sys
import argparse
import json
import logging
import csv
import gc
import torch
import numpy as np
from collections import OrderedDict
from itertools import cycle
import wandb
import time
import pandas as pd

from pytorch_nndct.apis import torch_quantizer, dump_xmodel

from run_on_fpga import execute_on_fpga

DIVIDER = "-----------------------------------------"


def quantize(dataset_path, batchsize, args):
    target = "zcu102"
    args.world_size = 1
    args.local_rank = 0
    args.distributed = False
    log_dir = "dataset_log/"
    log_dir = os.path.join(
        # log_dir, "{}-{}".format('compile-exp-vck190-more-more', time.strftime("%Y%m%d-%H%M%S"))
        log_dir,
        "{}-{}".format("compile-exp-ablation1-zcu102", time.strftime("%Y%m%d-%H%M%S")),
    )
    os.makedirs(os.path.join(log_dir, "quant_csv"))
    os.makedirs(os.path.join(log_dir, "quant_model"))
    os.makedirs(os.path.join(log_dir, "compiled_model"))

    # wandb_con = wandb.init(project='NASBenchFPGA', entity='europa1610', name='junk-models_quantacc-vck-MBonly-more-more', group='surrogate_dataset-quantacc')
    # wandb_con.save('./quantize_from_wandb.py')

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    """
    job_ids = [10265, 10269, 
               10277, 10298, 10325, 10342,
               10350, 2358639, 10355, 10354, 10356, 10357, 10359, 10360,
               10361, 10362, 10363, 10372, 10373, 10376, 10377, 10379, 10380, 10381,
               10382, 10383, 10384, 10386, 10387, 10402, 10403, 10404, 10406, 10407]
    """
    # job_ids = [10432, 10433, 10435, 10436, 10457, 10458, 10459, 10461, 10463, 10465, 10466, 10467, 10468]
    # job_ids = [10461, 10463, 10465, 10466, 10467, 10468]
    job_ids = [
        10474,
        10475,
        10476,
        10477,
        10478,
        10479,
        10000,
        10481,
        10001,
        10523,
        10522,
        10525,
    ]
    job_ids = [13555]
    # job_ids = [10527, 10528, 10529, 10530, 10531, 10532, 11996, 11998, 11999, 12000, 12001, 12003, 12004]
    quant_model = os.path.join(log_dir, "quant_model")
    compiled_dir = os.path.join(log_dir, "compiled_model")

    # run = wandb.init()
    # artifact = run.use_artifact(f'europa1610/NASBenchFPGA/train-code-jobid10478:v0', type='code')
    # code_dir = artifact.download()
    from models.accelbenchnet import AccelNet as Network

    code_dir = "./artifacts/train-code-jobid10478:v0/"
    sys.path.append(code_dir)
    print(code_dir)

    # from models import efficientnet as Network
    from dataloader import torchvision_dataloader
    from trainval.trainval import infer_tv

    # use GPU if available
    if torch.cuda.device_count() > 0:
        print("Selecting device 0..")
        device = torch.device("cuda:0")
    else:
        print("No CUDA devices available..selecting CPU")
        device = torch.device("cpu")

    version = 0
    models_done = 0
    missing_counter = 0
    table_rows = []
    subset_len = args.subset_len
    with open(os.path.join(log_dir, "quant_csv", "quant_results.csv"), "a+") as fh:
        writer = csv.writer(fh)
        for jid in job_ids:
            finished = False
            while not finished:
                try:
                    ####
                    api = wandb.Api()
                    # artifact = api.artifact(f'europa1610/NASBenchFPGA/models-random-jobid{jid}-model{version}:v0', type='model')
                    artifact = api.artifact(
                        f"europa1610/NASBenchFPGA/models-ablation1-exact-jobid{jid}-model{version}:v0",
                        type="model",
                    )
                    finished = False
                except KeyboardInterrupt:
                    print("Abort...")
                    exit(1)
                except:
                    print(
                        f"Cant find job {jid}, model {version}, missing counter {missing_counter}..."
                    )
                    version += 1
                    missing_counter += 1
                    if missing_counter == 10:
                        print(f"Finished loading model metadata for job {jid}...")
                        finished = True
                        version = 0
                        missing_counter = 0
                    continue

                ### Use artifact pth file to quantize, eval quantized model, generate xmodel, and send to FPGA
                archi = artifact.metadata["model_metadata"]["architecture"]
                if any("FMB" in blockspecs for blockspecs in archi):
                    raise ValueError("Architecture contains FMB. Skipping...")
                    version += 1
                    continue
                model_dir = artifact.download()
                sd = torch.load(
                    os.path.join(model_dir, "f_model.pth"), map_location="cpu"
                )
                design = sd["model_metadata"]["architecture"]
                # model = Network.efficientnet_b0(design=design, platform="fpga", mode="val").to(device)
                model = Network(design=design, activation_fn="relu", mode="val").to(
                    device
                )
                state_dict = sd["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v

                print("Model Info:")
                print(json.dumps(sd["model_metadata"], default=str))
                model.load_state_dict(new_state_dict)
                args.num_workers = 8
                args.data_path = args.dataset_path

                for quant_mode in ["calib", "test"]:
                    if quant_mode == "test":
                        batchsize = 1
                        args.subset_len = 100
                    else:
                        batchsize = args.batchsize
                        args.subset_len = subset_len
                    args.val_batch_size = batchsize
                    rand_in = torch.randn([1, 3, 224, 224])
                    quantizer = torch_quantizer(
                        quant_mode, model, (rand_in), output_dir=quant_model
                    )
                    quantized_model = quantizer.quant_model
                    # data loader
                    test_loader, _ = torchvision_dataloader.build_torchvision_loader(
                        args
                    )
                    acc_top1, acc_top5, _ = infer_tv(
                        test_loader,
                        quantized_model,
                        None,
                        args,
                        report_freq=1000,
                        lr_tta=False,
                        fast=False,
                    )

                    # export config
                    if quant_mode == "calib":
                        print(
                            f"Quant Calibration: Model Number {models_done}, Job ID {jid}, Model {version}, Top-1 {acc_top1}, Top5 {acc_top5}"
                        )
                        quantizer.export_quant_config()
                    if quant_mode == "test":
                        print(
                            f"Quant Test: Model Number {models_done}, Job ID {jid}, Model {version}, Top-1 {acc_top1}, Top5 {acc_top5}"
                        )
                        quantizer.export_xmodel(
                            deploy_check=True, output_dir=quant_model
                        )
                import subprocess

                shellscript = subprocess.Popen(
                    [
                        "bash",
                        "./compile/compile.sh",
                        f"{jid}",
                        f"{version}",
                        f"{log_dir}",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = shellscript.communicate()
                # sts = os.waitpid(shellscript.pid, 0)
                # print('####'*10)
                # print(stdout)
                print("====" * 10)
                # print(stderr)
                """ # TODO: Do this manually later
                compiled_model_name = f'model_{jid}_{version}_{target}.xmodel'
                compiled_model_loc = os.path.join(compiled_dir, compiled_model_name)
                pro = subprocess.Popen(['scp', compiled_model_loc, 'root@143.89.79.74:/home/root/nasbench_infer/models_to_eval/'])
                os.waitpid(pro.pid, 0)
                print(f'File {compiled_model_name} transferred to ZCU102...')
                """

                ### Use artifact metadata to write model specs to csv
                md = artifact.metadata["model_metadata"]
                print("===" * 10)
                print(f"Model Number {models_done}, Job {jid}, Model {version}...")
                exps = list(np.array(md["architecture"])[:, 1])
                ker = list(np.array(md["architecture"])[:, 2])
                layers = list(np.array(md["architecture"])[:, 6])
                se = list(np.array(md["architecture"])[:, 7])

                row = [
                    models_done,
                    jid,
                    version,
                    float(md["best_acc_top1"]),
                    float(md["best_acc_top5"]),
                    float(md["macs"]),
                    float(md["params"]),
                    abs(float(md["train_time"])),
                    float(acc_top1),
                    float(acc_top5),
                ]
                row.extend([int(b) for b in exps])
                row.extend([int(b) for b in ker])
                row.extend([int(b) for b in layers])
                row.extend([b for b in se])
                table_rows.append(row)
                writer.writerow(row)
                fh.flush()

                version += 1
                models_done += 1
                missing_counter = 0
                os.remove(os.path.join(model_dir, "f_model.pth"))
                del model, quantized_model, test_loader
                torch.cuda.empty_cache()
                gc.collect()

    """
    del model, quantized_model, test_loader 
    xmodel_fpath = os.path.join(compiled_xmodel, 'EfficientNet_u50.xmodel')
    threads = 4
    execute_on_fpga(dataset_path, threads, xmodel_fpath)
    """
    """
    columns = ['Model Num', 'Job ID', 'Model Rank', 'Top-1', 'Top-5', 'MACs', 'MParams', 'Train Time', 'Quant Top-1', 'Quant Top-5']
    df = pd.DataFrame(table_rows, columns=columns)
    table = wandb.Table(dataframe=df)
    artifact = wandb.Artifact('surrogate_dataset-quantacc', 'dataset')
    artifact.add(table, 'surrogate_dataset-quantacc')
    wandb_con.log_artifact(artifact)
    """

    return


def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_path",
        type=str,
        default="/workspace/imagenet",
        help="Path to ImageNet subset",
    )
    ap.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=32,
        help="Testing batchsize - must be an integer. Default is 50",
    )
    ap.add_argument(
        "--subset_len",
        type=int,
        default=150,
        help="Number of images used for calibration",
    )
    args = ap.parse_args()

    print("\n" + DIVIDER)
    print("PyTorch version : ", torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(" Command line options:")
    print("--dataset_path : ", args.dataset_path)
    print("--batchsize    : ", args.batchsize)
    print(DIVIDER)

    quantize(args.dataset_path, args.batchsize, args)

    return


if __name__ == "__main__":
    run_main()
