import os
import wandb
import torch
import csv
import json
import time
import numpy as np
import pandas as pd
import argparse


def csv_to_json(csv_path):
    log_dir = "dataset_log/"
    log_dir = os.path.join(
        log_dir, "{}-{}".format("csv_to_json", time.strftime("%Y%m%d-%H%M%S"))
    )
    os.makedirs(os.path.join(log_dir, "jsons"))

    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            model_grank = int(row[0])
            block = row[8:15]
            exps = row[15:22]
            ker = row[22:29]
            layers = row[29:]
            arch_json, metrics_json, model_info_json = dict(), dict(), dict()
            with open(
                os.path.join(log_dir, "jsons", f"result_{model_grank}.json"), "w"
            ) as js:
                for i in range(len(block)):
                    arch_json[f"block{i}_type"] = block[i]
                    arch_json[f"block{i}_k"] = int(ker[i])
                    arch_json[f"block{i}_e"] = int(exps[i])
                    arch_json[f"block{i}_l"] = int(layers[i])
                metrics_json["val_top1"] = float(row[3])
                metrics_json["val_top5"] = float(row[4])
                metrics_json["train_time"] = float(row[7])
                model_info_json["macs"] = float(row[5])
                model_info_json["params"] = float(row[6])
                config_dict = {
                    "architecture": arch_json,
                    "metrics": metrics_json,
                    "model_info": model_info_json,
                }
                json.dump(config_dict, js)


def csv_to_json_conditional(csv_path):
    log_dir = "dataset_log/"
    log_dir = os.path.join(
        log_dir, "{}-{}".format("csv_to_json_conditional", time.strftime("%Y%m%d-%H%M%S"))
    )
    os.makedirs(os.path.join(log_dir, "jsons"))
    model_counter = 0
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            block = row[8:15]
            if 'FMB' in block:
                continue
            exps = row[15:22]
            ker = row[22:29]
            layers = row[29:]
            print(model_counter, block, exps, ker, layers)
            arch_json, metrics_json, model_info_json = dict(), dict(), dict()
            with open(
                os.path.join(log_dir, "jsons", f"result_{model_counter}.json"), "w"
            ) as js:
                for i in range(len(block)):
                    arch_json[f"block{i}_type"] = block[i]
                    arch_json[f"block{i}_k"] = int(ker[i])
                    arch_json[f"block{i}_e"] = int(exps[i])
                    arch_json[f"block{i}_l"] = int(layers[i])
                metrics_json["val_top1"] = float(row[3])
                metrics_json["val_top5"] = float(row[4])
                metrics_json["train_time"] = float(row[7])
                model_info_json["macs"] = float(row[5])
                model_info_json["params"] = float(row[6])
                config_dict = {
                    "architecture": arch_json,
                    "metrics": metrics_json,
                    "model_info": model_info_json,
                }
                json.dump(config_dict, js)
                model_counter += 1


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to csv")
    args = parser.parse_args()
    csv_to_json_conditional(args.csv_path)


if __name__ == "__main__":
    main()