import os
import wandb
import torch
import csv
import json
import time
import numpy as np
import pandas as pd


def main():
    log_dir = 'dataset_log/'
    log_dir = os.path.join(
        log_dir, "{}-{}".format('multiseed', time.strftime("%Y%m%d-%H%M%S"))
    )
    os.makedirs(os.path.join(log_dir, 'csv'))

    wandb_con = wandb.init(project='NASBenchFPGA', entity='europa1610', name='models_metadata', group='surrogate_dataset')
    finished = False
    models_done = 0
    version = 0
    job_ids = [10401]#, 10399, 10400, 10401]
    table_rows = []
    with open(os.path.join(log_dir, 'csv', 'results.csv'), 'a+') as fh:
        writer = csv.writer(fh)
        for jid in job_ids:
            for seed in range(3):
                finished = False
                while not finished:
                    try:
                        artifact = wandb_con.use_artifact(f'europa1610/NASBenchFPGA/models-random-jobid{jid}-model{version}:v{seed}', type='model')
                        finished = False
                        md = artifact.metadata['model_metadata']
                        print('==='*10)
                        print(f'Model Number {models_done}, Job {jid}, Model {version}, Seed {seed}...')

                        block = list(np.array(md['architecture'])[:, 0])
                        exps = list(np.array(md['architecture'])[:, 1])
                        ker = list(np.array(md['architecture'])[:, 2])
                        layers = list(np.array(md['architecture'])[:, 6])
                        row = []
                        row.append([models_done, jid, version, int(md['model_seed']), float(md['best_acc_top1']), float(md['best_acc_top5']), float(md['macs']), float(md['params']), -1*float(md['train_time'])])
                        row.append([b for b in block])
                        row.append([int(b) for b in exps])
                        row.append([int(b) for b in ker])
                        row.append([int(b) for b in layers])
                        row = [item for sublist in row for item in sublist]
                        # print(row)
                        table_rows.append(row)
                        writer.writerow(row)

                        version += 1
                        models_done += 1
                        missing_counter = 0
                        # os.remove(os.path.join(model_dir, 'f_model.pth'))
                    except:
                        print(f'Cant find job {jid}, model {version}, seed {seed}...')
                        print(f'Finished loading model metadata for job {jid}, Seed {seed}...')
                        version += 1
                        missing_counter += 1
                        if missing_counter == 5:
                            finished = True
                            version = 0
                            missing_counter = 0
                        

if __name__ == '__main__':
    main()
