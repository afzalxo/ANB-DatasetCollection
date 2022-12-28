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
        log_dir, "{}-{}".format('exp', time.strftime("%Y%m%d-%H%M%S"))
    )
    os.makedirs(os.path.join(log_dir, 'csv'))
    os.makedirs(os.path.join(log_dir, 'jsons'))

    wandb_con = wandb.init(project='NASBenchFPGA', entity='europa1610', name='models_metadata', group='surrogate_dataset')
    finished = False
    models_done = 0
    version = 0
    '''
    job_ids = [10237, 10239, 10240, 10241, 10265, 10266, 10268, 10269, 10270, 10272,
               10277, 10279, 10280, 10286, 10298, 10325, 10338, 10342, 10344, 10345,
               10347, 10348, 10350, 2358639, 10355, 10354, 10356, 10357, 10359, 10360,
               10361, 10362, 10363, 10372, 10373, 10376, 10377, 10379, 10380, 10381,
               10382, 10383, 10384, 10386, 10387, 10402, 10403, 10404, 10406, 10407,
               10432, 10433, 10435, 10436, 10457, 10458, 10459, 10461, 10463]
    '''
    job_ids = [10432, 10457, 10458, 10459, 10461, 10463]
    job_dict = {}
    table_rows = []
    with open(os.path.join(log_dir, 'csv', 'results.csv'), 'a+') as fh:
        writer = csv.writer(fh)
        for jid in job_ids:
            metadata_dict = {}
            finished = False
            while not finished:
                try:
                    artifact = wandb_con.use_artifact(f'europa1610/NASBenchFPGA/models-random-jobid{jid}-model{version}:v0', type='model')
                    # model_dir = artifact.download()
                    finished = False
                    # sd = torch.load(os.path.join(model_dir, "f_model.pth"), map_location="cpu")
                    md = artifact.metadata['model_metadata']
                    metadata_dict[version] = md
                    print('==='*10)
                    print(f'Model Number {models_done}, Job {jid}, Model {version}...')

                    block = list(np.array(md['architecture'])[:, 0])
                    exps = list(np.array(md['architecture'])[:, 1])
                    ker = list(np.array(md['architecture'])[:, 2])
                    layers = list(np.array(md['architecture'])[:, 6])
                    row = []
                    row.append([models_done, jid, version, float(md['best_acc_top1']), float(md['best_acc_top5']), float(md['macs']), float(md['params']), -1*float(md['train_time'])])
                    row.append([b for b in block])
                    row.append([int(b) for b in exps])
                    row.append([int(b) for b in ker])
                    row.append([int(b) for b in layers])
                    row = [item for sublist in row for item in sublist]
                    # print(row)
                    table_rows.append(row)
                    writer.writerow(row)

                    # Dumping to JSON
                    arch_json = {}
                    metrics_json = {}
                    model_info_json = {}
                    with open(os.path.join(log_dir, 'jsons', f'result_{models_done}.json'), 'w') as js:
                        for i in range(len(block)):
                            arch_json[f'block{i}_type'] = block[i]
                            arch_json[f'block{i}_k'] = int(ker[i])
                            arch_json[f'block{i}_e'] = int(exps[i])
                            arch_json[f'block{i}_l'] = int(layers[i])
                        metrics_json['val_top1'] = float(md['best_acc_top1'])
                        metrics_json['val_top5'] = float(md['best_acc_top5'])
                        metrics_json['train_time'] = float(md['train_time'])
                        model_info_json['macs'] = float(md['macs'])
                        model_info_json['params'] = float(md['params'])
                        config_dict = {'architecture': arch_json, 'metrics': metrics_json, 'model_info': model_info_json}
                        json.dump(config_dict, js)

                    version += 1
                    models_done += 1
                    missing_counter = 0
                    # os.remove(os.path.join(model_dir, 'f_model.pth'))
                except:
                    print(f'Cant find job {jid}, model {version}, missing counter {missing_counter}...')
                    version += 1
                    missing_counter += 1
                    if missing_counter == 10:
                        print(f'Finished loading model metadata for job {jid}...')
                        finished = True
                        version = 0
                        missing_counter = 0
            job_dict[jid] = metadata_dict

    columns = ['Model Num', 'Job ID', 'Model Rank', 'Top-1', 'Top-5', 'MACs', 'MParams', 'Train Time']
    for i in range(7):
        columns.append(f'Block {i}')
    for i in range(7):
        columns.append(f'Expansion {i}')
    for i in range(7):
        columns.append(f'Kernel {i}')
    for i in range(7):
        columns.append(f'Layers {i}')
    df = pd.DataFrame(table_rows, columns=columns)
    print(df.head())
    table = wandb.Table(dataframe=df)
    artifact = wandb.Artifact('surrogate_dataset', 'dataset')
    artifact.add(table, 'surrogate_dataset')
    wandb_con.log_artifact(artifact)


if __name__ == '__main__':
    main()
