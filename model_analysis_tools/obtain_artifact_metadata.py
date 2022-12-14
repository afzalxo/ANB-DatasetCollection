import os
import wandb
import torch
import csv


def main():
    wandb_con = wandb.init(project='NASBenchFPGA', entity='europa1610', name='removeme')
    finished = False
    version = 0
    job_ids = [10222, 10225, 10232, 10221, 10218, 10226, 10217]
    job_dict = {}
    for jid in job_ids:
        metadata_dict = {}
        finished = False
        while not finished:
            try:
                artifact = wandb_con.use_artifact(f'europa1610/NASBenchFPGA/models-nasfpga-try1-random-all-jobid{jid}:v{version}', type='model')
                model_dir = artifact.download()
                finished = False
                version += 1
                sd = torch.load(os.path.join(model_dir, "f_model.pth"), map_location="cpu")
                model_metadata = sd['model_metadata']
                metadata_dict[version] = model_metadata
                print('==='*10)
                print(version)
                print(sd['model_metadata']['macs'])
                print(sd['model_metadata']['params'])
            except:
                print('Finished loading model metadata...')
                finished = True
                version = 0
        job_dict[jid] = metadata_dict

    with open('./junk.csv', 'a+') as fh:
        writer = csv.writer(fh)
        for k, v in job_dict.items():
            for key, val in v.items():
                writer.writerow([k, key, val['best_acc_top1'], val['best_acc_top5'], val['macs'], val['params'], val['train_time']])


if __name__ == '__main__':
    main()
