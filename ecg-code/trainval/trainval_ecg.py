import time
import torch
import numpy as np
import logging
import tqdm

from collections import Counter

# from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.cuda.amp import autocast
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

import auxiliary.utils as utils


def train(
    epoch,
    train_queue,
    valid_queue,
    model,
    criterion,
    optimizer,
    scaler,
    report_freq,
    fast,
    lr_peak_epoch,
    epochs,
    argslr,
    args,
):

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    model.train()
    iterator = train_queue
    for step, (input, target) in enumerate(iterator):
        b_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        input = input.reshape(-1, 1, 1000).to(f"cuda:{args.local_rank}")
        target = target.to(f"cuda:{args.local_rank}")
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        '''
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        '''
        batch_time.update(time.time() - b_start)
        prec1, prec5 = 0, 0
        n = input.size(0)
        objs.update(loss, n)
        top1.update(prec1, n)
        top5.update(prec5, n)
        if (step+1) % report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info(
                "TRAIN Step: %03d Objs: %e R1: %f\
                R5: %f Duration: %ds BTime: %.3fs",
                step,
                objs.avg,
                top1.avg,
                top5.avg,
                duration,
                batch_time.avg,
            )
        del loss, logits, target
    return 1.0, 1.0, scaler


def infer(valid_queue, valid_dataset, model, criterion, args, report_freq=100, fast=False):
    model.eval()
    all_logits_probs = []
    all_targets = []
    with torch.no_grad():
        # with autocast():
        for step, (input, target) in enumerate(valid_queue):
            input = input.reshape(-1, 1, 1000).to(f"cuda:{args.local_rank}")
            target = target.to(f"cuda:{args.local_rank}")
            # input = input.permute(0, 2, 1)
            logits = model(input)
            all_logits_probs.append(logits.cpu().data.numpy())
            all_targets.append(target)
    all_logits_prob = np.concatenate(all_logits_probs)
    all_logits = np.argmax(all_logits_prob, axis=1)

    final_pred = []
    final_gt = []
    pid_test = valid_dataset.pid
    for i_pid in np.unique(pid_test):
        tmp_pred = all_logits[pid_test==i_pid]
        tmp_gt = valid_dataset.label[pid_test==i_pid]
        final_pred.append(Counter(tmp_pred).most_common(1)[0][0])
        final_gt.append(Counter(tmp_gt).most_common(1)[0][0])
    ## classification report
    tmp_report = classification_report(final_gt, final_pred, output_dict=True)
    # print(confusion_matrix(final_gt, final_pred))
    f1_score = (tmp_report['0']['f1-score'] + tmp_report['1']['f1-score'] + tmp_report['2']['f1-score'] + tmp_report['3']['f1-score'])/4

    return f1_score, 0, 0


def train_x_epochs(
    _epochs,
    scheduler,
    dl_class_inst,
    train_queue,
    valid_queue,
    model,
    criterion,
    optimizer,
    global_rank,
    local_rank,
    world_size,
    wandb_con,
    args,
):
    from torch.cuda.amp import GradScaler

    train_sttime = time.time()
    scaler = GradScaler()
    best_acc_top1 = 0
    best_acc_top5 = 0
    valid_acc_top1, valid_acc_top5, valid_obj = None, None, None

    for epoch in range(_epochs):
        epoch_start = time.time()
        train_acc, train_obj, scaler = train(
            epoch,
            train_queue,
            valid_queue,
            model,
            criterion,
            optimizer,
            scaler,
            args.report_freq,
            args.fast,
            args.lr_peak_epoch,
            _epochs,
            args.lr,
            args,
        )
        # epoch_duration = time.time() - epoch_start
        # logging.info(
        #    "Epoch %d, Train_acc %f, Epoch time: %ds", epoch, train_acc, epoch_duration
        # )
        # validation
        if epoch >= 0:  # _epochs - 4:
            valid_acc_top1, valid_acc_top5, valid_obj = infer(
                valid_queue,
                dl_class_inst,
                model,
                criterion,
                args,
                args.report_freq,
                args.fast,
            )
            # logging.info('Epoch %d, Valid_acc_top1 %f, Valid_acc_top5 %f, Best_top1 %f, Best_top5 %f', epoch, valid_acc_top1, valid_acc_top5, best_acc_top1, best_acc_top5)
            avg_top1_val, avg_top5_val = valid_acc_top1, valid_acc_top5
            if wandb_con is not None:
                commit = True if epoch < _epochs - 1 else False
                wandb_con.log(
                    {
                        "F-1 Score": avg_top1_val,
                    },
                    commit=commit,
                )
                if epoch == _epochs - 1:
                    wandb_con.log(
                        {"Train Time": time.time() - train_sttime}, commit=True
                    )
            if avg_top5_val > best_acc_top5:
                best_acc_top5 = avg_top5_val
            if avg_top1_val > best_acc_top1:
                best_acc_top1 = avg_top1_val
            logging.info(
                "Epoch %d, F-1 Score %f, Best F-1 %f",
                epoch,
                avg_top1_val,
                best_acc_top1,
            )
    train_endtime = time.time()
    return [
        train_acc,
        train_obj,
        best_acc_top1,
        best_acc_top5,
        valid_obj,
        - train_sttime + train_endtime,
    ]
