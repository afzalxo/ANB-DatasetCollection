import time
import torch
import numpy as np
import logging
import tqdm

# from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.cuda.amp import autocast
from sklearn import metrics

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
    from torch.cuda.amp import autocast

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    model.train()
    lr_start, lr_end = utils.get_cyclic_lr(
        epoch, argslr, epochs, lr_peak_epoch
    ), utils.get_cyclic_lr(epoch + 1, argslr, epochs, lr_peak_epoch)
    iters = len(train_queue)
    lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])
    rank_finished = torch.zeros(args.world_size, dtype=torch.int)
    iterator = train_queue
    for step, (input, target) in enumerate(iterator):
        b_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        # with autocast():
            # logits = model(input.contiguous().unsqueeze(dim=1))
            # print(input.shape, target.shape)
        # input = input.permute(0, 2, 1)
        input = input.permute(0, 2, 1).to(f"cuda:{args.local_rank}")
        target = target.to(f"cuda:{args.local_rank}")
        logits = model(input)
            # print(logits.shape)
            # logits = model(input)
            # target = target.squeeze(dim=1)
            # print("-->", input.shape, target.shape, logits.shape)
            # exit(0)
            # print(target.shape)
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


def infer(valid_queue, model, criterion, args, report_freq=100, fast=False):
    from torch.cuda.amp import autocast
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        # with autocast():
        for step, (input, target) in enumerate(valid_queue):
            input = input.permute(0, 2, 1).to(f"cuda:{args.local_rank}")
            target = target.to(f"cuda:{args.local_rank}")
            # input = input.permute(0, 2, 1)
            logits = model(input)
            all_logits.append(logits)
            all_targets.append(target)
    result = torch.cat(all_logits, dim=0)
    target = torch.cat(all_targets, dim=0)
    result = result.cpu().detach().numpy()
    y = target.cpu().detach().numpy()
    result_shape = np.shape(result)

    fpr_list, tpr_list, auroc_list = [], [], []
    precision_list, recall_list, aupr_list = [], [], []
    for i in range(result_shape[1]):
        fpr_temp, tpr_temp, auroc_temp = calculate_auroc(result[:, i], y[:, i])
        precision_temp, recall_temp, aupr_temp = calculate_aupr(result[:, i], y[:, i])

        fpr_list.append(fpr_temp)
        tpr_list.append(tpr_temp)
        precision_list.append(precision_temp)
        recall_list.append(recall_temp)
        auroc_list.append(auroc_temp)
        aupr_list.append(aupr_temp)

    header = np.array([["auroc", "aupr"]])
    content = np.stack((auroc_list, aupr_list), axis=1)
    content = np.concatenate((header, content), axis=0)
    avg_auroc = np.nanmean(auroc_list)
    avg_aupr = np.nanmean(aupr_list)
    # print("AVG-AUROC:{:.3f}, AVG-AUPR:{:.3f}.\n".format(avg_auroc, avg_aupr))
    return avg_auroc, avg_aupr, 0


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
                        "avg_auroc": avg_top1_val,
                        "avg_aupr": avg_top5_val,
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
                "Epoch %d, AUROC %f, AUPRC %f, Best AUROC %f, Best AUPRC %f",
                epoch,
                avg_top1_val,
                avg_top5_val,
                best_acc_top1,
                best_acc_top5,
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


def infer_tv(valid_queue, model, criterion, args, report_freq=100, fast=False):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            if fast and step > 0:
                break
            input, target = input.to(args.local_rank), target.to(args.local_rank)
            logits = model(input.contiguous(memory_format=torch.channels_last))
            if criterion is not None:
                loss = criterion(logits, target)
            else:
                loss = None

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            if criterion is not None:
                objs.update(loss, n)
            top1.update(prec1, n)
            top5.update(prec5, n)

            if step % report_freq == 0:
                end_time = time.time()
                if step == 0:
                    duration = 0
                    start_time = time.time()
                else:
                    duration = end_time - start_time
                    start_time = time.time()
                logging.info(
                    "VALID Step: %03d R1: %f R5: %f Duration: %ds",
                    step,
                    top1.avg,
                    top5.avg,
                    duration,
                )

    valid_acc_top1 = torch.tensor(top1.avg).to(args.local_rank)
    if args.distributed:
        acc_tensor_list = [
            torch.zeros_like(valid_acc_top1) for r in range(args.world_size)
        ]
        dist.all_gather(acc_tensor_list, valid_acc_top1)
        avg_top1_val = torch.mean(torch.stack(acc_tensor_list))
    else:
        avg_top1_val = valid_acc_top1

    valid_acc_top5 = torch.tensor(top5.avg).to(args.local_rank)
    if args.distributed:
        acc_tensor_list = [
            torch.zeros_like(valid_acc_top5) for r in range(args.world_size)
        ]
        dist.all_gather(acc_tensor_list, valid_acc_top5)
        avg_top5_val = torch.mean(torch.stack(acc_tensor_list))
    else:
        avg_top5_val = valid_acc_top5

    if criterion is not None:
        loss_rank = torch.tensor(objs.avg).to(args.local_rank)
        if args.distributed:
            loss_list = [torch.zeros_like(loss_rank) for r in range(args.world_size)]
            dist.all_gather(loss_list, loss_rank)
            avg_loss = torch.mean(torch.stack(loss_list))
        else:
            avg_loss = loss_rank
    else:
        avg_loss = None

    return avg_top1_val, avg_top5_val, avg_loss


def throughput_gpu(valid_queue, model, args, report_freq=100):
    model.eval()
    warmup_reps = 1
    measurement_reps = 3
    total_time = 0
    rep_time = 0
    prev_rep_time = 0
    print(
        f"Warming up for {warmup_reps} repetitions and then averaging {measurement_reps} repetitions for measurements..."
    )
    throughput_measurements = []
    with torch.no_grad():
        with autocast():
            for rep in range(warmup_reps + measurement_reps):
                for step, (input, target) in enumerate(valid_queue):
                    starter, ender = torch.cuda.Event(
                        enable_timing=True, blocking=True
                    ), torch.cuda.Event(enable_timing=True, blocking=True)
                    starter.record()
                    _ = model(input.contiguous(memory_format=torch.channels_last))
                    ender.record()
                    torch.cuda.synchronize()
                    total_time += starter.elapsed_time(ender) / 1000
                rep_time = total_time - prev_rep_time
                prev_rep_time = total_time
                throughput = (len(valid_queue) * args.val_batch_size) / rep_time
                rep_type = "WARMUP" if rep < warmup_reps else "MEASUREMENT"
                print(f"Rep {rep}:[{rep_type}] Throughput: {throughput}")
                if rep >= warmup_reps:
                    throughput_measurements.append(throughput)
    mean_thr = np.mean(throughput_measurements)
    std_thr = np.std(throughput_measurements)
    # throughput = (warmup_reps*len(valid_queue)*args.val_batch_size)/total_time
    print("===" * 10)
    print(f"Mean: {mean_thr}, Std: {std_thr}")
    print("===" * 10)

    """
    valid_queue.batch_size = 1
    print(f'Measuring latency at batch size {valid_queue.batch_size}, Num Samples {len(valid_queue)}...')
    num_samp = 10000
    timings = np.zeros((num_samp,1))
    with torch.no_grad():
        with autocast():
            for step, (input, target) in enumerate(valid_queue):
                if step >= num_samp:
                    break
                starter, ender = torch.cuda.Event(enable_timing=True, blocking=True), torch.cuda.Event(enable_timing=True, blocking=True)
                starter.record()
                logits = model(input.contiguous(memory_format=torch.channels_last))
                ender.record()
                torch.cuda.synchronize()
                current_time = starter.elapsed_time(ender)/1000
                timings[step] = current_time
    mean_syn = np.sum(timings) / num_samp
    std_syn = np.std(timings)
    print('==='*10)
    print(mean_syn, std_syn)
    print('==='*10)
    """
    return throughput_measurements  # mean_thr, std_thr


def calculate_aupr(predictions, labels):
    precision_list, recall_list, threshold_list = metrics.precision_recall_curve(
        y_true=labels, probas_pred=predictions
    )
    aupr = metrics.auc(recall_list, precision_list)
    return precision_list, recall_list, aupr


def calculate_auroc(predictions, labels):
    fpr_list, tpr_list, threshold_list = metrics.roc_curve(
        y_true=labels, y_score=predictions
    )
    auroc = metrics.auc(fpr_list, tpr_list)
    return fpr_list, tpr_list, auroc


def calculate_stats(output, target, class_indices=None):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)
      class_indices: list
        explicit indices of classes to calculate statistics for

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    if class_indices is None:
        class_indices = range(classes_num)
    stats = []

    # Class-wise statistics
    for k in class_indices:
        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None
        )

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        dict = {"AP": avg_precision, "auc": auc}
        stats.append(dict)

    return stats
