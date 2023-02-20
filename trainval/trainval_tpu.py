import gc
import time
import torch
import numpy as np
import logging

from contextlib import suppress
import itertools
import torch.distributed as dist
import torch_xla.core.xla_model as xm

import auxiliary.utils as utils

import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils


def _train_update(device, step, loss, tracker, epoch, writer):
    import torch_xla.test.test_utils as test_utils
    test_utils.print_training_update(
      device,
      step,
      loss,
      tracker.rate(),
      tracker.global_rate(),
      epoch,
      summary_writer=writer)


def train_epoch_tpu(
    epoch,
    train_queue,
    model,
    model_ema,
    criterion,
    optimizer,
    lr_schedule,
    mixup_fn,
    report_freq,
    args
):
    tracker = xm.RateTracker()
    losses_m = utils.AverageMeter()
    top1 = utils.AverageMeter()
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    model.train()
    num_batches_per_epoch = len(train_queue)
    last_idx = num_batches_per_epoch - 1
    num_updates = epoch * num_batches_per_epoch
    optimizer.zero_grad()
    b_start = time.time()
    for step, (input, target) in enumerate(train_queue):
        last_batch = step == last_idx
        data_time_m.update(time.time() - b_start)
        if mixup_fn is not None:
            input, target = mixup_fn(input, target)
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        xm.reduce_gradients(optimizer)
        # xm.optimizer_step(optimizer)
        optimizer.step()
        xm.mark_step()
        optimizer.zero_grad()

        tracker.add(args.train_batch_size)
        batch_time_m.update(time.time() - b_start)

        if model_ema is not None:
            model_ema.update(model)

        num_updates += 1
        if last_batch or step % report_freq == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            losses_m.update(loss.data.item(), input.size(0))

            # xm.mark_step()
            # print('Here0')
            time_elapsed = batch_time_m.sum / 60.
            steps_left = num_batches_per_epoch - step
            time_left = ((batch_time_m.sum / (step+1)) * steps_left) / 60.
            logging.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'LR: {lr:.3e}  '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})  '
                'Time: [{time_elapsed:.1f}/{time_left:.1f} mins]'.format(
                    epoch,
                    step, len(train_queue),
                    100. * step / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) * args.world_size / batch_time_m.val,
                    rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m,
                    time_elapsed=time_elapsed,
                    time_left=time_left)
            )

            # xm.add_step_closure(_train_update, args=(args.local_rank, step, loss, tracker, epoch, args.writer))
            '''
            loss_clone = loss.data.clone()
            xm.all_reduce(xm.REDUCE_SUM, loss_clone)
            reduced_loss = loss_clone / args.world_size
            # reduced_loss = reduce_xla_tensor(loss.data, args.world_size)
            losses_m.update(reduced_loss.item(), input.size(0))
            '''
        if lr_schedule is not None:
            lr_schedule.step_update(num_updates=num_updates) #, metric=losses_m.avg)
        b_start = time.time()
    return top1.avg, losses_m.avg

def reduce_xla_tensor(tens, world_size):
    clone = tens.clone()
    clone = xm.all_reduce(xm.REDUCE_SUM, clone)
    reduced = clone / world_size
    return reduced

def infer_tpu(
    valid_queue, model, criterion, args, epoch, report_freq=100
):

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    batch_time_m = utils.AverageMeter()
    end = time.time()
    last_idx = len(valid_queue) - 1
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            last_batch = step == last_idx
            logits = model(input)
            loss = criterion(logits, target)
            xm.mark_step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

            #FIXME Necessary to reduce in each iteration? Or cumulative after done?
            # reduced_loss = reduce_xla_tensor(loss.data, args.world_size)
            # prec1 = reduce_xla_tensor(prec1, args.world_size)
            # prec5 = reduce_xla_tensor(prec5, args.world_size)

            objs.update(loss, input.size(0))
            top1.update(prec1, logits.size(0))
            top5.update(prec5, logits.size(0))

            # objs.update(loss.data.item(), input.size(0))
            # top1.update(prec1.data.item(), logits.size(0))
            # top5.update(prec5.data.item(), logits.size(0))
            batch_time_m.update(time.time() - end)
            if last_batch or step % report_freq == 0:
                # xm.add_step_closure(test_utils.print_test_update, args=(args.local_rank, top1.avg, epoch, step))
                xm.mark_step()
                logging.info(
                    'Test [{0}]: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        epoch, step, last_idx,
                        batch_time=batch_time_m,
                        loss=objs,
                        top1=top1,
                        top5=top5)
                )
            end = time.time()
        top1_acc = top1.avg
        top5_acc = top1.avg
        loss = objs.avg
        #FIXME Is  this needed since already reduced in each iteration
        accuracy_top1 = xm.mesh_reduce('top1_accuracy', top1_acc, np.mean)
        accuracy_top5 = xm.mesh_reduce('top5_accuracy', top5_acc, np.mean)
        loss_avg = xm.mesh_reduce('loss', loss, np.mean)
    return accuracy_top1, accuracy_top5, loss_avg# top1.avg, top5.avg, objs.avg


def train_x_epochs_tpu(
    _epochs,
    lr_scheduler,
    train_queue,
    valid_queue,
    model,
    model_ema,
    criterion_train,
    criterion_val,
    optimizer,
    mixup_fn,
    args,
):
    train_sttime = time.time()
    best_acc_top1 = 0
    best_acc_top5 = 0
    best_model_sd = model.state_dict()
    valid_acc_top1, valid_acc_top5, valid_obj = None, None, None
    device = xm.xla_device()
    sampler = train_queue.sampler

    for epoch in range(_epochs):
        # training
        sampler.set_epoch(epoch)
        epoch_start = time.time()
        train_acc, train_obj = train_epoch_tpu(
            epoch,
            train_queue,
            model,
            model_ema,
            criterion_train,
            optimizer,
            lr_scheduler,
            mixup_fn,
            args.report_freq,
            args,
        )

        epoch_duration = time.time() - epoch_start
        logging.info(
            "Epoch %d, Train_acc %f, Epoch time: %ds",
            epoch, train_acc, epoch_duration
        )
        if args.global_rank == 0 and args.wandb_con is not None:
            commit = False
            args.wandb_con.log({"t_acc": train_acc, "t_loss": train_obj}, commit=commit)
        # Distrubute bn mean and vars
        utils.distribute_bn_tpu(model, args.world_size, reduce=True)

        # validation
        if epoch > -1:
            valid_acc_top1, valid_acc_top5, valid_obj = infer_tpu(
                valid_queue,
                model,
                criterion_val,
                args,
                epoch,
                args.report_freq,
            )
            avg_top1_val, avg_top5_val = valid_acc_top1, valid_acc_top5

            # Distribute batchnorm model_ema here
            if model_ema is not None and not args.model_ema_force_cpu:
                utils.distribute_bn_tpu(model_ema, args.world_size, reduce=True)
                avg_top1_val_ema, avg_top5_val_ema, valid_obj_ema = infer_tpu(
                        valid_queue,
                        model_ema.module,
                        criterion_val,
                        args,
                        epoch,
                        args.report_freq,
                    )

            if args.global_rank == 0 and args.wandb_con is not None:
                #commit = True if epoch < _epochs - 1 else False
                commit = False
                args.wandb_con.log(
                    {
                        "valid_acc_top1": avg_top1_val,
                        "valid_acc_top5": avg_top5_val,
                        "v_loss": valid_obj,
                        "Train Time": time.time() - train_sttime,
                    },
                    commit=commit,
                )
                if model_ema is not None and not args.model_ema_force_cpu:
                    args.wandb_con.log(
                            { 
                            "ema_valid_top1": avg_top1_val_ema,
                            "ema_valid_top5": avg_top5_val_ema,
                            "ema_v_loss": valid_obj_ema,
                            }, commit=True)
            logging.info(
                "[TEST] Epoch %d, Valid_acc_top1 %f, Valid_acc_top5 %f",
                epoch, avg_top1_val, avg_top5_val
            )
            logging.info(
                "[TEST][EMA] Epoch %d, Valid_acc_top1 %f, Valid_acc_top5 %f, Best_top1 %f, Best_top5 %f",
                epoch, avg_top1_val_ema, avg_top5_val_ema, best_acc_top1, best_acc_top5
            )
            if model_ema is not None and not args.model_ema_force_cpu:
                avg_top1_val, avg_top5_val, valid_obj = avg_top1_val_ema, avg_top5_val_ema, valid_obj_ema
            if avg_top5_val > best_acc_top5:
                best_acc_top5 = avg_top5_val
            if avg_top1_val > best_acc_top1:
                best_acc_top1 = avg_top1_val
                best_model_sd = {k: v.cpu() for k,v in model.state_dict().items()}
                best_model_ema_sd = {k: v.cpu() for k,v in model_ema.state_dict().items()}
                if args.global_rank == 0:
                    utils.save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": best_model_sd,
                        "ema_state_dict": best_model_ema_sd,
                    },
                    args.save,
                    )
        lr_scheduler.step(epoch + 1, valid_acc_top1)
    train_endtime = time.time()
    if args.global_rank == 0:
        training_config_dict = {
            "model_num": args.model_num,
            "job_id": args.job_id,
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "batch_size": args.train_batch_size,
            "label_smoothing": args.label_smoothing,
            "seed": args.seed,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "world_size": args.world_size,
        }
        model_metadata_dict = {
            "macs": args.macs,
            "params": args.params,
            "train_time": train_endtime - train_sttime,
            "best_acc_top1": best_acc_top1,
            "best_acc_top5": best_acc_top5,
            "architecture": args.design,
        }
        utils.save_checkpoint(
            {
                "training_config": training_config_dict,
                "model_metadata": model_metadata_dict,
                "state_dict": best_model_sd,
                "ema_state_dict": best_model_ema_sd
            },
            args.save,
        )
        if args.use_wandb and args.wandb_con is not None:
            import wandb

            wandb_art = wandb.Artifact(
                name=f"deploymodel-tpu-jobid{args.job_id}-model{args.model_num}",
                type="model",
                metadata={
                    "training_config": training_config_dict,
                    "model_metadata": model_metadata_dict,
                },
            )
            wandb_art.add_file(f"{args.save}/f_model.pth")
            args.wandb_con.log_artifact(wandb_art)

    return [train_acc, train_obj, avg_top1_val, avg_top5_val, valid_obj]


import torch_xla.test.test_utils as test_utils
def throughput_train_tpu(
    valid_queue, model, optimizer, criterion, args, report_freq=100
):
    # model.eval()
    model.train()
    repetitions = 2
    total_time = 0
    warmup_reps = 0
    device = xm.xla_device()
    start_time = time.time()
    rep_time = []
    rates = []
    last_rate = 0
    tracker = xm.RateTracker()
    last_thro = 0
    repetitions = 2
    report_freq = 100000
    for rep in range(repetitions):
        rep_start = time.time()
        for step, (input, target) in enumerate(valid_queue):
            logits = model(input)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            xm.optimizer_step(optimizer)

            tracker.add(args.val_batch_size)
            if ((step + 1) % report_freq) == 0:
                cur_rate = tracker.rate()
                grate = tracker.global_rate()
                # xm.add_step_closure(_train_update, args=(args.local_rank, step, 0, tracker, 0, None))
                test_utils.print_training_update(
                      args.local_rank,
                      step,
                      cur_rate-last_rate,
                      cur_rate,
                      grate)


                last_rate = cur_rate
                if step+1 >= 60: # and step+1 <= 300:
                    rates.append(last_rate)
        xm.wait_device_ops()
        rep_time.append(time.time() - rep_start)
        print(rep_time)
    '''
    mean_thr = np.mean(rates)
    std_thr = np.square(np.std(rates))
    print(f'Mean: {mean_thr}, Std: {std_thr}')
    '''

    mean_thr = xm.mesh_reduce('mean_thr', rep_time[-1], np.mean)
    std_thr = np.sqrt(xm.mesh_reduce('std_thr', rep_time[-1], np.std))
    print(f'Mean: {mean_thr}, Std: {std_thr}')
    del model, valid_queue

    return mean_thr, std_thr

def throughput_val_tpu(
    valid_queue, model, args, report_freq=100
):
    model.eval()
    device = xm.xla_device()
    rep_time = []
    rates = []
    last_rate = 0
    tracker = xm.RateTracker()
    repetitions = 1
    report_freq = 50
    for rep in range(repetitions):
        rep_start = time.time()
        for step, (input, target) in enumerate(valid_queue):
            logits = model(input)
            xm.mark_step()

            tracker.add(args.val_batch_size)
            if ((step + 1) % report_freq) == 0:
                xm.wait_device_ops()
                cur_rate = tracker.rate()
                grate = tracker.global_rate()
                test_utils.print_training_update(
                      args.local_rank,
                      step,
                      cur_rate-last_rate,
                      cur_rate,
                      grate)
                last_rate = cur_rate
                if step+1 >= 300: # and step+1 <= 300:
                    rates.append(last_rate)
        xm.wait_device_ops()
        rep_time.append(time.time() - rep_start)
        # print(rep_time)
    mean_thr = np.mean(rates)
    std_thr = np.std(rates)
    print(f'Mean: {mean_thr}, Std: {std_thr}')

    del model, valid_queue

    return mean_thr, std_thr
