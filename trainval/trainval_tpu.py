import gc
import time
import torch
import numpy as np
import logging

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
      loss, # .item(),
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
    losses_m = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    model.train()
    num_batches_per_epoch = len(train_queue)
    last_idx = num_batches_per_epoch - 1
    num_updates = epoch * num_batches_per_epoch

    for step, (input, target) in enumerate(train_queue):
        b_start = time.time()
        last_batch = step == last_idx
        if mixup_fn is not None:
            input, target = mixup_fn(input, target)
        logits = model(input)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(optimizer)

        if model_ema is not None:
            model_ema.update(model)

        tracker.add(args.train_batch_size)
        batch_time.update(time.time() - b_start)
        num_updates += 1
        if last_batch or step % report_freq == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            xm.add_step_closure(_train_update, args=(args.local_rank, step, loss, tracker, epoch, args.writer))
            '''
            loss_clone = loss.data.clone()
            xm.all_reduce(xm.REDUCE_SUM, loss_clone)
            reduced_loss = loss_clone / args.world_size
            # reduced_loss = reduce_xla_tensor(loss.data, args.world_size)
            losses_m.update(reduced_loss.item(), input.size(0))
            '''
            losses_m.update(loss.data.item(), input.size(0))
        if lr_schedule is not None:
            lr_schedule.step_update(num_updates=num_updates, metric=losses_m.avg)
        del loss, logits, target
    return top1.avg, losses_m.avg

def reduce_xla_tensor(tens, world_size):
    clone = tens.clone()
    xm.all_reduce(xm.REDUCE_SUM, clone)
    reduced = clone / world_size
    return reduced

def infer_tpu(
    valid_queue, model, criterion, args, epoch, report_freq=100
):

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    last_idx = len(valid_queue) - 1

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            last_batch = step == last_idx
            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

            # reduced_loss = reduce_xla_tensor(loss.data, args.world_size)
            # prec1 = reduce_xla_tensor(prec1, args.world_size)
            # prec5 = reduce_xla_tensor(prec5, args.world_size)

            objs.update(loss.data.item(), input.size(0))
            top1.update(prec1.data.item(), logits.size(0))
            top5.update(prec5.data.item(), logits.size(0))

            if last_batch or step % report_freq == 0:
                xm.add_step_closure(test_utils.print_test_update, args=(args.local_rank, top1.avg, epoch, step))
            del loss, logits, target
        top1_acc = top1.avg
        top5_acc = top1.avg
        loss = objs.avg
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
    criterion,
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
    train_queue = pl.MpDeviceLoader(train_queue, device)
    valid_queue = pl.MpDeviceLoader(valid_queue, device)

    for epoch in range(_epochs):
        # training
        '''
        lr = utils.adjust_lr(optimizer, epoch, args.lr, args.epochs)
        if epoch < 5 and args.train_batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * (epoch + 1) / 5.0
            print("Warming-up Epoch: %d, LR: %e" % (epoch, lr * (epoch + 1) / 5.0))
        '''
        sampler.set_epoch(epoch)
        epoch_start = time.time()
        train_acc, train_obj = train_epoch_tpu(
            epoch,
            train_queue,
            model,
            model_ema,
            criterion,
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
            #commit = True if epoch <= _epochs - 4 else False
            commit = False
            args.wandb_con.log({"t_acc": train_acc, "t_loss": train_obj}, commit=commit)
        # validation
        if epoch > -1:
            valid_acc_top1, valid_acc_top5, valid_obj = infer_tpu(
                valid_queue,
                model,
                criterion,
                args,
                epoch,
                args.report_freq,
            )
            avg_top1_val, avg_top5_val = valid_acc_top1, valid_acc_top5

            if args.global_rank == 0 and args.wandb_con is not None:
                #commit = True if epoch < _epochs - 1 else False
                commit = True
                args.wandb_con.log(
                    {
                        "valid_acc_top1": avg_top1_val,
                        "valid_acc_top5": avg_top5_val,
                        "v_loss": valid_obj,
                        "Train Time": time.time() - train_sttime,
                    },
                    commit=commit,
                )
            if avg_top5_val > best_acc_top5:
                best_acc_top5 = avg_top5_val
            if avg_top1_val > best_acc_top1:
                best_acc_top1 = avg_top1_val
                best_model_sd = model.state_dict()
                utils.save_checkpoint(
                {
                    "state_dict": best_model_sd,
                },
                args.save,
                )

            if args.global_rank == 0:
                print(
                    "Epoch %d, Valid_acc_top1 %f,\
                    Valid_acc_top5 %f, Best_top1 %f, Best_top5 %f"
                    % (epoch, avg_top1_val, avg_top5_val, best_acc_top1, best_acc_top5)
                )
            logging.info(
                "Epoch %d, Valid_acc_top1 %f,\
                Valid_acc_top5 %f, Best_top1 %f, Best_top5 %f",
                epoch, avg_top1_val, avg_top5_val, best_acc_top1, best_acc_top5
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
def throughput_tpu(
    valid_queue, model, optimizer, criterion, args, report_freq=100
):
    # model.eval()
    model.train()
    repetitions = 1
    total_time = 0
    warmup_reps = 0
    device = xm.xla_device()
    start_time = time.time()
    rep_time = []
    rates = []
    cur_rate = [torch.zeros(1, device=device)]
    last_rate = 0
    tracker = xm.RateTracker()
    last_thro = 0
    repetitions = 1 
    report_freq = 20
    loader = valid_queue#itertools.chain(valid_queue, valid_queue)
    import torch_xla.test.test_utils as test_utils
    for rep in range(repetitions):
        rep_start = time.time()
        for step, (input, target) in enumerate(loader):
            if step > 300:
                continue
            logits = model(input)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            xm.optimizer_step(optimizer)

            tracker.add(args.val_batch_size)
            if ((step + 1) % report_freq) == 0:
                # xm.wait_device_ops()
                cur_rate = tracker.rate()
                grate = tracker.global_rate()
                # xm.add_step_closure(_train_update, args=(args.local_rank, step, 0, tracker, 0, None))
                test_utils.print_training_update(
                      args.local_rank,
                      step,
                      cur_rate-last_rate,
                      cur_rate,
                      grate)


                # xm.add_step_closure(test_utils.print_test_update, args=(args.local_rank, None, 0, step))
                # old_rate = cur_rate
                # cur_rate = [torch.tensor(tracker.rate(), device=device)]
                # xm.all_reduce(xm.REDUCE_SUM, cur_rate)
                # print(f'Rates: {tracker.rate()}, {tracker.global_rate()}, Sum: {cur_rate}, Change: {cur_rate[0]-old_rate[0]}')
                # print(f'Rank: {args.local_rank}, step: {step}, Current Rate: {cur_rate}, Global Rate: {tracker.global_rate()}, Rate Change: {cur_rate-last_rate}')
                last_rate = cur_rate
                if step+1 >= 180 and step+1 <= 300:
                    rates.append(last_rate)
                # if step+1 == 450:
                #    break
                # xm.rendezvous('step_finished')
        rep_time.append(time.time() - rep_start)
        # print(rep_time)
        # print(rates)
    # total_time = sum(rep_time[warmup_reps:])
    mean_thr = np.mean(rates)
    std_thr = np.square(np.std(rates))
    print(f'Mean: {mean_thr}, Std: {std_thr}')

    mean_thr = xm.mesh_reduce('mean_thr', mean_thr, np.sum)
    std_thr = np.sqrt(xm.mesh_reduce('std_thr', std_thr, np.sum))
    print(f'Mean: {mean_thr}, Std: {std_thr}')
    del model, valid_queue

    # throughput = [torch.tensor(mean_thr, device=device)]
    # total_time = 0.1
    # throughput = [torch.tensor(((repetitions - warmup_reps) * args.val_batch_size * len(valid_queue))/total_time, device=device)]
    # print(f'Throughput: {throughput}')
    # xm.all_reduce(xm.REDUCE_SUM, throughput)
    # print(f'Summed Throughput: {throughput}')
    # _std = torch.std(throughput[])
    # print(f'Summed Std: {_std}')

    # avg_top1_val, avg_top5_val, avg_loss = top1.avg, top5.avg, objs.avg
    return mean_thr, std_thr
