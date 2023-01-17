import gc
import time
import torch
import numpy as np
import logging

# from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
import torch_xla.core.xla_model as xm

import auxiliary.utils as utils

from models.accelbenchnet import AccelNet as Network
import torch_xla.distributed.parallel_loader as pl
from timm.utils import reduce_tensor


def _train_update(device, step, loss, tracker, epoch, writer):
    import torch_xla.test.test_utils as test_utils
    test_utils.print_training_update(
      device,
      step,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      epoch,
      summary_writer=writer)


def train_epoch_tpu(
    epoch,
    train_queue,
    model,
    criterion,
    optimizer,
    lr_schedule,
    mixup_fn,
    report_freq,
    fast,
    epochs,
    argslr,
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

    iterator = train_queue
    for step, (input, target) in enumerate(iterator):
        b_start = time.time()
        last_batch = step == last_idx

        # _step = step // args.update_freq
        # if _step >= args.train_steps_per_epoch:
        #    continue
        # it = epoch*args.train_steps_per_epoch + _step
        '''
        if lr_schedule is not None or wd_schedule is not None and step % args.update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule is not None:
                    param_group['lr'] = lr_schedule[it] * param_group['lr_scale']
                if wd_schedule is not None and param_group['weight_decay'] > 0:
                    param_group['weight_decay'] = wd_schedule[it]
        '''

        if mixup_fn is not None:
            input, target = mixup_fn(input, target)

        logits = model(input)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss /= args.update_freq
        loss.backward()
        if (step + 1) % args.update_freq == 0:
            xm.optimizer_step(optimizer)
            # optimizer.zero_grad(set_to_none=True)
        tracker.add(args.train_batch_size)
        batch_time.update(time.time() - b_start)
        num_updates += 1
        # prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        # n = input.size(0)
        # objs.update(loss.data.item(), n)
        # top1.update(prec1.data.item(), n)
        # top5.update(prec5.data.item(), n)
        if last_batch or step % report_freq == 0:
            # end_time = time.time()
            xm.add_step_closure(_train_update, args=(args.local_rank, step, loss, tracker, epoch, args.writer))
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            losses_m.update(reduced_loss.item(), input.size(0))
        lr_schedule.step_update(num_updates=num_updates, metric=losses_m.avg)

        del loss, logits, target
    return top1.avg, losses_m.avg


def infer_tpu(
    valid_queue, model, criterion, args, epoch, report_freq=100, fast=False
):

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    import torch_xla.test.test_utils as test_utils

    for step, (input, target) in enumerate(valid_queue):
        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % report_freq == 0:
            end_time = time.time()
            xm.add_step_closure(test_utils.print_test_update, args=(args.local_rank, None, epoch, step))
            '''
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info(
                "VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds",
                step,
                objs.avg,
                top1.avg,
                top5.avg,
                duration,
            )
            '''
        del loss, logits, target
    avg_top1_val, avg_top5_val, avg_loss = top1.avg, top5.avg, objs.avg
    return avg_top1_val, avg_top5_val, avg_loss


def train_x_epochs_tpu(
    _epochs,
    lr_scheduler,
    train_queue,
    valid_queue,
    model,
    criterion,
    optimizer,
    mixup_fn,
    global_rank,
    local_rank,
    world_size,
    wandb_con,
    args,
):
    train_sttime = time.time()
    best_acc_top1 = 0
    best_acc_top5 = 0
    valid_acc_top1, valid_acc_top5, valid_obj = None, None, None
    device = xm.xla_device()
    train_queue = pl.MpDeviceLoader(train_queue, device)
    valid_queue = pl.MpDeviceLoader(valid_queue, device)

    for epoch in range(_epochs):
        # training
        lr = utils.adjust_lr(optimizer, epoch, args.lr, args.epochs)
        if epoch < 5 and args.train_batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * (epoch + 1) / 5.0
            print("Warming-up Epoch: %d, LR: %e" % (epoch, lr * (epoch + 1) / 5.0))

        epoch_start = time.time()
        train_acc, train_obj = train_epoch_tpu(
            epoch,
            train_queue,
            valid_queue,
            model,
            criterion,
            optimizer,
            lr_scheduler,
            mixup_fn,
            args.report_freq,
            args.fast,
            _epochs,
            args.lr,
            args,
        )

        # scheduler.step()
        # logging.info('Train_acc %f', train_acc)
        epoch_duration = time.time() - epoch_start
        logging.info(
            "Epoch %d, Train_acc %f, Epoch time: %ds",
            epoch, train_acc, epoch_duration
        )
        if global_rank == 0:
            if wandb_con is not None:
                #commit = True if epoch <= _epochs - 4 else False
                commit = False
                wandb_con.log({"t_acc": train_acc, "t_loss": train_obj}, commit=commit)
        # validation
        if epoch > -1:#_epochs - 4:
            valid_acc_top1, valid_acc_top5, valid_obj = infer_tpu(
                valid_queue,
                model,
                criterion,
                args,
                epoch,
                args.report_freq,
                args.fast,
            )
            # logging.info('Epoch %d, Valid_acc_top1 %f, Valid_acc_top5 %f, Best_top1 %f, Best_top5 %f', epoch, valid_acc_top1, valid_acc_top5, best_acc_top1, best_acc_top5)
            avg_top1_val, avg_top5_val = valid_acc_top1, valid_acc_top5

            if global_rank == 0 and wandb_con is not None:
                #commit = True if epoch < _epochs - 1 else False
                commit = True
                wandb_con.log(
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
            if global_rank == 0:
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
    train_endtime = time.time()
    if global_rank == 0:
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
            "train_time": train_sttime - train_endtime,
            "best_acc_top1": best_acc_top1,
            "best_acc_top5": best_acc_top5,
            "architecture": args.design,
        }
        utils.save_checkpoint(
            {
                "training_config": training_config_dict,
                "model_metadata": model_metadata_dict,
                "state_dict": model.state_dict(),
            },
            args.save,
        )
        if args.use_wandb and wandb_con is not None:
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
            wandb_con.log_artifact(wandb_art)

    return [train_acc, train_obj, avg_top1_val, avg_top5_val, valid_obj, True]


def infer_tv(
    valid_queue, model, criterion, args, report_freq=100, fast=False
):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
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
                objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

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
