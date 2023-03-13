import gc
import time
import torch
import numpy as np
import logging

# from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.cuda.amp import autocast

import auxiliary.utils as utils

from models.accelbenchnet import AccelNet as Network


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
        if args.distributed:
            dist.barrier()  # Synchronize at every step
        if fast and step > report_freq:
            break  # Dry run to ensure trainability
        for param_group in optimizer.param_groups:
            param_group["lr"] = lrs[step]

        b_start = time.time()

        optimizer.zero_grad(set_to_none=True)
        if rank_finished[args.local_rank] == 0:
            try:
                with autocast():
                    logits = model(input.contiguous(memory_format=torch.channels_last))
                    loss = criterion(logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            except RuntimeError:
                print(f"Rank {args.local_rank} ran out of GPU memory...")
                rank_finished[args.local_rank] = 1
                # return 0, 0, None
        # if step % report_freq == 0:
        #    dist.barrier()
        #    dist.all_reduce(rank_finished, op=dist.ReduceOp.SUM)
        #    dist.barrier()
        if any(rank_finished):
            # This conditional ensures that when one GPU runs out of memory, remaining GPUs abandon training
            continue
        batch_time.update(time.time() - b_start)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
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
    if args.distributed:
        dist.barrier()
    if any(rank_finished):
        return 0, 0, None
    return top1.avg.data.item(), objs.avg.data.item(), scaler


def infer(valid_queue, model, criterion, args, report_freq=100, fast=False):

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    from torch.cuda.amp import autocast

    with torch.no_grad():
        with autocast():
            for step, (input, target) in enumerate(valid_queue):
                try:
                    logits = model(input.contiguous(memory_format=torch.channels_last))
                    loss = criterion(logits, target)

                    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                    n = input.size(0)
                    objs.update(loss, n)
                    top1.update(prec1, n)
                    top5.update(prec5, n)
                except RuntimeError:
                    # raise RuntimeError("Ran out of GPU memory...")
                    print("Ran out of GPU memory...")
                    return 0, 0, 0

                if step % report_freq == 0:
                    end_time = time.time()
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
                del loss, logits, target

    valid_acc_top1 = torch.tensor(top1.avg).to(args.local_rank)
    if args.distributed:
        dist.barrier()
        acc_tensor_list = [
            torch.zeros_like(valid_acc_top1) for r in range(args.world_size)
        ]
        dist.all_gather(acc_tensor_list, valid_acc_top1)
        avg_top1_val = torch.mean(torch.stack(acc_tensor_list))
        del acc_tensor_list
    else:
        avg_top1_val = valid_acc_top1

    valid_acc_top5 = torch.tensor(top5.avg).to(args.local_rank)
    if args.distributed:
        dist.barrier()
        acc_tensor_list = [
            torch.zeros_like(valid_acc_top5) for r in range(args.world_size)
        ]
        dist.all_gather(acc_tensor_list, valid_acc_top5)
        avg_top5_val = torch.mean(torch.stack(acc_tensor_list))
        del acc_tensor_list
    else:
        avg_top5_val = valid_acc_top5

    loss_rank = torch.tensor(objs.avg).to(args.local_rank)
    if args.distributed:
        dist.barrier()
        loss_list = [torch.zeros_like(loss_rank) for r in range(args.world_size)]
        dist.all_gather(loss_list, loss_rank)
        avg_loss = torch.mean(torch.stack(loss_list))
        del loss_list
    else:
        avg_loss = loss_rank
    del loss_rank
    return avg_top1_val, avg_top5_val, avg_loss


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
    from dataloader.ffcv_dataloader import get_resolution

    for epoch in range(_epochs):
        # training
        res = get_resolution(
            epoch, args.min_res, args.max_res, args.end_ramp, args.start_ramp
        )
        dl_class_inst.decoder.output_size = (res, res)
        lr = utils.adjust_lr(optimizer, epoch, args.lr, args.epochs)
        if epoch < 5 and args.train_batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * (epoch + 1) / 5.0
            print("Warming-up Epoch: %d, LR: %e" % (epoch, lr * (epoch + 1) / 5.0))

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
        if train_acc == 0 and train_obj == 0:
            if args.distributed:
                dist.barrier()
            return [0, 0, 0, 0, 0, False]

        scheduler.step()
        # logging.info('Train_acc %f', train_acc)
        epoch_duration = time.time() - epoch_start
        logging.info(
            "Epoch %d, Train_acc %f, Epoch time: %ds", epoch, train_acc, epoch_duration
        )
        if global_rank == 0:
            if wandb_con is not None:
                commit = True if epoch <= _epochs - 4 else False
                wandb_con.log({"t_acc": train_acc, "t_loss": train_obj}, commit=commit)
        # validation
        if epoch > _epochs - 4:
            valid_acc_top1, valid_acc_top5, valid_obj = infer(
                valid_queue,
                model,
                criterion,
                args,
                args.report_freq,
                args.fast,
            )
            if valid_acc_top1 == 0 and valid_acc_top5 == 0:
                return [0, 0, 0, 0, 0, False]
            # logging.info('Epoch %d, Valid_acc_top1 %f, Valid_acc_top5 %f, Best_top1 %f, Best_top5 %f', epoch, valid_acc_top1, valid_acc_top5, best_acc_top1, best_acc_top5)
            avg_top1_val, avg_top5_val = valid_acc_top1, valid_acc_top5
            if global_rank == 0 and wandb_con is not None:
                commit = True if epoch < _epochs - 1 else False
                wandb_con.log(
                    {
                        "valid_acc_top1": avg_top1_val,
                        "valid_acc_top5": avg_top5_val,
                        "v_loss": valid_obj,
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
            """
            if global_rank == 0:
                print(
                    "Epoch %d, Valid_acc_top1 %f,\
                    Valid_acc_top5 %f, Best_top1 %f, Best_top5 %f"
                    % (epoch, avg_top1_val, avg_top5_val, best_acc_top1, best_acc_top5)
                )
            """
            logging.info(
                "Epoch %d, Valid_acc_top1 %f,\
                Valid_acc_top5 %f, Best_top1 %f, Best_top5 %f",
                epoch,
                avg_top1_val,
                avg_top5_val,
                best_acc_top1,
                best_acc_top5,
            )
        if args.distributed:
            dist.barrier()
    train_endtime = time.time()
    if global_rank == 0:
        training_config_dict = {
            "model_num": args.model_num,
            "job_id": args.job_id,
            "epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "val_resolution": args.val_resolution,
            "label_smoothing": args.label_smoothing,
            "min_res": args.min_res,
            "max_res": args.max_res,
            "start_ramp": args.start_ramp,
            "end_ramp": args.end_ramp,
            "seed": args.seed,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_peak_epoch": args.lr_peak_epoch,
            "world_size": args.world_size,
        }
        model_metadata_dict = {
            "macs": args.macs,
            "params": args.params,
            "train_time": train_sttime - train_endtime,
            "best_acc_top1": best_acc_top1.cpu().numpy(),
            "best_acc_top5": best_acc_top5.cpu().numpy(),
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
        train_recipe = {
            "epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "min_res": args.min_res,
            "max_res": args.max_res,
            "start_ramp": args.start_ramp,
            "end_ramp": args.end_ramp,
        }
        artifact_dict = {
            "key": args.model_num,
            "macs": args.macs,
            "params": args.params,
            "train_recipe": train_recipe,
            "acc": best_acc_top1,
            "train_time": train_endtime - train_sttime,
        }
        if args.use_wandb and wandb_con is not None:
            import wandb

            wandb_art = wandb.Artifact(
                # name=f"models-grid-proxified-jobid{args.job_id}-model{args.model_num}",
                name=f"models-searchrecipe-jobid{args.job_id}-{args.arch_epoch}-{args.episode}",
                type="model",
                metadata={
                    "training_config": training_config_dict,
                    "model_metadata": model_metadata_dict,
                    "grid_eval_dict": artifact_dict
                },
            )
            wandb_art.add_file(f"{args.save}/f_model.pth")
            wandb_con.log_artifact(wandb_art)
    if args.distributed:
        dist.barrier()

    return [train_acc, train_obj, avg_top1_val, avg_top5_val, valid_obj, True]


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


def dry_run(design, platform, mode, criterion, args):
    from dataloader.ffcv_dataloader import get_ffcv_loaders
    from torch.cuda.amp import GradScaler

    logging.info("Performing dry run on design with peak resolution...")
    scaler = GradScaler()
    train_queue, valid_queue, dl = get_ffcv_loaders(args.local_rank, args)
    dl.decoder.output_size = (args.max_res, args.max_res)

    model = Network(design=args.design, platform=platform, mode=mode)
    model = model.to(memory_format=torch.channels_last)
    model = model.to(f"cuda:{args.local_rank}")
    dist.barrier()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer, scheduler = utils.create_optimizer(
        model, args.lr, args.weight_decay, args
    )
    _, success, _ = train(
        0,
        train_queue,
        valid_queue,
        model,
        criterion,
        optimizer,
        scaler,
        report_freq=10,
        fast=True,
        lr_peak_epoch=2,
        epochs=1,
        argslr=args.lr,
        args=args,
    )
    del train_queue, valid_queue, dl, model, optimizer, scheduler
    if success:
        logging.info("Design trainable...")
    torch.cuda.empty_cache()
    gc.collect()
    return success


def throughput_gpu(valid_queue, model, args, report_freq=100):
    model.eval()
    warmup_reps = 1
    measurement_reps = 2
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
    return mean_thr, std_thr
