import time
import torch
import numpy as np
import logging
#from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist

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
):
    from torch.cuda.amp import autocast
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    model.train()

    lr_start, lr_end = utils.get_cyclic_lr(
        epoch, argslr, epochs, lr_peak_epoch
    ), utils.get_cyclic_lr(epoch + 1, argslr, epochs, lr_peak_epoch)
    iters = len(train_queue)
    lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

    iterator = train_queue
    for step, (input, target) in enumerate(iterator):
        if fast and step > 10:
            break
        for param_group in optimizer.param_groups:
            param_group["lr"] = lrs[step]

        b_start = time.time()

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(input)
            loss = criterion(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - b_start)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
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
                "TRAIN Step: %03d Objs: %e R1: %f\
                R5: %f Duration: %ds BTime: %.3fs",
                step,
                objs.avg,
                top1.avg,
                top5.avg,
                duration,
                batch_time.avg,
            )

    return top1.avg, objs.avg, scaler


def infer(
    valid_queue, model, criterion, args, report_freq=100, lr_tta=True, fast=False
):
    from torch.cuda.amp import autocast
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        if fast and step > 0:
            break
        with torch.no_grad():
            # input, target = input.to(args.local_rank), target.to(args.local_rank)
            with autocast():
                logits = model(input.contiguous())
                if lr_tta:
                    sss = model(torch.flip(input, dims=[3]).contiguous())
                    logits += sss
                loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
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
                "VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds",
                step,
                objs.avg,
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

    loss_rank = torch.tensor(objs.avg).to(args.local_rank)
    if args.distributed:
        loss_list = [torch.zeros_like(loss_rank) for r in range(args.world_size)]
        dist.all_gather(loss_list, loss_rank)
        avg_loss = torch.mean(torch.stack(loss_list))
    else:
        avg_loss = loss_rank
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
    wandb_art,
    args,
):
    from torch.cuda.amp import GradScaler
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
        )
        scheduler.step()
        # logging.info('Train_acc %f', train_acc)
        epoch_duration = time.time() - epoch_start
        if global_rank == 0:
            print(
                "Epoch %d, Train_acc %f, Epoch time: %ds"
                % (epoch, train_acc, epoch_duration)
            )
            if wandb_con is not None:
                wandb_con.log({"t_acc": train_acc, "t_loss": train_obj}, commit=False)
        # validation
        if epoch > _epochs - 4:
            valid_acc_top1, valid_acc_top5, valid_obj = infer(
                valid_queue, model, criterion, args, args.report_freq, args.fast
            )
            # logging.info('Epoch %d, Valid_acc_top1 %f, Valid_acc_top5 %f, Best_top1 %f, Best_top5 %f', epoch, valid_acc_top1, valid_acc_top5, best_acc_top1, best_acc_top5)
            avg_top1_val, avg_top5_val = valid_acc_top1, valid_acc_top5

            if global_rank == 0 and wandb_con is not None:
                wandb_con.log(
                    {
                        "valid_acc_top1": avg_top1_val,
                        "valid_acc_top5": avg_top5_val,
                        "v_loss": valid_obj,
                    },
                    commit=True,
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
    if global_rank == 0:
        utils.save_checkpoint(
            {
                "model_info": {"macs": args.macs,
                               "params": args.params,
                               'best_acc_top1': best_acc_top1,
                               'best_acc_top5': best_acc_top5,
                               'architecture': args.design},
                "state_dict": model.state_dict(),
            },
            args.save,
        )
        if args.use_wandb and wandb_con is not None:
            print(f'{args.save}/f_model.pth')
            wandb_art.add_file(f'{args.save}/f_model.pth')
            wandb_con.log_artifact(wandb_art)

    return [train_acc, train_obj, avg_top1_val, avg_top5_val, valid_obj]


def infer_tv(
    valid_queue, model, criterion, args, report_freq=100, lr_tta=True, fast=False
):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        if fast and step > 0:
            break
        with torch.no_grad():
            input, target = input.to(args.local_rank), target.to(args.local_rank)
            logits = model(input.contiguous())
            if lr_tta:
                sss = model(torch.flip(input, dims=[3]).contiguous())
                logits += sss
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
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
                "VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds",
                step,
                objs.avg,
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

    loss_rank = torch.tensor(objs.avg).to(args.local_rank)
    if args.distributed:
        loss_list = [torch.zeros_like(loss_rank) for r in range(args.world_size)]
        dist.all_gather(loss_list, loss_rank)
        avg_loss = torch.mean(torch.stack(loss_list))
    else:
        avg_loss = loss_rank
    return avg_top1_val, avg_top5_val, avg_loss
