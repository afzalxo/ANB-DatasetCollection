import time
import torch
import logging
import torch.distributed as dist


from timm.models import model_parameters
import auxiliary.utils as utils


def train_epoch_gpu(
    epoch,
    train_queue,
    model,
    model_ema,
    criterion,
    optimizer,
    lr_schedule,
    amp_autocast,
    loss_scaler,
    mixup_fn,
    report_freq,
    args
):
    losses_m = utils.AverageMeter()
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    model.train()
    end = time.time()
    num_batches_per_epoch = len(train_queue)
    last_idx = num_batches_per_epoch - 1
    num_updates = epoch * num_batches_per_epoch
    for step, (input, target) in enumerate(train_queue):
        last_batch = step == last_idx
        data_time_m.update(time.time() - end)
        if mixup_fn is not None:
            input, target = mixup_fn(input, target)
        with amp_autocast():
            logits = model(input)
            loss = criterion(logits, target)
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=None,
                clip_mode='norm',
                parameters=model_parameters(model, exclude_head=False),
                create_graph=False
            )
        else:
            raise NotImplementedError('Have to use loss_scaler for now')

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()

        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or step % report_freq == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
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
        if lr_schedule is not None:
            lr_schedule.step_update(num_updates=num_updates, metric=losses_m.avg)
        end = time.time()
        del loss, logits, target
    return 0.0, losses_m.avg


def reduce_gpu_tensor(tens, world_size):
    clone = tens.clone()
    dist.all_reduce(dist.op.REDUCE_SUM, clone)
    reduced = clone / world_size
    return reduced


def infer_gpu(
    valid_queue, model, criterion, args, epoch, amp_autocast, report_freq=50
):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    batch_time_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(valid_queue) - 1
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            last_batch = step == last_idx
            with amp_autocast():
                logits = model(input)
                loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                prec1 = utils.reduce_tensor(prec1, args.world_size)
                prec5 = utils.reduce_tensor(prec5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            objs.update(reduced_loss.item(), input.size(0))
            top1.update(prec1.item(), logits.size(0))
            top5.update(prec5.item(), logits.size(0))
            
            batch_time_m.update(time.time() - end)
            if args.rank == 0 and (last_batch or step % report_freq == 0):
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
    return top1.avg, top5.avg, objs.avg


def train_x_epochs_gpu(
    _epochs,
    lr_scheduler,
    dset_train,
    train_queue,
    valid_queue,
    model,
    model_ema,
    criterion_train,
    criterion_val,
    optimizer,
    mixup_fn,
    amp_autocast,
    loss_scaler,
    args,
):
    train_sttime = time.time()
    best_acc_top1 = 0
    best_acc_top5 = 0
    best_model_sd = {k: v.cpu() for k,v in model.state_dict().items()}
    valid_acc_top1, valid_acc_top5, valid_obj = None, None, None

    for epoch in range(_epochs):
        if hasattr(dset_train, 'set_epoch'):
            dset_train.set_epoch(epoch)
        elif args.distributed and hasattr(train_queue.sampler, 'set_epoch'):
            train_queue.sampler.set_epoch(epoch)
        epoch_start = time.time()
        train_acc, train_obj = train_epoch_gpu(
            epoch,
            train_queue,
            model,
            model_ema,
            criterion_train,
            optimizer,
            lr_scheduler,
            amp_autocast,
            loss_scaler,
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
        # validation
        if epoch > -1:
            valid_acc_top1, valid_acc_top5, valid_obj = infer_gpu(
                valid_queue,
                model,
                criterion_val,
                args,
                epoch,
                amp_autocast,
                args.report_freq,
            )
            avg_top1_val, avg_top5_val = valid_acc_top1, valid_acc_top5

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed:
                    utils.distribute_bn(model_ema, args.world_size, reduce=True)
                avg_top1_val_ema, avg_top5_val_ema, valid_obj_ema = infer_gpu(
                        valid_queue,
                        model_ema.module,
                        criterion_val,
                        args,
                        epoch,
                        amp_autocast
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
                commit = True
                args.wandb_con.log(
                    {
                        "ema_valid_top1": avg_top1_val_ema,
                        "ema_valid_top5": avg_top5_val_ema,
                        "ema_v_loss": valid_obj_ema,
                    },
                    commit=commit,
                )
            if model_ema is not None and not args.model_ema_force_cpu:
                avg_top1_val, avg_top5_val, valid_obj = avg_top1_val_ema, avg_top5_val_ema, valid_obj_ema
            if avg_top5_val > best_acc_top5:
                best_acc_top5 = avg_top5_val
            if avg_top1_val > best_acc_top1:
                best_acc_top1 = avg_top1_val
                best_model_sd = {k: v.cpu() for k,v in model.state_dict().items()}
                best_model_ema_sd = {k: v.cpu() for k,v in model_ema.state_dict().items()}
                utils.save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": best_model_sd,
                    "ema_state_dict": best_model_ema_sd
                },
                args.save,
                )

            logging.info(
                "Epoch %d, Valid_acc_top1 %f, Valid_acc_top5 %f, Best_top1 %f, Best_top5 %f",
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
                name=f"deploymodel-gpu-jobid{args.job_id}-model{args.model_num}",
                type="model",
                metadata={
                    "training_config": training_config_dict,
                    "model_metadata": model_metadata_dict,
                },
            )
            wandb_art.add_file(f"{args.save}/f_model.pth")
            args.wandb_con.log_artifact(wandb_art)

    return [train_acc, train_obj, avg_top1_val, avg_top5_val, valid_obj]
