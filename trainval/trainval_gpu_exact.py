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

        batch_time.update(time.time() - b_start)
        num_updates += 1
        if last_batch or step % report_freq == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            '''
            loss_clone = loss.data.clone()
            xm.all_reduce(xm.REDUCE_SUM, loss_clone)
            reduced_loss = loss_clone / args.world_size
            # reduced_loss = reduce_xla_tensor(loss.data, args.world_size)
            losses_m.update(reduced_loss.item(), input.size(0))
            '''
            loss_val = loss.data.item()
            losses_m.update(loss_val, input.size(0))
            print(f'Loss: {loss_val:.4f}')
        if lr_schedule is not None:
            lr_schedule.step_update(num_updates=num_updates, metric=losses_m.avg)
        del loss, logits, target
    return top1.avg, losses_m.avg


def reduce_gpu_tensor(tens, world_size):
    clone = tens.clone()
    dist.all_reduce(dist.op.REDUCE_SUM, clone)
    reduced = clone / world_size
    return reduced


def infer_gpu(
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
                # xm.add_step_closure(test_utils.print_test_update, args=(args.local_rank, top1.avg, epoch, step))
                #TODO print stats here
                print(top1.avg, top5.avg, objs.avg)
            del loss, logits, target
        top1_acc = top1.avg
        top5_acc = top5.avg
        loss = objs.avg
        '''
        accuracy_top1 = xm.mesh_reduce('top1_accuracy', top1_acc, np.mean)
        accuracy_top5 = xm.mesh_reduce('top5_accuracy', top5_acc, np.mean)
        loss_avg = xm.mesh_reduce('loss', loss, np.mean)
        '''
        # accuracy_top1, accuracy_top5, loss_avg
    return top1.avg, top5.avg, objs.avg


def train_x_epochs_gpu(
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
    amp_autocast,
    loss_scaler,
    args,
):
    train_sttime = time.time()
    best_acc_top1 = 0
    best_acc_top5 = 0
    best_model_sd = model.state_dict()
    valid_acc_top1, valid_acc_top5, valid_obj = None, None, None
    sampler = train_queue.sampler

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
            #commit = True if epoch <= _epochs - 4 else False
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
