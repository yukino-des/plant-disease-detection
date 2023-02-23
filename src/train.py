import datetime
import os

import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from utils.yolo_v4 import YoloBody
from utils.yolo_training import YOLOLoss, get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import download_weights, get_anchors, get_classes, show_config
from utils.utils_fit import fit_one_epoch


def main():
    # todo update `Cuda = True`
    Cuda = False
    distributed = False
    fp16 = False
    classes_path = "../data/voc_classes.txt"
    anchors_path = "../data/yolo_anchors.txt"
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # todo update `model_path = "../data/best_epoch_weights.pth"` if training is interrupted
    model_path = "../data/mobilenetv2_yolov4_voc.pth"
    input_shape = [416, 416]
    backbone = "mobilenetv2"
    pretrained = False
    mosaic = True
    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio = 0.7
    label_smoothing = 0
    # `optimizer_type = "adam"`
    Init_Epoch = 0  # update `Init_Epoch` if training is interrupted
    Freeze_Epoch = 50
    Freeze_batch_size = 16  # 8
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 8
    Freeze_Train = True
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.937
    weight_decay = 0
    lr_decay_type = "cos"  # "step"
    # `optimizer_type = "sgd"`
    """
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 16
    UnFreeze_Epoch = 300
    Unfreeze_batch_size = 8
    Freeze_Train = True
    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01
    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-4
    lr_decay_type = "cos"  # "step"
    """
    focal_loss = False
    focal_alpha = 0.25
    focal_gamma = 2
    save_period = 10
    save_dir = "data"
    eval_flag = True
    eval_period = 10
    num_workers = 2
    train_annotation_path = "train.txt"
    val_annotation_path = "val.txt"
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count:", ngpus_per_node)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)
    model = YoloBody(anchors_mask, num_classes, backbone=backbone, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != "":
        if local_rank == 0:
            print("Load weights {}.".format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccess:", str(load_key)[:500], "\nSuccess Num:", len(load_key))
            print("\nFail:", str(no_load_key)[:500], "\nFail Num:", len(no_load_key))
    yolo_loss = YOLOLoss(anchors,
                         num_classes,
                         input_shape,
                         Cuda,
                         anchors_mask,
                         label_smoothing,
                         focal_loss,
                         focal_alpha,
                         focal_gamma)
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None
    model_train = model.train()
    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train,
                                                                    device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    with open(train_annotation_path, encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding="utf-8") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    if local_rank == 0:
        show_config(classes_path=classes_path,
                    anchors_path=anchors_path,
                    anchors_mask=anchors_mask,
                    model_path=model_path,
                    input_shape=input_shape,
                    Init_Epoch=Init_Epoch,
                    Freeze_Epoch=Freeze_Epoch,
                    UnFreeze_Epoch=UnFreeze_Epoch,
                    Freeze_batch_size=Freeze_batch_size,
                    Unfreeze_batch_size=Unfreeze_batch_size,
                    Freeze_Train=Freeze_Train,
                    Init_lr=Init_lr,
                    Min_lr=Min_lr,
                    optimizer_type=optimizer_type,
                    momentum=momentum,
                    lr_decay_type=lr_decay_type,
                    save_period=save_period,
                    save_dir=save_dir,
                    num_workers=num_workers,
                    num_train=num_train,
                    num_val=num_val)
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError("The dataset is too small.")
    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type in ["adam", "adamw"] else 5e-2
        lr_limit_min = 3e-4 if optimizer_type in ["adam", "adamw"] else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            "adam": optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            "adamw": optim.AdamW(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            "sgd": optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small.")
        train_dataset = YoloDataset(train_lines,
                                    input_shape,
                                    num_classes,
                                    epoch_length=UnFreeze_Epoch,
                                    mosaic=mosaic,
                                    mixup=mixup,
                                    mosaic_prob=mosaic_prob,
                                    mixup_prob=mixup_prob,
                                    train=True,
                                    special_aug_ratio=special_aug_ratio)
        val_dataset = YoloDataset(val_lines,
                                  input_shape,
                                  num_classes,
                                  epoch_length=UnFreeze_Epoch,
                                  mosaic=False,
                                  mixup=False,
                                  mosaic_prob=0,
                                  mixup_prob=0,
                                  train=False,
                                  special_aug_ratio=0)
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
        gen = DataLoader(train_dataset,
                         shuffle=shuffle,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True,
                         collate_fn=yolo_dataset_collate,
                         sampler=train_sampler)
        gen_val = DataLoader(val_dataset,
                             shuffle=shuffle,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True,
                             collate_fn=yolo_dataset_collate,
                             sampler=val_sampler)
        if local_rank == 0:
            eval_callback = EvalCallback(model,
                                         input_shape,
                                         anchors,
                                         anchors_mask,
                                         class_names,
                                         num_classes,
                                         val_lines,
                                         log_dir,
                                         Cuda,
                                         eval_flag=eval_flag,
                                         period=eval_period)
        else:
            eval_callback = None
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type in ["adam", "adamw"] else 5e-2
                lr_limit_min = 3e-4 if optimizer_type in ["adam", "adamw"] else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                for param in model.backbone.parameters():
                    param.requires_grad = True
                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small.")
                if distributed:
                    batch_size = batch_size // ngpus_per_node
                gen = DataLoader(train_dataset,
                                 shuffle=shuffle,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True,
                                 collate_fn=yolo_dataset_collate,
                                 sampler=train_sampler)
                gen_val = DataLoader(val_dataset,
                                     shuffle=shuffle,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True,
                                     collate_fn=yolo_dataset_collate,
                                     sampler=val_sampler)
                UnFreeze_flag = True
            if distributed:
                train_sampler.set_epoch(epoch)
            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model_train,
                          model,
                          yolo_loss,
                          loss_history,
                          eval_callback,
                          optimizer,
                          epoch,
                          epoch_step,
                          epoch_step_val,
                          gen,
                          gen_val,
                          UnFreeze_Epoch,
                          Cuda,
                          fp16,
                          scaler,
                          save_period,
                          save_dir,
                          local_rank)
            if distributed:
                dist.barrier()
        if local_rank == 0:
            loss_history.writer.close()
