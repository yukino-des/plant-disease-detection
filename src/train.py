import datetime
import numpy as np
import os
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from utils.util import download_weights, get_anchors, get_classes, show_config, EvalCallback, LossHistory
from utils.yolo import (fit_one_epoch, get_lr_scheduler, set_optimizer_lr, weights_init, yolo_dataset_collate,
                        YoloBody, YoloDataset, YOLOLoss)

if __name__ == '__main__':
    # todo update `cuda = True`
    cuda = False
    classes_path = "../data/classes.txt"
    anchors_path = "../data/anchors.txt"
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # todo update `model_path = "../data/best.pth"` if training is interrupted
    model_path = "../data/best.pth"
    input_shape = [416, 416]
    backbone = "mobilenetv2"
    pretrained = False
    mosaic = True
    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio = 0.7
    # ---------- `optimizer_type = "adam"` ----------
    Init_Epoch = 0  # update `Init_Epoch` if training is interrupted
    Freeze_Epoch = 50
    Freeze_batch_size = 16  # 8
    unfreeze_epoch = 100
    Unfreeze_batch_size = 8
    Freeze_Train = True
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.937
    weight_decay = 0
    lr_decay_type = "cos"  # "step"
    # ---------- `optimizer_type = "sgd"` ----------
    """
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 16
    unfreeze_epoch = 300
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
    save_dir = "../data"
    eval_flag = True
    eval_period = 10
    num_workers = 2
    train_annotation_path = "train.txt"
    val_annotation_path = "val.txt"
    ngpus_per_node = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = 0
    if pretrained:
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
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, cuda, anchors_mask, focal_loss, focal_alpha, focal_gamma)
    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M%S")
    log_dir = os.path.join(save_dir, "loss" + str(time_str))
    if local_rank == 0:
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None
    model_train = model.train()
    if cuda:
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
        show_config(classes_path=classes_path, anchors_path=anchors_path, anchors_mask=anchors_mask,
                    model_path=model_path, input_shape=input_shape, Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch,
                    unfreeze_epoch=unfreeze_epoch, Freeze_batch_size=Freeze_batch_size,
                    Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, Init_lr=Init_lr, Min_lr=Min_lr,
                    optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type,
                    save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train,
                    num_val=num_val)
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step = num_train // Unfreeze_batch_size * unfreeze_epoch
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
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, unfreeze_epoch)
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small.")
        train_dataset = YoloDataset(train_lines, input_shape, num_classes, epoch_length=unfreeze_epoch, mosaic=mosaic,
                                    mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True,
                                    special_aug_ratio=special_aug_ratio)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, epoch_length=unfreeze_epoch, mosaic=False,
                                  mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        train_sampler = None
        val_sampler = None
        shuffle = True
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines,
                                         log_dir, cuda, eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None
        for epoch in range(Init_Epoch, unfreeze_epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type in ["adam", "adamw"] else 5e-2
                lr_limit_min = 3e-4 if optimizer_type in ["adam", "adamw"] else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, unfreeze_epoch)
                for param in model.backbone.parameters():
                    param.requires_grad = True
                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small.")
                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate,
                                 sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate,
                                     sampler=val_sampler)
                UnFreeze_flag = True
            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                          epoch_step_val, gen, gen_val, unfreeze_epoch, cuda, save_period, save_dir, local_rank)
        if local_rank == 0:
            loss_history.writer.close()
