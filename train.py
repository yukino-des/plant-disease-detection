import numpy as np
import os
import torch
from datetime import datetime
from model import EvalCallback, LossHistory, YoloBody, YoloDataset, YoloLoss
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from utils import (fit1epoch, get_anchors, get_classes, get_lr_scheduler, get_txts, set_optimizer_lr, print_config,
                   yolo_dataset_collate)

if __name__ == "__main__":
    # 如果训练中断，请修改：model_path = "data/current.pth"
    model_path = ""
    # 如果训练中断，请修改：init_epoch = 已训练的epoch数量
    init_epoch = 0
    get_txts(0, 0.9, 0.9)
    flag = input("Start training (y/n)? ")
    if flag != "y" and flag != "Y":
        exit(0)
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # lr_decay_type = "step"
    lr_decay_type = "cos"
    # -----------------------------------------------------------------------
    # | optimizer_type, freeze_train, unfreeze_epoch, init_lr, weight_decay |
    # |         "adam",         True,            100,    1e-3,            0 |
    # |         "adam",        False,            100,    1e-3,            0 |
    # |          "sgd",         True,            300,    1e-2,         5e-4 |
    # |          "sgd",        False,            300,    1e-2,         5e-4 |
    # -----------------------------------------------------------------------
    optimizer_type, freeze_train, unfreeze_epoch, init_lr, weight_decay = "adam", True, 100, 1e-3, 0
    input_shape = [416, 416]
    mosaic = True  # 是否使用马赛克数据增强
    mixup = True  # 是否使用mixup数据增强
    focal_loss = False  # 是否使用focal loss平衡正负样本
    eval_flag = True  # 是否在训练时对验证集进行评估
    mosaic_prob = 0.5
    mixup_prob = 0.5
    special_aug_ratio = 0.7
    freeze_epoch = 50
    freeze_batch_size = 16
    unfreeze_batch_size = 8
    min_lr = init_lr * 0.01
    momentum = 0.937
    focal_alpha = 0.25
    focal_gamma = 2
    save_period = 10
    eval_period = 10
    num_workers = 2
    ngpus_per_node = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names, num_classes = get_classes("data/classes.txt")
    anchors, num_anchors = get_anchors("data/anchors.txt")
    model = YoloBody(anchors_mask, num_classes)
    if model_path != "":
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        temp_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
    cuda = torch.cuda.is_available()
    yolo_loss = YoloLoss(anchors, num_classes, input_shape, cuda, anchors_mask, focal_loss, focal_alpha, focal_gamma)
    time_str = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
    log_dir = os.path.join("data/cache/loss", time_str)
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    model_train = model.train()
    if cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    with open("data/train.txt", encoding="utf-8") as f:
        train_lines = f.readlines()
    with open("data/val.txt", encoding="utf-8") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    print_config(classes_path="data/classes.txt", anchors_path="data/anchors.txt", anchors_mask=anchors_mask,
                 model_path=model_path, input_shape=input_shape, init_epoch=init_epoch, freeze_epoch=freeze_epoch,
                 unfreeze_epoch=unfreeze_epoch, freeze_batch_size=freeze_batch_size,
                 unfreeze_batch_size=unfreeze_batch_size, freeze_train=freeze_train, init_lr=init_lr, min_lr=min_lr,
                 optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type, save_period=save_period,
                 save_dir="data", num_workers=num_workers, num_train=num_train, num_val=num_val,
                 cuda=torch.cuda.is_available())
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step = num_train // unfreeze_batch_size * unfreeze_epoch
    if total_step <= wanted_step and num_train // unfreeze_batch_size == 0:
        raise ValueError("Dataset not qualified.")
    unfreeze_flag = False
    if freeze_train:
        for param in model.backbone.parameters():
            param.requires_grad = False
    batch_size = freeze_batch_size if freeze_train else unfreeze_batch_size
    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type in ["adam", "adamw"] else 5e-2
    lr_limit_min = 3e-4 if optimizer_type in ["adam", "adamw"] else 5e-4
    init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    optimizer = {"adam": optim.Adam(pg0, init_lr_fit, betas=(momentum, 0.999)),
                 "adamw": optim.AdamW(pg0, init_lr_fit, betas=(momentum, 0.999)),
                 "sgd": optim.SGD(pg0, init_lr_fit, momentum=momentum, nesterov=True)}[optimizer_type]
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, unfreeze_epoch)
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("Dataset not qualified.")
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
    eval_callback = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines,
                                 log_dir, cuda, eval_flag=eval_flag, period=eval_period)
    for epoch in range(init_epoch, unfreeze_epoch):
        if epoch >= freeze_epoch and not unfreeze_flag and freeze_train:
            batch_size = unfreeze_batch_size
            nbs = 64
            lr_limit_max = 1e-3 if optimizer_type in ["adam", "adamw"] else 5e-2
            lr_limit_min = 3e-4 if optimizer_type in ["adam", "adamw"] else 5e-4
            init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
            min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, unfreeze_epoch)
            for param in model.backbone.parameters():
                param.requires_grad = True
            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size
            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("Dataset not qualified.")
            gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
            gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)
            unfreeze_flag = True
        gen.dataset.epoch_now = epoch
        gen_val.dataset.epoch_now = epoch
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit1epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, unfreeze_epoch, cuda, save_period)
    loss_history.writer.close()
