from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from model import LossHistory, YoloBody, YoloDataset, YoloLoss
from utils import (fit1epoch, get_anchors, get_classes, get_lr_scheduler, get_txt, set_optimizer_lr, print_config,
                   weights_init, yolo_dataset_collate)

if __name__ == "__main__":
    # 如果训练中断，请修改model_path = "data/cache/loss/current.pth"
    model_path = "data/pretrain.pth"
    # 如果训练中断，请修改init_epoch = 已训练的epoch数
    init_epoch = 0
    get_txt(0, 0.9, 0.9)
    lr_decay_type = "cos"  # "step"
    # ---------------------------------------------------------------
    # |  optimizer_type, freeze_train, epoch, init_lr, weight_decay |
    # | "adam" "adam_w",         True,   300,    1e-3,            0 |
    # | "adam" "adam_w",        False,   300,    1e-3,            0 |
    # |           "sgd",         True,   500,    1e-2,         5e-4 |
    # |           "sgd",        False,   500,    1e-2,         5e-4 |
    # ---------------------------------------------------------------
    optimizer_type, freeze_train, epoch, init_lr, weight_decay = "adam", True, 100, 1e-3, 0
    # 多线程训练
    num_workers = 2  # 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names, num_classes = get_classes("data/classes.txt")
    anchors, num_anchors = get_anchors("data/anchors.txt")
    model = YoloBody(num_classes)
    weights_init(model)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    temp_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    yolo_loss = YoloLoss(anchors, num_classes, True, 0.25, 2)
    time_str = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
    log_dir = f"data/cache/loss/{time_str}"
    loss_history = LossHistory(log_dir, model)
    model_train = model.train()
    if torch.cuda.is_available():
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    with open("data/train.txt", encoding="utf-8") as f:
        train_lines = f.readlines()
    with open("data/val.txt", encoding="utf-8") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    print_config(classes_path="data/classes.txt",
                 anchors_path="data/anchors.txt",
                 save_dir="data/cache",
                 anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 model_path=model_path,  # ["", "data/cache/loss/current.pth"]
                 input_shape=[416, 416],  # [416, 416]
                 init_epoch=init_epoch,  # [0, ...]
                 epoch=epoch,  # [300, 500]
                 freeze_train=freeze_train,
                 init_lr=init_lr,  # [1e-3, 1e-2]
                 optimizer_type=optimizer_type,  # ["adam", "adam_w", "step"]
                 lr_decay_type=lr_decay_type,  # ["cos", "step"]
                 num_workers=num_workers,  # [2, 4]
                 num_train=num_train,
                 num_val=num_val,
                 cuda=torch.cuda.is_available())  # [True, False]
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step = num_train // 8 * epoch
    if total_step <= wanted_step and num_train // 8 == 0:
        raise ValueError("数据集不合格")
    unfreeze_flag = False
    if freeze_train:
        for param in model.backbone.parameters():
            param.requires_grad = False
    batch_size = 16 if freeze_train else 8
    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type in ["adam", "adam_w"] else 5e-2
    lr_limit_min = 3e-4 if optimizer_type in ["adam", "adam_w"] else 5e-4
    init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(batch_size / nbs * init_lr * 0.01, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    optimizer = {"adam": optim.Adam(pg0, init_lr_fit, betas=(0.937, 0.999)),
                 "adam_w": optim.AdamW(pg0, init_lr_fit, betas=(0.937, 0.999)),
                 "sgd": optim.SGD(pg0, init_lr_fit, momentum=0.937, nesterov=True)}[optimizer_type]
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, epoch)
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集不合格")
    train_dataset = YoloDataset(train_lines, num_classes, epoch, True, True, 0.5, 0.5, True, 0.7)
    val_dataset = YoloDataset(val_lines, num_classes, epoch, False, False, 0, 0, False, 0)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate, sampler=None)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate, sampler=None)
    for e in range(init_epoch, epoch):
        if e >= 50 and not unfreeze_flag and freeze_train:
            batch_size = 8
            nbs = 64
            lr_limit_max = 1e-3 if optimizer_type in ["adam", "adam_w"] else 5e-2
            lr_limit_min = 3e-4 if optimizer_type in ["adam", "adam_w"] else 5e-4
            init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
            min_lr_fit = min(max(batch_size / nbs * init_lr * 0.01, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, epoch)
            for param in model.backbone.parameters():
                param.requires_grad = True
            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size
            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集不合格")
            gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, sampler=None)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, sampler=None)
            unfreeze_flag = True
        gen.dataset.epoch_now = e
        gen_val.dataset.epoch_now = e
        set_optimizer_lr(optimizer, lr_scheduler_func, e)
        fit1epoch(model_train, model, yolo_loss, loss_history, optimizer, e, epoch_step, epoch_step_val, gen, gen_val,
                  epoch, 10)
    loss_history.writer.close()
