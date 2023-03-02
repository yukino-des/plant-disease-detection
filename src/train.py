import datetime
import numpy as np
import os
import random
import torch
import sys
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from xml.etree import ElementTree as ET

sys.path.append(os.path.dirname(sys.path[0]))
from utils.util import download_weights, get_anchors, get_classes, print_table, show_config, EvalCallback, LossHistory
from utils.yolo import (fit_one_epoch, get_lr_scheduler, set_optimizer_lr, weights_init, yolo_dataset_collate,
                        YoloBody, YoloDataset, YOLOLoss)

if __name__ == "__main__":
    random.seed(0)
    classes_path = "../data/classes.txt"
    trainval_percent = 0.9
    train_percent = 0.9
    classes, _ = get_classes(classes_path)
    photo_nums = np.zeros(2)
    nums = np.zeros(len(classes))
    annotations = "../VOC/Annotations"
    image_sets = "../VOC/ImageSets/Main"
    os.makedirs(image_sets, exist_ok=True)
    temp_xml = os.listdir(annotations)
    total_xml = []
    for xml in temp_xml:
        if xml.endswith(".xml"):
            total_xml.append(xml)
    num = len(total_xml)
    num_list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(num_list, tv)
    train = random.sample(trainval, tr)
    ftrainval = open(os.path.join(image_sets, "trainval.txt"), "w")
    ftest = open(os.path.join(image_sets, "test.txt"), "w")
    ftrain = open(os.path.join(image_sets, "train.txt"), "w")
    fval = open(os.path.join(image_sets, "val.txt"), "w")
    for i in num_list:
        name = total_xml[i][:-4] + "\n"
        if i in trainval:
            ftrainval.write(name)
            ftrain.write(name) if i in train else fval.write(name)
        else:
            ftest.write(name)
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    type_index = 0
    for image_set in ["train", "val"]:
        image_ids = open("../VOC/ImageSets/Main/%s.txt" % image_set, encoding="utf-8").read().strip().split()
        list_file = open("%s.txt" % image_set, "w", encoding="utf-8")
        for image_id in image_ids:
            list_file.write("../VOC/JPEGImages/%s.jpg" % image_id)
            in_file = open("../VOC/Annotations/%s.xml" % image_id, encoding="utf-8")
            tree = ET.parse(in_file)
            root = tree.getroot()
            for obj in root.iter("object"):
                difficult = 0
                if obj.find("difficult") is not None:
                    difficult = obj.find("difficult").text
                cls = obj.find("name").text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find("bndbox")
                b = (int(float(xmlbox.find("xmin").text)), int(float(xmlbox.find("ymin").text)),
                     int(float(xmlbox.find("xmax").text)), int(float(xmlbox.find("ymax").text)))
                list_file.write(" " + ",".join([str(a) for a in b]) + "," + str(cls_id))
                nums[classes.index(cls)] = nums[classes.index(cls)] + 1
            list_file.write("\n")
        photo_nums[type_index] = len(image_ids)
        type_index += 1
        list_file.close()
    str_nums = [str(int(x)) for x in nums]
    tableData = [classes, str_nums]
    colWidths = [0] * len(tableData)
    for i in range(len(tableData)):
        for j in range(len(tableData[i])):
            if len(tableData[i][j]) > colWidths[i]:
                colWidths[i] = len(tableData[i][j])
    print_table(tableData, colWidths)
    if photo_nums[0] <= 500:
        raise ValueError("Dataset not qualified.")
    if np.sum(nums) == 0:
        raise ValueError(f"{classes_path} error.")
    anchors_path = "../data/anchors.txt"
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    model_path = "../data/pretrain.pth"  # update `model_path = "../data/best.pth"`,
    init_epoch = 0  # `init_epoch` if training is interrupted
    lr_decay_type = "cos"  # ["cos", "step"]

    optimizer_type = "adam"
    unfreeze_epoch = 100
    init_lr = 1e-3
    weight_decay = 0
    """
    optimizer_type = "sgd"
    unfreeze_epoch = 300
    init_lr = 1e-2
    weight_decay = 5e-4
    """

    input_shape = [416, 416]
    pretrained = False
    mosaic = True
    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio = 0.7
    freeze_epoch = 50
    freeze_batch_size = 16
    unfreeze_batch_size = 8
    freeze_train = True
    min_lr = init_lr * 0.01
    momentum = 0.937
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
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)
    model = YoloBody(anchors_mask, num_classes, pretrained=pretrained)
    if pretrained:
        download_weights()
    else:
        weights_init(model)
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
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, cuda, anchors_mask, focal_loss, focal_alpha, focal_gamma)
    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M%S")
    log_dir = os.path.join(save_dir, "loss" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)
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
    show_config({"classes_path": classes_path,
                 "anchors_path": anchors_path,
                 "anchors_mask": anchors_mask,
                 "model_path": model_path,
                 "input_shape": input_shape,
                 "init_epoch": init_epoch,
                 "freeze_epoch": freeze_epoch,
                 "unfreeze_epoch": unfreeze_epoch,
                 "freeze_batch_size": freeze_batch_size,
                 "unfreeze_batch_size": unfreeze_batch_size,
                 "freeze_train": freeze_train,
                 "init_lr": init_lr,
                 "min_lr": min_lr,
                 "optimizer_type": optimizer_type,
                 "momentum": momentum,
                 "lr_decay_type": lr_decay_type,
                 "save_period": save_period,
                 "save_dir": save_dir,
                 "num_workers": num_workers,
                 "num_train": num_train,
                 "num_val": num_val,
                 "cuda": torch.cuda.is_available()})
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step = num_train // unfreeze_batch_size * unfreeze_epoch
    if total_step <= wanted_step and num_train // unfreeze_batch_size == 0:
        raise ValueError("Dataset not qualified.")
    ctn = input("Continue (y/n)? ")
    if ctn != "y" and ctn != "Y":
        exit(0)
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
    optimizer = {
        "adam": optim.Adam(pg0, init_lr_fit, betas=(momentum, 0.999)),
        "adamw": optim.AdamW(pg0, init_lr_fit, betas=(momentum, 0.999)),
        "sgd": optim.SGD(pg0, init_lr_fit, momentum=momentum, nesterov=True)
    }[optimizer_type]
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
                             pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate,
                             sampler=train_sampler)
            gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate,
                                 sampler=val_sampler)
            unfreeze_flag = True
        gen.dataset.epoch_now = epoch
        gen_val.dataset.epoch_now = epoch
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                      epoch_step_val, gen, gen_val, unfreeze_epoch, cuda, save_period, save_dir)
    loss_history.writer.close()
