import glob
import json
import math
import matplotlib
import numpy as np
import operator
import os
import random
import shutil
import torch
from collections import OrderedDict
from functools import partial
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from tqdm import tqdm
from xml.etree import ElementTree as ET

if os.name == "nt":
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def avg_iou(box, cluster):
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def cas_iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])
    intersection = x * y
    area1 = box[0] * box[1]
    area2 = cluster[:, 0] * cluster[:, 1]
    return intersection / (area1 + area2 - intersection)


# 标准卷积
def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride,
                                                         padding=pad, groups=groups, bias=False)),  # 3x3标准卷积
                                      ("bn", nn.BatchNorm2d(filter_out)),  # 批量归一化
                                      ("relu", nn.ReLU6(inplace=True))]))  # ReLU6激活


# 深度可分离卷积
def conv_dw(filter_in, filter_out, stride=1):
    return nn.Sequential(nn.Conv2d(filter_in, filter_in, kernel_size=3, stride=stride, padding=1, groups=filter_in,
                                   bias=False),  # 3x3深度卷积
                         nn.BatchNorm2d(filter_in),  # 批量归一化
                         nn.ReLU6(inplace=True),  # ReLU6激活
                         nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),  # 1x1点卷积
                         nn.BatchNorm2d(filter_out),  # 批量归一化
                         nn.ReLU6(inplace=True))  # ReLU6激活


def cvt_color(image):
    if len(np.shape(image)) != 3 or np.shape(image)[2] != 3:
        image = image.convert("RGB")
    return image


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, plot_color):
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    plt.barh(range(n_classes), sorted_values, color=plot_color)
    fig = plt.gcf()
    axes = plt.gca()
    r = fig.canvas.get_renderer()
    for i, val in enumerate(sorted_values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {:.2f}".format(val)
        t = plt.text(val, i, str_val, color=plot_color, va="center", fontweight="bold")
        if i == (len(sorted_values) - 1):
            adjust_axes(r, t, fig, axes)
    fig.canvas.manager.set_window_title(window_title)
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    init_height = fig.get_figheight()
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)
    height_in = height_pt / dpi
    top_margin = 0.15
    bottom_margin = 0.05
    figure_height = height_in / (1 - top_margin - bottom_margin)
    if figure_height > init_height:
        fig.set_figheight(figure_height)
    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_label, fontsize="large")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close()


def file_lines_to_list(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def fit1epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
              gen, gen_val, unfreeze_epoch, cuda, save_period, save_dir):
    loss = 0
    val_loss = 0
    pbar = tqdm(total=epoch_step, desc=f"epoch {epoch + 1}/{unfreeze_epoch}", postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(0)
                targets = [ann.cuda(0) for ann in targets]
        optimizer.zero_grad()
        outputs = model_train(images)
        loss_sum = 0
        for l in range(len(outputs)):
            loss_item = yolo_loss(l, outputs[l], targets)
            loss_sum += loss_item
        loss_value = loss_sum
        loss_value.backward()
        optimizer.step()
        loss += loss_value.item()
        pbar.set_postfix(**{"loss": loss / (iteration + 1), "lr": get_lr(optimizer)})
        pbar.update(1)
    pbar.close()
    pbar = tqdm(total=epoch_step_val, desc=f"epoch {epoch + 1}/{unfreeze_epoch}", postfix=dict, mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(0)
                targets = [ann.cuda(0) for ann in targets]
            optimizer.zero_grad()
            outputs = model_train(images)
            loss_value_all = 0
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all
        val_loss += loss_value.item()
        pbar.set_postfix(**{"val_loss": val_loss / (iteration + 1)})
        pbar.update(1)
    pbar.close()
    loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
    eval_callback.on_epoch_end(epoch + 1, model_train)
    print("epoch: " + str(epoch + 1) + "/" + str(unfreeze_epoch))
    print("total loss: %.3f; val loss: %.3f" % (loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == unfreeze_epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, "epoch%03d-loss%.3f-val_loss%.3f.pth" % (
            epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print("data/best.pth saved.")
        torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
    torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))


def get_anchors(anchors_path):
    with open(anchors_path, encoding="utf-8") as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(",")]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


def get_classes(classes_path):
    with open(classes_path, encoding="utf-8") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)
    return func


def get_map(min_overlap, draw_plot, score_threshold=0.5, path="tmp/maps"):
    gt_path = os.path.join(path, ".gt")  # ground_truth
    dr_path = os.path.join(path, ".dr")  # detection_results
    temp_files_path = os.path.join(path, ".temp_files")
    results_files_path = os.path.join(path, "results")
    os.makedirs(temp_files_path, exist_ok=True)
    if os.path.exists(results_files_path):
        shutil.rmtree(results_files_path)
    else:
        os.makedirs(results_files_path)
    if draw_plot:
        os.makedirs(os.path.join(results_files_path, "AP"))
        os.makedirs(os.path.join(results_files_path, "F1"))
        os.makedirs(os.path.join(results_files_path, "recall"))
        os.makedirs(os.path.join(results_files_path, "precision"))
    ground_truth_files_list = glob.glob(gt_path + "/*.txt")
    if len(ground_truth_files_list) == 0:
        raise FileNotFoundError("Ground-truth files not found error.")
    ground_truth_files_list.sort()
    gt_counter_per_class = {}
    counter_images_per_class = {}
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        temp_path = os.path.join(dr_path, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"File not found error: {temp_path}.\n")
        lines_list = file_lines_to_list(txt_file)
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except:
                if "difficult" in line:
                    line_split = line.split()
                    _difficult = line_split[-1]
                    bottom = line_split[-2]
                    right = line_split[-3]
                    top = line_split[-4]
                    left = line_split[-5]
                    class_name = ""
                    for name in line_split[:-5]:
                        class_name += name + " "
                    class_name = class_name[:-1]
                    is_difficult = True
                else:
                    line_split = line.split()
                    bottom = line_split[-1]
                    right = line_split[-2]
                    top = line_split[-3]
                    left = line_split[-4]
                    class_name = ""
                    for name in line_split[:-4]:
                        class_name += name + " "
                    class_name = class_name[:-1]
            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    gt_counter_per_class[class_name] = 1
                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)
        with open(os.path.join(temp_files_path, f"{file_id}_ground_truth.json"), "w") as outfile:
            json.dump(bounding_boxes, outfile)
    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    dr_files_list = glob.glob(dr_path + "/*.txt")
    dr_files_list.sort()
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(gt_path, file_id + ".txt")
            if class_index == 0:
                if not os.path.exists(temp_path):
                    raise FileNotFoundError(f"File not found error: {temp_path}.\n")
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except:
                    line_split = line.split()
                    bottom = line_split[-1]
                    right = line_split[-2]
                    top = line_split[-3]
                    left = line_split[-4]
                    confidence = line_split[-5]
                    tmp_class_name = ""
                    for name in line_split[:-5]:
                        tmp_class_name += name + " "
                    tmp_class_name = tmp_class_name[:-1]
                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
        bounding_boxes.sort(key=lambda x: float(x["confidence"]), reverse=True)
        with open(os.path.join(temp_files_path, f"{class_name}_dr.json"), "w") as outfile:
            json.dump(bounding_boxes, outfile)
    sum_ap = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    with open(f"{results_files_path}/results.txt", "w") as results_file:
        results_file.write("AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            dr_file = os.path.join(temp_files_path, f"{class_name}_dr.json")
            dr_data = json.load(open(dr_file))
            nd = len(dr_data)
            tp = [0] * nd
            fp = [0] * nd
            score = [0.0] * nd
            score_threshold_idx = 0
            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                score[idx] = float(detection["confidence"])
                if score[idx] >= score_threshold:
                    score_threshold_idx = idx
                gt_file = os.path.join(temp_files_path, f"{file_id}_ground_truth.json")
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                bb = [float(x) for x in detection["bbox"].split()]
                for obj in ground_truth_data:
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (
                                    bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            with open(gt_file, "w") as f:
                                f.write(json.dumps(ground_truth_data))
                        else:
                            fp[idx] = 1
                else:
                    fp[idx] = 1
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / np.maximum(gt_counter_per_class[class_name], 1)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), 1)
            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            f1 = np.array(rec) * np.array(prec) * 2 / np.where((np.array(prec) + np.array(rec)) == 0, 1,
                                                               (np.array(prec) + np.array(rec)))
            sum_ap += ap
            text = class_name + "; AP={:.2f}%".format(ap * 100)
            if len(prec) > 0:
                f1_text = class_name + "; F1=" + "{:.2f}".format(f1[score_threshold_idx])
                recall_text = class_name + "; recall=" + "{:.2f}%".format(rec[score_threshold_idx] * 100)
                precision_text = class_name + "; precision=" + "{:.2f}%".format(prec[score_threshold_idx] * 100)
            else:
                f1_text = class_name + "; F1=0.00"
                recall_text = class_name + "; recall=0.00%"
                precision_text = class_name + "; precision=0.00%"
            rounded_prec = ["%.2f" % elem for elem in prec]
            rounded_rec = ["%.2f" % elem for elem in rec]
            results_file.write(text + "\nprecision: " + str(rounded_prec) + "\nrecall: " + str(rounded_rec) + "\n\n")
            ap_dictionary[class_name] = ap
            n_images = counter_images_per_class[class_name]
            lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
            lamr_dictionary[class_name] = lamr
            if draw_plot:
                plt.plot(rec, prec, "-o")
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor="r")
                fig = plt.gcf()
                fig.canvas.manager.set_window_title("AP " + class_name)
                plt.title("class: " + text)
                plt.xlabel("recall")
                plt.ylabel("precision")
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(os.path.join(results_files_path, "AP", class_name + ".png"))
                plt.cla()
                plt.plot(score, f1, "-", color="orangered")
                plt.title("class: " + f1_text + "\nscore threshold=" + str(score_threshold))
                plt.xlabel("score threshold")
                plt.ylabel("F1")
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(os.path.join(results_files_path, "F1", class_name + ".png"))
                plt.cla()
                plt.plot(score, rec, "-H", color="gold")
                plt.title("class: " + recall_text + "\nscore threshold=" + str(score_threshold))
                plt.xlabel("score threshold")
                plt.ylabel("recall")
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(os.path.join(results_files_path, "recall", class_name + ".png"))
                plt.cla()
                plt.plot(score, prec, "-s", color="palevioletred")
                plt.title("class: " + precision_text + "\nscore threshold=" + str(score_threshold))
                plt.xlabel("score threshold")
                plt.ylabel("precision")
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(os.path.join(results_files_path, "precision", class_name + ".png"))
                plt.cla()
        if n_classes == 0:
            raise ValueError("data/classes.txt error.")
        results_file.write("\nmAP of all classes\n")
        mAP = sum_ap / n_classes
        text = "mAP={:.2f}%".format(mAP * 100)
        results_file.write(text + "\n")
        print(text)
    shutil.rmtree(temp_files_path)
    det_counter_per_class = {}
    for txt_file in dr_files_list:
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            if class_name in det_counter_per_class:
                det_counter_per_class[class_name] += 1
            else:
                det_counter_per_class[class_name] = 1
    dr_classes = list(det_counter_per_class.keys())
    with open(results_files_path + "/results.txt", "a") as results_file:
        results_file.write("\nnumber of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(f"{class_name}: str(gt_counter_per_class[class_name])\n")
    for class_name in dr_classes:
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0
    with open(results_files_path + "/results.txt", "a") as results_file:
        results_file.write("\nnumber of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ": " + str(n_det)
            text += f" (tp: {str(count_true_positives[class_name])}"
            text += f", fp: {str(n_det - count_true_positives[class_name])})\n"
            results_file.write(text)
    if draw_plot:
        window_title = "ground-truth"
        plot_title = "ground-truth\n"
        plot_title += str(len(ground_truth_files_list)) + " images; " + str(n_classes) + " classes"
        x_label = "number of objects per class"
        output_path = results_files_path + "/ground-truth.png"
        plot_color = "forestgreen"
        draw_plot_func(gt_counter_per_class, n_classes, window_title, plot_title, x_label, output_path, plot_color)
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = results_files_path + "/lamr.png"
        plot_color = "royalblue"
        draw_plot_func(lamr_dictionary, n_classes, window_title, plot_title, x_label, output_path, plot_color)
        window_title = "mAP"
        plot_title = "mAP={:.2f}%".format(mAP * 100)
        x_label = "average precision"
        output_path = results_files_path + "/mAP.png"
        plot_color = "royalblue"
        draw_plot_func(ap_dictionary, n_classes, window_title, plot_title, x_label, output_path, plot_color)
    return mAP


def get_txts(seed=0, trainval_percent=0.9, train_percent=0.9):
    random.seed(seed)
    trainval_percent = trainval_percent
    train_percent = train_percent
    classes, _ = get_classes("data/classes.txt")
    photo_nums = np.zeros(2)
    nums = np.zeros(len(classes))
    temp_xml = os.listdir("VOC/Annotations")
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
    os.makedirs("VOC/ImageSets/Main", exist_ok=True)
    ftrainval = open("VOC/ImageSets/Main/trainval.txt", "w")
    ftest = open("VOC/ImageSets/Main/test.txt", "w")
    ftrain = open("VOC/ImageSets/Main/train.txt", "w")
    fval = open("VOC/ImageSets/Main/val.txt", "w")
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
        image_ids = open(f"VOC/ImageSets/Main/{image_set}.txt", encoding="utf-8").read().strip().split()
        list_file = open(f"data/{image_set}.txt", "w", encoding="utf-8")
        for image_id in image_ids:
            list_file.write(f"VOC/JPEGImages/{image_id}.jpg")
            in_file = open(f"VOC/Annotations/{image_id}.xml", encoding="utf-8")
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
    table_data = [classes, str_nums]
    col_widths = [0] * len(table_data)
    for i in range(len(table_data)):
        for j in range(len(table_data[i])):
            if len(table_data[i][j]) > col_widths[i]:
                col_widths[i] = len(table_data[i][j])
    print_table(table_data, col_widths)
    if photo_nums[0] <= 500:
        raise ValueError("Dataset not qualified.")
    if np.sum(nums) == 0:
        raise ValueError("data/classes.txt error.")


def kmeans(box, k):
    row = box.shape[0]
    distance = np.empty((row, k))
    last_clu = np.zeros((row,))
    np.random.seed()
    cluster = box[np.random.choice(row, k, replace=False)]
    iternum = 0
    while True:
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)
        near = np.argmin(distance, axis=1)
        if (last_clu == near).all():
            break
        for j in range(k):
            cluster[j] = np.median(box[near == j], axis=0)
        last_clu = near
        if iternum % 5 == 0:
            print("iter: {:d}; avg_iou: {:.2f}".format(iternum, avg_iou(box, cluster)))
        iternum += 1
    return cluster, near


def load_data(path):
    data = []
    for xml_file in tqdm(glob.glob(f"{path}/*xml")):
        tree = ET.parse(xml_file)
        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
        if height <= 0 or width <= 0:
            continue
        for obj in tree.iter("object"):
            xmin = np.float64(int(float(obj.findtext("bndbox/xmin"))) / width)
            ymin = np.float64(int(float(obj.findtext("bndbox/ymin"))) / height)
            xmax = np.float64(int(float(obj.findtext("bndbox/xmax"))) / width)
            ymax = np.float64(int(float(obj.findtext("bndbox/ymax"))) / height)
            data.append([xmax - xmin, ymax - ymin])
    return np.array(data)


def log_average_miss_rate(precision, fp_cumsum, num_images):
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi
    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)
    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
    return lamr, mr, fppi


def logistic(x):
    if np.all(x >= 0):
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def make3conv(filters_list, in_filters):
    return nn.Sequential(conv2d(in_filters, filters_list[0], 1),
                         conv_dw(filters_list[0], filters_list[1]),
                         conv2d(filters_list[1], filters_list[0], 1))


def make5conv(filters_list, in_filters):
    return nn.Sequential(conv2d(in_filters, filters_list[0], 1),
                         conv_dw(filters_list[0], filters_list[1]),
                         conv2d(filters_list[1], filters_list[0], 1),
                         conv_dw(filters_list[0], filters_list[1]),
                         conv2d(filters_list[1], filters_list[0], 1))


def print_table(table_data, col_widths):
    print("-" * (sum(col_widths) + 7))
    for i in range(len(table_data[0])):
        print(f"| {table_data[0][i].rjust(col_widths[0])} | {table_data[1][i].rjust(col_widths[1])} |")
    print("-" * (sum(col_widths) + 7))


def resize_image(image, size):
    w, h = size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def show_config(**_map):
    width_k = max(len(k) for k in _map.keys())
    width_v = max(len(str(v)) for v in _map.values())
    width = width_k + width_v + 7
    print("-" * width)
    for k, v in _map.items():
        print(f"| {k.rjust(width_k)} | {str(v).rjust(width_v)} |")
    print("-" * width)


def step_lr(lr, decay_rate, step_size, iters):
    if step_size < 1:
        raise ValueError("step_size error.")
    n = iters // step_size
    return lr * decay_rate ** n


def voc_ap(rec, prec):
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes


def yolo_head(filters_list, in_filters):
    return nn.Sequential(conv_dw(in_filters, filters_list[0]), nn.Conv2d(filters_list[0], filters_list[1], 1))


def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
    elif iters >= total_iters - no_aug_iter:
        lr = min_lr
    else:
        lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(
            math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter)))
    return lr
