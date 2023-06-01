import glob
import json
import math
import operator
import os
import random
import shutil
from collections import OrderedDict
from functools import partial
from xml.etree import ElementTree

import matplotlib
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

if os.name == "nt":
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")


def adjust_axes(r, t, fig, axes):
    b0 = t.get_window_extent(renderer=r)
    text_width_inches = b0.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    proportion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * proportion])


def avg_iou(box, cluster):
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def cas_iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])
    intersection = x * y
    area1 = box[0] * box[1]
    area2 = cluster[:, 0] * cluster[:, 1]
    return intersection / (area1 + area2 - intersection)


def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups,
                           bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True))]))


def conv_dw(filter_in, filter_out, stride=1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, kernel_size=3, stride=stride, padding=1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),
        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True))


def cvt_color(image):
    if len(np.shape(image)) != 3 or np.shape(image)[2] != 3:
        image = image.convert("RGB")
    return image


def draw_plot(dictionary, n_classes, window_title, plot_title, x_label, output_path, plot_color):
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


def lines_to_list(path):
    with open(path) as f:
        content = f.readlines()
    return [x.strip() for x in content]


def fit1epoch(model_train, model, yolo_loss, loss_history, optimizer, e, epoch_step, epoch_step_val, gen, gen_val,
              epoch, period):
    loss = 0
    val_loss = 0
    bar = tqdm(total=epoch_step, desc=f"epoch {e + 1}/{epoch}", postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if torch.cuda.is_available():
                images = images.cuda(0)
                targets = [ann.cuda(0) for ann in targets]
        optimizer.zero_grad()
        outputs = model_train(images)
        if torch.cuda.is_available():
            loss_sum_all = torch.tensor(0, dtype=torch.float32, device="cuda")
        else:
            loss_sum_all = torch.tensor(0, dtype=torch.float32)
        for i in range(len(outputs)):
            loss_item = yolo_loss(i, outputs[i], targets)
            loss_sum_all += loss_item
        loss_sum = loss_sum_all
        loss_sum.backward()
        optimizer.step()
        loss += loss_sum.item()
        bar.set_postfix(**{"loss": loss / (iteration + 1), "lr": get_lr(optimizer)})
        bar.update(1)
    bar.close()
    bar = tqdm(total=epoch_step_val, desc=f"epoch {e + 1}/{epoch}", postfix=dict, mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if torch.cuda.is_available():
                images = images.cuda(0)
                targets = [ann.cuda(0) for ann in targets]
            optimizer.zero_grad()
            outputs = model_train(images)
            if torch.cuda.is_available():
                val_loss_sum = torch.tensor(0, dtype=torch.float32, device="cuda")
            else:
                val_loss_sum = torch.tensor(0, dtype=torch.float32)
            for i in range(len(outputs)):
                val_loss_item = yolo_loss(i, outputs[i], targets)
                val_loss_sum += val_loss_item
        val_loss += val_loss_sum.item()
        bar.set_postfix(**{"val_loss": val_loss / (iteration + 1)})
        bar.update(1)
    bar.close()
    loss_history.append_loss(e + 1, loss / epoch_step, val_loss / epoch_step_val)
    print(f"epoch: {str(e + 1)}/{str(epoch)}")
    print("loss=%.3f; val_loss=%.3f" % (loss / epoch_step, val_loss / epoch_step_val))
    if (e + 1) % period == 0 or e + 1 == epoch:
        torch.save(model.state_dict(), "data/cache/loss/epoch%03d-loss%.3f-val_loss%.3f.pth" % (
            e + 1, loss / epoch_step, val_loss / epoch_step_val))
    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        torch.save(model.state_dict(), "data/model.pth")
        print("data/model.pth saved.")
    torch.save(model.state_dict(), "data/cache/loss/current.pth")


def get_anchors(anchors_txt):
    with open(anchors_txt, encoding="utf-8") as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(",")]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


def get_classes(classes_txt):
    with open(classes_txt, encoding="utf-8") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iter, warmup_iter_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    if lr_decay_type == "cos":
        warmup_total_iter = min(max(warmup_iter_ratio * total_iter, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iter, 1), 15)
        func = partial(warm_cos_lr, lr, min_lr, total_iter, warmup_total_iter, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iter / step_num
        func = partial(step_lr, lr, decay_rate, step_size)
    return func


def get_map(min_overlap, score_threshold):
    os.makedirs("data/cache/map/temp", exist_ok=True)
    os.makedirs("data/cache/map/AP", exist_ok=True)
    os.makedirs("data/cache/map/F1", exist_ok=True)
    os.makedirs("data/cache/map/recall", exist_ok=True)
    os.makedirs("data/cache/map/precision", exist_ok=True)
    ground_truth_files = glob.glob("data/cache/map/ground-truth/*.txt")
    if len(ground_truth_files) == 0:
        raise FileNotFoundError("Ground-truth files not found.")
    ground_truth_files.sort()
    gt_counter_per_class = {}
    counter_images_per_class = {}
    for txt_file in ground_truth_files:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        temp_path = f"data/cache/map/result/{file_id}.txt"
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"{temp_path} not found.")
        lines_list = lines_to_list(txt_file)
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _ = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except ValueError:
                if "difficult" in line:
                    line_split = line.split()
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
            bbox = f"{left} {top} {right} {bottom}"
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
        with open(f"data/cache/map/temp/{file_id}-ground-truth.json", "w") as outfile:
            json.dump(bounding_boxes, outfile)
    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    dr_files_list = glob.glob("data/cache/map/result/*.txt")
    dr_files_list.sort()
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = f"data/cache/map/ground-truth/{file_id}.txt"
            if class_index == 0:
                if not os.path.exists(temp_path):
                    raise FileNotFoundError(f"{temp_path} not found.")
            lines = lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
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
                    bbox = f"{left} {top} {right} {bottom}"
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
        bounding_boxes.sort(key=lambda x: float(x["confidence"]), reverse=True)
        with open(f"data/cache/map/temp/{class_name}_dr.json", "w") as outfile:
            json.dump(bounding_boxes, outfile)
    sumAP = 0.0
    ap_dict = {}
    log_avg_miss_rate_dict = {}
    with open("data/cache/map/results.txt", "w") as results_file:
        results_file.write("AP, precision, recall\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            dr_file = f"data/cache/map/temp/{class_name}_dr.json"
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
                gt_file = f"data/cache/map/temp/{file_id}-ground-truth.json"
                ground_truth_data = json.load(open(gt_file))
                ov_max = -1
                gt_match = -1
                b0 = [float(x) for x in detection["bbox"].split()]
                for obj in ground_truth_data:
                    if obj["class_name"] == class_name:
                        b1 = [float(x) for x in obj["bbox"].split()]
                        bi = [max(b0[0], b1[0]), max(b0[1], b1[1]), min(b0[2], b1[2]), min(b0[3], b1[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            ua = (b0[2] - b0[0] + 1) * (b0[3] - b0[1] + 1) + (b1[2] - b1[0] + 1) * (
                                    b1[3] - b1[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ov_max:
                                ov_max = ov
                                gt_match = obj
                if ov_max >= min_overlap:
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
            cum_sum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cum_sum
                cum_sum += val
            cum_sum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cum_sum
                cum_sum += val
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / np.maximum(gt_counter_per_class[class_name], 1)
            precision = tp[:]
            for idx, val in enumerate(tp):
                precision[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), 1)
            ap, m_recall, m_precision = voc_ap(rec[:], precision[:])
            f1 = np.array(rec) * np.array(precision) * 2 / np.where((np.array(precision) + np.array(rec)) == 0, 1,
                                                                    (np.array(precision) + np.array(rec)))
            sumAP += ap
            text = class_name + "\nAP={:.2f}%".format(ap * 100)
            if len(precision) > 0:
                f1_text = class_name + "\nF1={:.2f}".format(f1[score_threshold_idx])
                recall_text = class_name + "\nrecall={:.2f}%".format(rec[score_threshold_idx] * 100)
                precision_text = class_name + "\nprecision={:.2f}%".format(precision[score_threshold_idx] * 100)
            else:
                f1_text = class_name + "\nF1=0.00"
                recall_text = class_name + "\nrecall=0.00%"
                precision_text = class_name + "\nprecision=0.00%"
            rounded_precision = ["%.2f" % elem for elem in precision]
            rounded_rec = ["%.2f" % elem for elem in rec]
            results_file.write(f"{text}\nprecision={str(rounded_precision)}\nrecall={str(rounded_rec)}\n\n")
            ap_dict[class_name] = ap
            n_images = counter_images_per_class[class_name]
            log_avg_miss_rate, mr, false_pos_per_image = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
            log_avg_miss_rate_dict[class_name] = log_avg_miss_rate
            plt.plot(rec, precision, "-o")
            area_under_curve_x = m_recall[:-1] + [m_recall[-2]] + [m_recall[-1]]
            area_under_curve_y = m_precision[:-1] + [0.0] + [m_precision[-1]]
            plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor="r")
            fig = plt.gcf()
            plt.title(text)
            plt.xlabel("recall")
            plt.ylabel("precision")
            axes = plt.gca()
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])
            fig.savefig(f"data/cache/map/AP/{class_name}.png")
            plt.cla()
            plt.plot(score, f1, "-", color="orangered")
            plt.title(f1_text)
            plt.xlabel("score_threshold=" + str(score_threshold))
            plt.ylabel("F1")
            axes = plt.gca()
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])
            fig.savefig(f"data/cache/map/F1/{class_name}.png")
            plt.cla()
            plt.plot(score, rec, "-H", color="gold")
            plt.title(recall_text)
            plt.xlabel("score_threshold=" + str(score_threshold))
            plt.ylabel("recall")
            axes = plt.gca()
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])
            fig.savefig(f"data/cache/map/recall/{class_name}.png")
            plt.cla()
            plt.plot(score, precision, "-s", color="palevioletred")
            plt.title("class: " + precision_text)
            plt.xlabel("score_threshold=" + str(score_threshold))
            plt.ylabel("precision")
            axes = plt.gca()
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])
            fig.savefig(f"data/cache/map/precision/{class_name}.png")
            plt.cla()
        if n_classes == 0:
            raise ValueError("data/classes.txt error")
        mAP = sumAP / n_classes
        text = "mAP={:.2f}%".format(mAP * 100)
        results_file.write(text + "\n")
        print(text)
    shutil.rmtree("data/cache/map/temp")
    det_counter_per_class = {}
    for txt_file in dr_files_list:
        lines_list = lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            if class_name in det_counter_per_class:
                det_counter_per_class[class_name] += 1
            else:
                det_counter_per_class[class_name] = 1
    dr_classes = list(det_counter_per_class.keys())
    for class_name in dr_classes:
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0
    with open("data/cache/map/results.txt", "a") as results_file:
        results_file.write("\nnumber of ground-truth objects\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(f"{class_name}: {str(gt_counter_per_class[class_name])}\n")
        results_file.write("\nnumber of detected objects\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ": " + str(n_det)
            text += f" (tp: {str(count_true_positives[class_name])}"
            text += f", fp: {str(n_det - count_true_positives[class_name])})\n"
            results_file.write(text)
    window_title = "ground-truth"
    plot_title = f"{window_title}\n"
    plot_title += str(len(ground_truth_files)) + " images; " + str(n_classes) + " classes"
    x_label = "number of ground-truth objects"
    output_path = "data/cache/map/ground-truth.png"
    plot_color = "forestgreen"
    draw_plot(gt_counter_per_class, n_classes, window_title, plot_title, x_label, output_path, plot_color)
    window_title = "log-average miss rate"
    plot_title = window_title
    x_label = "log-average miss rate"
    output_path = "data/cache/map/log-average_miss_rate.png"
    plot_color = "royalblue"
    draw_plot(log_avg_miss_rate_dict, n_classes, window_title, plot_title, x_label, output_path, plot_color)
    window_title = "mAP={:.2f}%".format(mAP * 100)
    plot_title = window_title
    x_label = "Average Precision"
    output_path = "data/cache/map/mAP.png"
    plot_color = "royalblue"
    draw_plot(ap_dict, n_classes, window_title, plot_title, x_label, output_path, plot_color)
    return mAP


def get_txt(seed, train_val_percent, train_percent):
    random.seed(seed)
    classes, _ = get_classes("data/classes.txt")
    photo_nums = np.zeros(2)
    nums = np.zeros(len(classes))
    temp_xml = os.listdir("data/VOCdevkit/VOC2007/Annotations")
    total_xml = []
    for xml in temp_xml:
        if xml.endswith(".xml"):
            total_xml.append(xml)
    num = len(total_xml)
    num_list = range(num)
    tv = int(num * train_val_percent)
    tr = int(tv * train_percent)
    train_val = random.sample(num_list, tv)
    train = random.sample(train_val, tr)
    os.makedirs("data/VOCdevkit/VOC2007/ImageSets/Main", exist_ok=True)
    train_val_txt = open("data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt", "w")
    test_txt = open("data/VOCdevkit/VOC2007/ImageSets/Main/test.txt", "w")
    train_txt = open("data/VOCdevkit/VOC2007/ImageSets/Main/train.txt", "w")
    val_txt = open("data/VOCdevkit/VOC2007/ImageSets/Main/val.txt", "w")
    for i in num_list:
        name = total_xml[i][:-4] + "\n"
        if i in train_val:
            train_val_txt.write(name)
            train_txt.write(name) if i in train else val_txt.write(name)
        else:
            test_txt.write(name)
    train_val_txt.close()
    train_txt.close()
    val_txt.close()
    test_txt.close()
    type_index = 0
    for image_set in ["train", "val"]:
        image_ids = open(f"data/VOCdevkit/VOC2007/ImageSets/Main/{image_set}.txt",
                         encoding="utf-8").read().strip().split()
        list_file = open(f"data/{image_set}.txt", "w", encoding="utf-8")
        for image_id in image_ids:
            list_file.write(f"data/VOCdevkit/VOC2007/JPEGImages/{image_id}.jpg")
            in_file = open(f"data/VOCdevkit/VOC2007/Annotations/{image_id}.xml", encoding="utf-8")
            tree = ElementTree.parse(in_file)
            root = tree.getroot()
            for obj in root.iter("object"):
                difficult = 0
                if obj.find("difficult") is not None:
                    difficult = obj.find("difficult").text
                cls = obj.find("name").text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xml_box = obj.find("bndbox")
                b = (int(float(xml_box.find("xmin").text)), int(float(xml_box.find("ymin").text)),
                     int(float(xml_box.find("xmax").text)), int(float(xml_box.find("ymax").text)))
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
        raise ValueError("data/classes.txt error")


def k_means(box, k):
    row = box.shape[0]
    distance = np.empty((row, k))
    last_clu = np.zeros((row,))
    np.random.seed()
    cluster = box[np.random.choice(row, k, replace=False)]
    iter_num = 0
    while True:
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)
        near = np.argmin(distance, axis=1)
        if (last_clu == near).all():
            break
        for j in range(k):
            cluster[j] = np.median(box[near == j], axis=0)
        last_clu = near
        if iter_num % 5 == 0:
            print("iter: {:d}; avg_iou: {:.2f}".format(iter_num, avg_iou(box, cluster)))
        iter_num += 1
    return cluster, near


def log_average_miss_rate(precision, false_pos_cum_sum, num_images):
    if precision.size == 0:
        log_avg_miss_rate = 0
        mr = 1
        false_pos_per_image = 0
        return log_avg_miss_rate, mr, false_pos_per_image
    false_pos_per_image = false_pos_cum_sum / float(num_images)
    mr = (1 - precision)
    fp_per_image_tmp = np.insert(false_pos_per_image, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(fp_per_image_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]
    log_avg_miss_rate = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
    return log_avg_miss_rate, mr, false_pos_per_image


def logistic(x):
    if np.all(x >= 0):
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


def make3conv(filters_list, in_filters):
    return nn.Sequential(conv2d(in_filters, filters_list[0], 1), conv_dw(filters_list[0], filters_list[1]),
                         conv2d(filters_list[1], filters_list[0], 1))


def make5conv(filters_list, in_filters):
    return nn.Sequential(conv2d(in_filters, filters_list[0], 1), conv_dw(filters_list[0], filters_list[1]),
                         conv2d(filters_list[1], filters_list[0], 1), conv_dw(filters_list[0], filters_list[1]),
                         conv2d(filters_list[1], filters_list[0], 1))


def make_divisible(v, divisor):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def print_config(**_map):
    width_k = max(len(k) for k in _map.keys())
    width_v = max(len(str(v)) for v in _map.values())
    width = width_k + width_v + 7
    print("-" * width)
    for k, v in _map.items():
        print(f"| {k.rjust(width_k)} | {str(v).rjust(width_v)} |")
    print("-" * width)


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


def step_lr(lr, decay_rate, step_size, _iter):
    if step_size < 1:
        raise ValueError("step_size error")
    n = _iter // step_size
    return lr * decay_rate ** n


def voc_ap(recall, precision):
    recall.insert(0, 0.0)
    recall.append(1.0)
    m_recall = recall[:]
    precision.insert(0, 0.0)
    precision.append(0.0)
    m_precision = precision[:]
    for i in range(len(m_precision) - 2, -1, -1):
        m_precision[i] = max(m_precision[i], m_precision[i + 1])
    i_list = []
    for i in range(1, len(m_recall)):
        if m_recall[i] != m_recall[i - 1]:
            i_list.append(i)
    ap = 0.0
    for i in i_list:
        ap += ((m_recall[i] - m_recall[i - 1]) * m_precision[i])
    return ap, m_recall, m_precision


def weights_init(net):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def yolo_dataset_collate(batch):
    image_list = []
    bbox_list = []
    for image, box in batch:
        image_list.append(image)
        bbox_list.append(box)
    images = torch.from_numpy(np.array(image_list)).type(torch.FloatTensor)
    bbox_list = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bbox_list]
    return images, bbox_list


def yolo_head(filters_list, in_filters):
    return nn.Sequential(conv_dw(in_filters, filters_list[0]), nn.Conv2d(filters_list[0], filters_list[1], 1))


def warm_cos_lr(lr, min_lr, total_iter, warmup_total_iter, warmup_lr_start, no_aug_iter, _iter):
    if _iter <= warmup_total_iter:
        lr = (lr - warmup_lr_start) * pow(_iter / float(warmup_total_iter), 2) + warmup_lr_start
    elif _iter >= total_iter - no_aug_iter:
        lr = min_lr
    else:
        lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(
            math.pi * (_iter - warmup_total_iter) / (total_iter - warmup_total_iter - no_aug_iter)))
    return lr
