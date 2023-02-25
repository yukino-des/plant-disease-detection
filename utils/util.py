import glob
import json
import math
import matplotlib
import numpy as np
import operator
import os
import shutil
import sys
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.hub import load_state_dict_from_url
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import nms
from tqdm import tqdm
from scipy import signal

matplotlib.use("TkAgg")


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def cvt_color(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert("RGB")
        return image


def download_weights(backbone, model_dir="../data"):
    download_urls = {
        "mobilenetv2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
    }
    url = download_urls[backbone]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)


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
            str_val = " {0:.2f}".format(val)
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


def error(msg):
    print(msg)
    sys.exit(0)


def file_lines_to_list(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


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


def get_map(min_overlap, draw_plot, score_threshold=0.5, path="./maps_out"):
    gt_path = os.path.join(path, "ground-truth")
    dr_path = os.path.join(path, "detection")
    temp_files_path = os.path.join(path, ".temp_files")
    results_files_path = os.path.join(path, "results")
    if not os.path.exists(temp_files_path):
        os.makedirs(temp_files_path)
    if os.path.exists(results_files_path):
        shutil.rmtree(results_files_path)
    else:
        os.makedirs(results_files_path)
    if draw_plot:
        try:
            matplotlib.use("TkAgg")
        except:
            pass
        os.makedirs(os.path.join(results_files_path, "AP"))
        os.makedirs(os.path.join(results_files_path, "F1"))
        os.makedirs(os.path.join(results_files_path, "Recall"))
        os.makedirs(os.path.join(results_files_path, "Precision"))
    ground_truth_files_list = glob.glob(gt_path + "/*.txt")
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    gt_counter_per_class = {}
    counter_images_per_class = {}
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        temp_path = os.path.join(dr_path, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error(error_msg)
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
        with open(temp_files_path + "/" + file_id + "_ground_truth.json", "w") as outfile:
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
            temp_path = os.path.join(gt_path, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error(error_msg)
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
        with open(temp_files_path + "/" + class_name + "_dr.json", "w") as outfile:
            json.dump(bounding_boxes, outfile)
    sum_ap = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    with open(results_files_path + "/results.txt", "w") as results_file:
        results_file.write("AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            dr_file = temp_files_path + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))
            nd = len(dr_data)
            tp = [0] * nd
            fp = [0] * nd
            score = [0] * nd
            score_threshold_idx = 0
            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                score[idx] = float(detection["confidence"])
                if score[idx] >= score_threshold:
                    score_threshold_idx = idx
                gt_file = temp_files_path + "/" + file_id + "_ground_truth.json"
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
            text = class_name + "; AP={0:.2f}%".format(ap * 100)
            if len(prec) > 0:
                f1_text = class_name + "; F1=" + "{0:.2f}".format(f1[score_threshold_idx])
                recall_text = class_name + "; Recall=" + "{0:.2f}%".format(rec[score_threshold_idx] * 100)
                precision_text = class_name + "; Precision=" + "{0:.2f}%".format(prec[score_threshold_idx] * 100)
            else:
                f1_text = class_name + "; F1=0.00"
                recall_text = class_name + "; Recall=0.00%"
                precision_text = class_name + "; Precision=0.00%"
            rounded_prec = ["%.2f" % elem for elem in prec]
            rounded_rec = ["%.2f" % elem for elem in rec]
            results_file.write(text + "\nPrecision: " + str(rounded_prec) + "\nRecall: " + str(rounded_rec) + "\n\n")
            if len(prec) > 0:
                print(text + "; score_threshold=" + str(score_threshold) + "; F1=" + "{0:.2f}".format(
                    f1[score_threshold_idx]) + "; Recall=" + "{0:.2f}%".format(
                    rec[score_threshold_idx] * 100) + "; Precision=" + "{0:.2f}%".format(
                    prec[score_threshold_idx] * 100))
            else:
                print(text + "; score_threshold=" + str(
                    score_threshold) + "; " + "F1=0.00; Recall=0.00%; Precision=0.00%")
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
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(results_files_path + "/AP/" + class_name + ".png")
                plt.cla()
                plt.plot(score, f1, "-", color="orangered")
                plt.title("class: " + f1_text + "\nscore_threshold=" + str(score_threshold))
                plt.xlabel("Score_Threshold")
                plt.ylabel("F1")
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(results_files_path + "/F1/" + class_name + ".png")
                plt.cla()
                plt.plot(score, rec, "-H", color="gold")
                plt.title("class: " + recall_text + "\nscore_threshold=" + str(score_threshold))
                plt.xlabel("Score_Threshold")
                plt.ylabel("Recall")
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(results_files_path + "/Recall/" + class_name + ".png")
                plt.cla()
                plt.plot(score, prec, "-s", color="palevioletred")
                plt.title("class: " + precision_text + "\nscore_threshold=" + str(score_threshold))
                plt.xlabel("Score_Threshold")
                plt.ylabel("Precision")
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(results_files_path + "/Precision/" + class_name + ".png")
                plt.cla()
        if n_classes == 0:
            print("../data/classes.txt error.")
            return 0
        results_file.write("\nmAP of all classes\n")
        _map_ = sum_ap / n_classes
        text = "mAP = {0:.2f}%".format(_map_ * 100)
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
        results_file.write("\nNumber of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")
    for class_name in dr_classes:
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0
    with open(results_files_path + "/results.txt", "a") as results_file:
        results_file.write("\nNumber of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ": " + str(n_det)
            text += " (tp: " + str(count_true_positives[class_name])
            text += ", fp: " + str(n_det - count_true_positives[class_name]) + ")\n"
            results_file.write(text)
    if draw_plot:
        window_title = "ground-truth-info"
        plot_title = "ground-truth\n"
        plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
        x_label = "Number of objects per class"
        output_path = results_files_path + "/ground-truth-info.png"
        plot_color = "forestgreen"
        draw_plot_func(gt_counter_per_class, n_classes, window_title, plot_title, x_label, output_path, plot_color)
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = results_files_path + "/lamr.png"
        plot_color = "royalblue"
        draw_plot_func(lamr_dictionary, n_classes, window_title, plot_title, x_label, output_path, plot_color)
        window_title = "mAP"
        plot_title = "mAP={0:.2f}%".format(_map_ * 100)
        x_label = "Average Precision"
        output_path = results_files_path + "/mAP.png"
        plot_color = "royalblue"
        draw_plot_func(ap_dictionary, n_classes, window_title, plot_title, x_label, output_path, plot_color)
    return _map_


def logistic(x):
    if np.all(x >= 0):
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


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


def preprocess_input(image):
    image /= 255.0
    return image


def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def show_config(**kwargs):
    print("Configurations:")
    print("-" * 70)
    print("|%25s | %40s|" % ("keys", "values"))
    print("-" * 70)
    for key, value in kwargs.items():
        print("|%25s | %40s|" % (str(key), str(value)))
    print("-" * 70)


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


class DecodeBox:
    def __init__(self, anchors, num_classes, input_shape, anchors_mask=None):
        super(DecodeBox, self).__init__()
        if anchors_mask is None:
            anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask

    def decode_box(self, inputs):
        outputs = []
        for i, _input in enumerate(inputs):
            batch_size = _input.size(0)
            input_height = _input.size(2)
            input_width = _input.size(3)
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                              self.anchors[self.anchors_mask[i]]]
            prediction = _input.view(batch_size,
                                     len(self.anchors_mask[i]),
                                     self.bbox_attrs,
                                     input_height,
                                     input_width).permute(0, 1, 3, 4, 2).contiguous()
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
            w = prediction[..., 2]
            h = prediction[..., 3]
            conf = torch.sigmoid(prediction[..., 4])
            pred_cls = torch.sigmoid(prediction[..., 5:])
            float_tensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            long_tensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(float_tensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(float_tensor)
            anchor_w = float_tensor(scaled_anchors).index_select(1, long_tensor([0]))
            anchor_h = float_tensor(scaled_anchors).index_select(1, long_tensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
            pred_boxes = float_tensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(float_tensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)
        return outputs

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)
        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape
            box_yx = (box_yx - offset) * scale
            box_hw *= scale
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def non_max_suppression(self, prediction,
                            num_classes,
                            input_shape,
                            image_shape,
                            letterbox_image,
                            conf_thres=0.5,
                            nms_thres=0.4):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
            unique_labels = detections[:, -1].cpu().unique()
            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()
            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4] * detections_class[:, 5],
                    nms_thres
                )
                max_detections = detections_class[keep]
                output[i] = max_detections if output[i] is None else torch.cat([output[i], max_detections])
            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output


class EvalCallback:
    def __init__(self, net, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir, cuda,
                 maps_out_path="../tmp/.maps_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True,
                 min_overlap=0.5, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        self.net = net
        self.input_shape = input_shape
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.cuda = cuda
        self.maps_out_path = maps_out_path
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.min_overlap = min_overlap
        self.eval_flag = eval_flag
        self.period = period
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)
        self.maps = [0]
        self.epochs = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), "a") as f:
                f.write(str(0))
                f.write("\n")

    def get_map_txt(self, image_id, image, class_names, maps_out_path):
        f = open(os.path.join(maps_out_path, "detection/" + image_id + ".txt"), "w", encoding="utf-8")
        image_shape = np.array(np.shape(image)[0:2])
        image = cvt_color(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
            if results[0] is None:
                return
            top_label = np.array(results[0][:, 6], dtype="int32")
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        top_100 = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes = top_boxes[top_100]
        top_conf = top_conf[top_100]
        top_label = top_label[top_100]
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])
            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue
            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
        f.close()
        return

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            if not os.path.exists(self.maps_out_path):
                os.makedirs(self.maps_out_path)
            if not os.path.exists(os.path.join(self.maps_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.maps_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.maps_out_path, "detection")):
                os.makedirs(os.path.join(self.maps_out_path, "detection"))
            print("Get map.")
            for annotation_line in tqdm(self.val_lines):
                line = annotation_line.split()
                image_id = os.path.basename(line[0]).split(".")[0]
                image = Image.open(line[0])
                gt_boxes = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])
                self.get_map_txt(image_id, image, self.class_names, self.maps_out_path)
                with open(os.path.join(self.maps_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
            print("Calculate Map.")
            temp_map = get_map(self.min_overlap, False, path=self.maps_out_path)
            self.maps.append(temp_map)
            self.epochs.append(epoch)
            with open(os.path.join(self.log_dir, "epoch_map.txt"), "a") as f:
                f.write(str(temp_map))
                f.write("\n")
            plt.figure()
            plt.plot(self.epochs, self.maps, "red", linewidth=2, label="train map")
            plt.grid(True)
            plt.xlabel("Epoch")
            plt.ylabel("Map %s" % str(self.min_overlap))
            plt.title("A Map Curve")
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")
            print("Get map done.")
            shutil.rmtree(self.maps_out_path)


class LossHistory:
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), "a") as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), "a") as f:
            f.write(str(val_loss))
            f.write("\n")
        self.writer.add_scalar("loss", loss, epoch)
        self.writer.add_scalar("val_loss", val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, "red", linewidth=2, label="train loss")
        plt.plot(iters, self.val_loss, "coral", linewidth=2, label="val loss")
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, signal.savgol_filter(self.losses, num, 3), "green", linestyle="--", linewidth=2,
                     label="train loss")
            plt.plot(iters, signal.savgol_filter(self.val_loss, num, 3), "#8B4513", linestyle="--", linewidth=2,
                     label="val loss")
        except:
            pass
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")
