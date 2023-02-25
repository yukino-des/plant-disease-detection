import cv2
import glob
import json
import math
import matplotlib
import numpy as np
import operator
import os
import shutil
import sys
from matplotlib import pyplot as plt

matplotlib.use("Agg")


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


def error(msg):
    print(msg)
    sys.exit(0)


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


def file_lines_to_list(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


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
            results_file.write(text + "\nPrecision:" + str(rounded_prec) + "\nRecall:" + str(rounded_rec) + "\n\n")
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
            print("../data/voc_classes.txt ERROR!")
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
        draw_plot_func(gt_counter_per_class,
                       n_classes,
                       window_title,
                       plot_title,
                       x_label,
                       output_path,
                       plot_color)
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
