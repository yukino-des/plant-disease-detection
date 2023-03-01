import colorsys
import cv2
import math
import matplotlib
import numpy as np
import onnx
import onnxsim
import os
import time
import torch
from collections import OrderedDict
from functools import partial
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from random import sample, shuffle
from torch import nn
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from utils.mobilenet import mobilenet_v2
from utils.util import (cvt_color, DecodeBox, get_anchors, get_classes, get_lr, logistic, preprocess_input,
                        resize_image, show_config)

if os.name == "nt":
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")

defaults = {
    "model_path": "../data/best.pth",
    "classes_path": "../data/classes.txt",
    "anchors_path": "../data/anchors.txt",
    "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    "input_shape": [416, 416],
    "confidence": 0.5,
    "nms_iou": 0.3,
    "cuda": torch.cuda.is_available()
}


# 标准卷积
def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=pad,
                           groups=groups,
                           bias=False)),  # 3x3标准卷积
        ("bn", nn.BatchNorm2d(filter_out)),  # 批量归一化
        ("relu", nn.ReLU6(inplace=True))]))  # 激活


# 深度可分离卷积
def conv_dw(filter_in, filter_out, stride=1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in,
                  kernel_size=3,
                  stride=stride,
                  padding=1,
                  groups=filter_in,
                  bias=False),  # 3x3深度卷积
        nn.BatchNorm2d(filter_in),  # 批量归一化
        nn.ReLU6(inplace=True),  # 激活
        nn.Conv2d(filter_in, filter_out, 1, 1, 0,
                  bias=False),  # 1x1点卷积
        nn.BatchNorm2d(filter_out),  # 批量归一化
        nn.ReLU6(inplace=True))  # 激活


def fit_one_epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, unfreeze_epoch, cuda, save_period, save_dir, local_rank=0):
    loss = 0
    val_loss = 0
    pbar = None
    if local_rank == 0:
        pbar = tqdm(total=epoch_step, desc=f"Epoch {epoch + 1}/{unfreeze_epoch}", postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
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
        if local_rank == 0:
            pbar.set_postfix(**{"loss": loss / (iteration + 1), "lr": get_lr(optimizer)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        pbar = tqdm(total=epoch_step_val, desc=f"Epoch {epoch + 1}/{unfreeze_epoch}", postfix=dict, mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
            optimizer.zero_grad()
            outputs = model_train(images)
            loss_value_all = 0
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all
        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{"val_loss": val_loss / (iteration + 1)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print("epoch: " + str(epoch + 1) + "/" + str(unfreeze_epoch))
        print("total loss: %.3f; val loss: %.3f" % (loss / epoch_step, val_loss / epoch_step_val))
        if (epoch + 1) % save_period == 0 or epoch + 1 == unfreeze_epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "epoch%03d-loss%.3f-val_loss%.3f.pth" % (
                epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print(f"{defaults['model_path']} saved.")
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
        torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))


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


def make3conv(filters_list, in_filters):
    return nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1))


def make5conv(filters_list, in_filters):
    return nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1))


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def step_lr(lr, decay_rate, step_size, iters):
    if step_size < 1:
        raise ValueError("Let step_size >= 1.")
    n = iters // step_size
    return lr * decay_rate ** n


def weights_init(net):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


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
    m = nn.Sequential(
        conv_dw(in_filters, filters_list[0]),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
    elif iters >= total_iters - no_aug_iter:
        lr = min_lr
    else:
        lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(
            math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter)))
    return lr


class Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super(Backbone, self).__init__()
        self.model = mobilenet_v2(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:14](out3)
        out5 = self.model.features[14:18](out4)
        return out3, out4, out5


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=None):
        super(SpatialPyramidPooling, self).__init__()
        if pool_sizes is None:
            pool_sizes = [5, 9, 13]
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)
        return features


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode="nearest")
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


class YOLO(object):
    def __init__(self, **kwargs):
        self.classes_path = ""
        self.anchors_path = ""
        self.input_shape = []
        self.anchors_mask = []
        self.model_path = ""
        self.cuda = torch.cuda.is_available()
        self.confidence = 0
        self.nms_iou = 0
        self.__dict__.update(defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            defaults[name] = value
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        show_config(**defaults)

    def generate(self, onnx=False):
        self.net = YoloBody(self.anchors_mask, self.num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, crop=False):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvt_color(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
            if results[0] is None:
                return image, {}
            top_label = np.array(results[0][:, 6], dtype="int32")
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        font = ImageFont.truetype(font="../data/simhei.ttf", size=np.floor(3e-2 * image.size[1] + 0.5).astype("int32"))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        for i in range(self.num_classes):
            num = np.sum(top_label == i)
            if num > 0:
                print(self.class_names[i] + ": ", num)
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype("int32"))
                left = max(0, np.floor(left).astype("int32"))
                bottom = min(image.size[1], np.floor(bottom).astype("int32"))
                right = min(image.size[0], np.floor(right).astype("int32"))
                dir_save_path = "../tmp/imgs_out"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print(dir_save_path + "/crop_" + str(i) + ".png saved.")
        image_info = {}
        count = 0
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top).astype("int32"))
            left = max(0, np.floor(left).astype("int32"))
            bottom = min(image.size[1], np.floor(bottom).astype("int32"))
            right = min(image.size[0], np.floor(right).astype("int32"))
            label = "{} {:.2f}".format(predicted_class, score)
            count += 1
            key = "{}-{:02}".format(predicted_class, count)
            image_info[key] = ["{}×{}".format(right - left, bottom - top), np.round(float(score), 3)]
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode("utf-8")
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            for d in range(thickness):
                draw.rectangle((left + d, top + d, right - d, bottom - d), outline=self.colors[c])
            draw.rectangle((tuple(text_origin), tuple(text_origin + label_size)), fill=self.colors[c])
            draw.text(tuple(text_origin), str(label, "UTF-8"), fill=(0, 0, 0), font=font)
            del draw
        return image, image_info

    def get_fps(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvt_color(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, image_shape,
                                               conf_thres=self.confidence, nms_thres=self.nms_iou)
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                   image_shape, conf_thres=self.confidence,
                                                   nms_thres=self.nms_iou)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        image = cvt_color(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
        plt.imshow(image, alpha=1)
        plt.axis("off")
        mask = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, 3, -1, h, w]), [0, 3, 4, 1, 2])[0]
            score = np.max(logistic(sub_output[..., 4]), -1)
            score = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score = (score * 255).astype("uint8")
            mask = np.maximum(mask, normed_score)
        plt.imshow(mask, alpha=0.5, interpolation="nearest", cmap="jet")
        plt.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches="tight", pad_inches=-0.1)
        print(heatmap_save_path + " saved.")

    def convert_to_onnx(self, simplify, model_path):
        self.generate(onnx=True)
        im = torch.zeros(1, 3, *self.input_shape).to("cpu")
        input_layer_names = ["images"]
        output_layer_names = ["output"]
        torch.onnx.export(self.net, im, f=model_path, verbose=False, opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL, do_constant_folding=True,
                          input_names=input_layer_names, output_names=output_layer_names, dynamic_axes=None)
        model_onnx = onnx.load(model_path)
        onnx.checker.check_model(model_onnx)
        if simplify:
            model_onnx, check = onnxsim.simplify(model_onnx, dynamic_input_shape=False, input_shapes=None)
            assert check, "assert check failed"
            onnx.save(model_onnx, model_path)
        print(model_path + " saved.")

    def get_map_txt(self, image_id, image, class_names, maps_out_path):
        f = open(os.path.join(maps_out_path, ".dr/" + image_id + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])
        image = cvt_color(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.confidence, self.nms_iou)
            if results[0] is None:
                return
            top_label = np.array(results[0][:, 6], dtype="int32")
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
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


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained=False):
        super(YoloBody, self).__init__()
        self.backbone = Backbone(pretrained=pretrained)
        in_filters = [32, 96, 320]
        self.conv1 = make3conv([512, 1024], in_filters[2])
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make3conv([512, 1024], 2048)
        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = conv2d(in_filters[1], 256, 1)
        self.make_five_conv1 = make5conv([256, 512], 512)
        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = conv2d(in_filters[0], 128, 1)
        self.make_five_conv2 = make5conv([128, 256], 256)
        self.yolo_head3 = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)], 128)
        self.down_sample1 = conv_dw(128, 256, stride=2)
        self.make_five_conv3 = make5conv([256, 512], 512)
        self.yolo_head2 = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)], 256)
        self.down_sample2 = conv_dw(256, 512, stride=2)
        self.make_five_conv4 = make5conv([512, 1024], 1024)
        self.yolo_head1 = yolo_head([1024, len(anchors_mask[2]) * (5 + num_classes)], 512)

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        p5 = self.conv1(x0)
        p5 = self.SPP(p5)
        p5 = self.conv2(p5)
        p5_upsample = self.upsample1(p5)
        p4 = self.conv_for_P4(x1)
        p4 = torch.cat([p4, p5_upsample], dim=1)
        p4 = self.make_five_conv1(p4)
        p4_upsample = self.upsample2(p4)
        p3 = self.conv_for_P3(x2)
        p3 = torch.cat([p3, p4_upsample], dim=1)
        p3 = self.make_five_conv2(p3)
        p3_downsample = self.down_sample1(p3)
        p4 = torch.cat([p3_downsample, p4], dim=1)
        p4 = self.make_five_conv3(p4)
        p4_downsample = self.down_sample2(p4)
        p5 = torch.cat([p4_downsample, p5], dim=1)
        p5 = self.make_five_conv4(p5)
        out2 = self.yolo_head3(p3)
        out1 = self.yolo_head2(p4)
        out0 = self.yolo_head1(p5)
        return out0, out1, out2


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, mosaic, mixup, mosaic_prob, mixup_prob,
                 train, special_aug_ratio=0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio
        self.epoch_now = -1
        self.length = len(self.annotation_lines)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < (
                self.epoch_length * self.special_aug_ratio):
            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            shuffle(lines)
            image, box = self.get_random_data_with_mosaic(lines, self.input_shape)
            if self.mixup and self.rand() < self.mixup_prob:
                lines = sample(self.annotation_lines, 1)
                image_2, box_2 = self.get_random_data(lines[0], self.input_shape, random=self.train)
                image, box = self.get_random_data_with_mixup(image, box, image_2, box_2)
        else:
            image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box

    def rand(self, a=0.0, b=1.0):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        image = Image.open(line[0])
        image = cvt_color(image)
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])
        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
            return image_data, box
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image_data = np.array(image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
        return image_data, box

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                if i == 0:
                    if x1 > cutx or y1 > cuty:
                        continue
                    if y1 <= cuty <= y2:
                        y2 = cuty
                    if x1 <= cutx <= x2:
                        x2 = cutx
                if i == 1:
                    if x1 > cutx or y2 < cuty:
                        continue
                    if y1 <= cuty <= y2:
                        y1 = cuty
                    if x1 <= cutx <= x2:
                        x2 = cutx
                if i == 2:
                    if x2 < cutx or y2 < cuty:
                        continue
                    if y1 <= cuty <= y2:
                        y1 = cuty
                    if x1 <= cutx <= x2:
                        x1 = cutx
                if i == 3:
                    if x2 < cutx or y1 > cuty:
                        continue
                    if y1 <= cuty <= y2:
                        y2 = cuty
                    if x1 <= cutx <= x2:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)
        image_datas = []
        box_datas = []
        index = 0
        for line in annotation_line:
            line_content = line.split()
            image = Image.open(line_content[0])
            image = cvt_color(image)
            iw, ih = image.size
            box = np.array([np.array(list(map(int, box.split(",")))) for box in line_content[1:]])
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]
            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)
            dx = 0
            dy = 0
            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh
            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)
            index = index + 1
            box_data = []
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box
            image_datas.append(image_data)
            box_datas.append(box_data)
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)
        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]
        new_image = np.array(new_image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)
        return new_image, new_boxes

    def get_random_data_with_mixup(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask=None,
                 focal_loss=False, alpha=0.25, gamma=2):
        super(YOLOLoss, self).__init__()
        if anchors_mask is None:
            anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio = 1 * (num_classes / 80)
        self.focal_loss = focal_loss
        self.focal_loss_ratio = 10
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_threshold = 0.5
        self.cuda = cuda

    def clip_by_tensor(self, t: torch.Tensor, t_min: float, t_max: float):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def bce_loss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def box_ciou(self, b1, b2):
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / torch.clamp(union_area, min=1e-6)
        center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), dim=-1)
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), dim=-1)
        ciou = iou - 1.0 * center_distance / torch.clamp(enclose_diagonal, min=1e-6)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
            b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
            b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
        alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
        ciou = ciou - alpha * v
        return ciou

    def forward(self, l, _input, targets=None):
        bs = _input.size(0)
        in_h = _input.size(2)
        in_w = _input.size(3)
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        prediction = _input.view(bs,
                                 len(self.anchors_mask[l]),
                                 self.bbox_attrs,
                                 in_h,
                                 in_w).permute(0, 1, 3, 4, 2).contiguous()
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        y_true, noobj_mask, _ = self.get_target(l, targets, scaled_anchors, in_h, in_w)
        noobj_mask, pred_boxes = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)
        if self.cuda:
            y_true = y_true.type_as(x)
            noobj_mask = noobj_mask.type_as(x)
        loss = 0
        obj_mask = y_true[..., 4] == 1
        n = torch.sum(obj_mask)
        if n != 0:
            ciou = self.box_ciou(pred_boxes, y_true[..., :4]).type_as(x)
            loss_loc = torch.mean((1 - ciou)[obj_mask])
            loss_cls = torch.mean(self.bce_loss(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio
        if self.focal_loss:
            pos_neg_ratio = torch.where(obj_mask, torch.ones_like(conf) * self.alpha,
                                        torch.ones_like(conf) * (1 - self.alpha))
            hard_easy_ratio = torch.where(obj_mask, torch.ones_like(conf) - conf, conf) ** self.gamma
            loss_conf = torch.mean((self.bce_loss(conf, obj_mask.type_as(conf)) * pos_neg_ratio * hard_easy_ratio)[
                                       noobj_mask.bool() | obj_mask]) * self.focal_loss_ratio
        else:
            loss_conf = torch.mean(self.bce_loss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])
        loss += loss_conf * self.balance[l] * self.obj_ratio
        return loss

    def calculate_iou(self, _box_a, _box_b):
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
        a = box_a.size(0)
        b = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(a, b, 2), box_b[:, 2:].unsqueeze(0).expand(a, b, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(a, b, 2), box_b[:, :2].unsqueeze(0).expand(a, b, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
        union = area_a + area_b - inter
        return inter / union

    def get_target(self, l, targets, anchors, in_h, in_w):
        bs = len(targets)
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad=False)
        for b in range(bs):
            if len(targets[b]) == 0:
                continue
            batch_target = torch.zeros_like(targets[b])
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()
            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))
            anchor_shapes = torch.FloatTensor(
                torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)
            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[l]:
                    continue
                k = self.anchors_mask[l].index(best_n)
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                c = batch_target[t, 4].long()
                noobj_mask[b, k, j, i] = 0
                y_true[b, k, j, i, 0] = batch_target[t, 0]
                y_true[b, k, j, i, 1] = batch_target[t, 1]
                y_true[b, k, j, i, 2] = batch_target[t, 2]
                y_true[b, k, j, i, 3] = batch_target[t, 3]
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 5] = 1
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        bs = len(targets)
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        pred_boxes_x = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h) * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)
        for b in range(bs):
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])
                batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
                batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
                batch_target = batch_target[:, :4].type_as(x)
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask, pred_boxes
