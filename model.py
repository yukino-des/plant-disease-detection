import colorsys
import cv2
import math
import numpy as np
import onnx
import onnxsim
import os
import shutil
import time
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from random import sample, shuffle
from scipy import signal
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import nms
from tqdm import tqdm
from utils import (conv2d, conv_dw, cvt_color, get_anchors, get_classes, get_map, logistic, make3conv, make5conv,
                   make_divisible, resize_image, print_config, yolo_head)


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        model = MobileNetV2()
        state_dict = load_state_dict_from_url(url="https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
                                              model_dir="data", progress=True)
        model.load_state_dict(state_dict)
        self.model = model

    def forward(self, x):
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:14](out3)
        out5 = self.model.features[14:18](out4)
        return out3, out4, out5


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True))


class DecodeBox:
    def __init__(self, anchors, num_classes, input_shape):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    @staticmethod
    def yolo_correct_boxes(box_xy, box_wh, image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        image_shape = np.array(image_shape)
        box_min = box_yx - (box_hw / 2.)
        box_max = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_min[..., 0:1], box_min[..., 1:2], box_max[..., 0:1], box_max[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

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
            prediction = _input.view(batch_size, len(self.anchors_mask[i]), self.bbox_attrs, input_height,
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
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale, conf.view(batch_size, -1, 1),
                                pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)
        return outputs

    def non_max_suppression(self, prediction, num_classes, image_shape, conf_threshold=0.5, nms_threshold=0.4):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        output: list = [None]
        for i, image_pred in enumerate(prediction):
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_threshold).squeeze()
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
                keep = nms(detections_class[:, :4], detections_class[:, 4] * detections_class[:, 5], nms_threshold)
                max_detections = detections_class[keep]
                output[i] = max_detections if output[i] is None else torch.cat([output[i], max_detections])
            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, image_shape)
        return output


class EvalCallback:
    def __init__(self,
                 net,
                 input_shape,
                 anchors,
                 class_names,
                 num_classes,
                 val_lines,
                 log_dir,
                 max_boxes=100,
                 confidence=0.05,
                 nms_iou=0.5,
                 min_overlap=0.5,
                 eval_flag=True,
                 period=1):
        super(EvalCallback, self).__init__()
        self.net = net
        self.input_shape = input_shape
        self.anchors = anchors
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.cuda = torch.cuda.is_available()
        self.map_path = "data/cache/map"
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.min_overlap = min_overlap
        self.eval_flag = eval_flag
        self.period = period
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]))
        self.maps = [0.0]
        self.epochs = [0]
        if self.eval_flag:
            with open(f"{self.log_dir}/map.txt", "a") as f:
                f.write(str(0) + "\n")

    def get_map_txt(self, image_id, image, class_names, map_path):
        f = open(f"{map_path}/.dr/{image_id}.txt", "w", encoding="utf-8")
        image_shape = np.array(np.shape(image)[0:2])
        image = cvt_color(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(np.array(image_data, dtype="float32") / 255.0, (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, image_shape,
                                                         self.confidence, self.nms_iou)
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
            f.write(f"{predicted_class} {score[:6]} "
                    f"{str(int(left))} {str(int(top))} {str(int(right))} {str(int(bottom))}\n")
        f.close()
        return

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            os.makedirs(self.map_path, exist_ok=True)
            os.makedirs(f"{self.map_path}/.gt", exist_ok=True)
            os.makedirs(f"{self.map_path}/.dr", exist_ok=True)
            for annotation_line in tqdm(self.val_lines):
                line = annotation_line.split()
                image_id = os.path.basename(line[0]).split(".")[0]
                image = Image.open(line[0])
                gt_boxes = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])
                self.get_map_txt(image_id, image, self.class_names, self.map_path)
                with open(f"{self.map_path}/.gt/{image_id}.txt", "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write(f"{obj_name} {left} {top} {right} {bottom}\n")
            temp_map = get_map(self.min_overlap, False)
            self.maps.append(temp_map)
            self.epochs.append(epoch)
            with open(f"{self.log_dir}/map.txt", "a") as f:
                f.write(str(temp_map))
                f.write("\n")
            plt.figure()
            plt.plot(self.epochs, self.maps, "red", linewidth=2, label="train map")
            plt.grid(True)
            plt.xlabel("epoch")
            plt.ylabel(f"map {str(self.min_overlap)}")
            plt.title("a map curve")
            plt.legend(loc="upper right")
            plt.savefig(f"{self.log_dir}/map.png")
            plt.cla()
            plt.close("all")
            shutil.rmtree(self.map_path)


# 倒残差
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                       nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                       nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class LossHistory:
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
        self.writer.add_graph(model, dummy_input)

    def append_loss(self, epoch, loss, val_loss):
        os.makedirs(self.log_dir, exist_ok=True)
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(f"{self.log_dir}/loss.txt", "a") as f:
            f.write(str(loss))
            f.write("\n")
        with open(f"{self.log_dir}/val_loss.txt", "a") as f:
            f.write(str(val_loss))
            f.write("\n")
        self.writer.add_scalar("loss", loss, epoch)
        self.writer.add_scalar("val_loss", val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        _iter = range(len(self.losses))
        plt.figure()
        plt.plot(_iter, self.losses, "red", linewidth=2, label="train loss")
        plt.plot(_iter, self.val_loss, "green", linewidth=2, label="val loss")
        num = 5 if len(self.losses) < 25 else 15
        try:
            plt.plot(_iter, signal.savgol_filter(self.losses, num, 3), "blue", linestyle="--", linewidth=2,
                     label="train loss")
            plt.plot(_iter, signal.savgol_filter(self.val_loss, num, 3), "#21b3b9", linestyle="--", linewidth=2,
                     label="val loss")
        except ValueError:
            pass
        plt.grid(True)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(loc="upper right")
        plt.savefig(f"{self.log_dir}/loss.png")
        plt.cla()
        plt.close("all")


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_multi=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        if inverted_residual_setting is None:
            inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1],
                                         [6, 160, 3, 2], [6, 320, 1, 1]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting error.")
        input_channel = make_divisible(input_channel * width_multi, round_nearest)
        self.last_channel = make_divisible(last_channel * max(1.0, width_multi), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_multi, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        return self.classifier(x)


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=None):
        super(SpatialPyramidPooling, self).__init__()
        if pool_sizes is None:
            pool_sizes = [5, 9, 13]
        self.max_pools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [max_pool(x) for max_pool in self.max_pools[::-1]]
        return torch.cat(features + [x], dim=1)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(conv2d(in_channels, out_channels, 1),
                                      nn.Upsample(scale_factor=2, mode="nearest"))

    def forward(self, x):
        return self.upsample(x)


class Yolo(object):
    def __init__(self, confidence=0.5, nms_iou=0.3):
        self.anchors_path = "data/anchors.txt"
        self.input_shape = [416, 416]
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.model_path = "data/model.pth"
        self.cuda = torch.cuda.is_available()
        self.classes_path = "data/classes.txt"
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.class_names, self.num_classes = get_classes()
        self.anchors, self.num_anchors = get_anchors()
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]))
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.net = YoloBody(self.num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        if self.cuda:
            self.net = nn.DataParallel(self.net).cuda()
        print_config(anchors_path=self.anchors_path,
                     input_shape=self.input_shape,
                     anchors_mask=self.anchors_mask,
                     model_path=self.model_path,
                     cuda=self.cuda,
                     classes_path=self.classes_path,
                     confidence=self.confidence,
                     nms_iou=self.nms_iou)

    def convert_to_onnx(self, simplify):
        self.net = YoloBody(self.num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        im = torch.zeros(1, 3, *self.input_shape).to("cpu")
        input_layer_names = ["images"]
        output_layer_names = ["output"]
        torch.onnx.export(self.net, im, f="data/cache/model.onnx", verbose=False, opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL, do_constant_folding=True,
                          input_names=input_layer_names, output_names=output_layer_names, dynamic_axes=None)
        model_onnx = onnx.load("data/cache/model.onnx")
        onnx.checker.check_model(model_onnx)
        if simplify:
            model_onnx, check = onnxsim.simplify(model_onnx, dynamic_input_shape=False, input_shapes=None)
            assert check, "assert check failed"
            onnx.save(model_onnx, "data/cache/model.onnx")
        print("data/cache/model.onnx saved.")

    def detect_heatmap(self, image):
        image = cvt_color(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(np.array(image_data, dtype="float32") / 255.0, (2, 0, 1)), 0)
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
        plt.savefig("data/cache/heatmap.png", dpi=200, bbox_inches="tight", pad_inches=-0.1)
        print("data/cache/heatmap.png saved.")

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvt_color(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(np.array(image_data, dtype="float32") / 255.0, (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, image_shape,
                                                         self.confidence, self.nms_iou)
            if results[0] is None:
                return image, {}
            top_label = np.array(results[0][:, 6], dtype="int32")
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        font = ImageFont.truetype(font="data/あ.ttc", size=np.floor(3e-2 * image.size[1] + 0.5).astype("int32"))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
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
            image_info[key] = [f"{right - left}×{bottom - top}", np.round(float(score), 3)]
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
        image_data = np.expand_dims(np.transpose(np.array(image_data, dtype="float32") / 255.0, (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, image_shape,
                                               self.confidence, self.nms_iou)
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, image_shape,
                                                   self.confidence, self.nms_iou)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names):
        f = open(f"data/cache/map/.dr/{image_id}.txt", "w")
        image_shape = np.array(np.shape(image)[0:2])
        image = cvt_color(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(np.array(image_data, dtype="float32") / 255.0, (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, image_shape,
                                                         self.confidence, self.nms_iou)
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
            f.write(f"{predicted_class} {score[:6]} "
                    f"{str(int(left))} {str(int(top))} {str(int(right))} {str(int(bottom))}\n")
        f.close()
        return


class YoloBody(nn.Module):
    def __init__(self, num_classes):
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        super(YoloBody, self).__init__()
        self.backbone = Backbone()
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
        p3_down_sample = self.down_sample1(p3)
        p4 = torch.cat([p3_down_sample, p4], dim=1)
        p4 = self.make_five_conv3(p4)
        p4_down_sample = self.down_sample2(p4)
        p5 = torch.cat([p4_down_sample, p5], dim=1)
        p5 = self.make_five_conv4(p5)
        out2 = self.yolo_head3(p3)
        out1 = self.yolo_head2(p4)
        out0 = self.yolo_head1(p5)
        return out0, out1, out2


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, mosaic, mix_up, mosaic_prob,
                 mix_up_prob, train, special_aug_ratio=0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mix_up = mix_up
        self.mix_up_prob = mix_up_prob
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
            if self.mix_up and self.rand() < self.mix_up_prob:
                lines = sample(self.annotation_lines, 1)
                image_2, box_2 = self.get_random_data(lines[0], self.input_shape, random=self.train)
                image, box = self.get_random_data_with_mix_up(image, box, image_2, box_2)
        else:
            image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)
        image = np.transpose(np.array(image, dtype=np.float32) / 255.0, (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box

    @staticmethod
    def get_random_data_with_mix_up(image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes

    @staticmethod
    def merge_bbox(bbox_list, cut_x, cut_y):
        merge_bbox = []
        for i in range(len(bbox_list)):
            for box in bbox_list[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                if i == 0:
                    if x1 > cut_x or y1 > cut_y:
                        continue
                    if y1 <= cut_y <= y2:
                        y2 = cut_y
                    if x1 <= cut_x <= x2:
                        x2 = cut_x
                if i == 1:
                    if x1 > cut_x or y2 < cut_y:
                        continue
                    if y1 <= cut_y <= y2:
                        y1 = cut_y
                    if x1 <= cut_x <= x2:
                        x2 = cut_x
                if i == 2:
                    if x2 < cut_x or y2 < cut_y:
                        continue
                    if y1 <= cut_y <= y2:
                        y1 = cut_y
                    if x1 <= cut_x <= x2:
                        x1 = cut_x
                if i == 3:
                    if x2 < cut_x or y1 > cut_y:
                        continue
                    if y1 <= cut_y <= y2:
                        y2 = cut_y
                    if x1 <= cut_x <= x2:
                        x1 = cut_x
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    @staticmethod
    def rand(a=0.0, b=1.0):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=0.3, hue=0.1, sat=0.7, val=0.4, random=True):
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
        scale = self.rand(0.25, 2)
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
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image_data = np.array(image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        d_type = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(d_type)
        lut_sat = np.clip(x * r[1], 0, 255).astype(d_type)
        lut_val = np.clip(x * r[2], 0, 255).astype(d_type)
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

    def get_random_data_with_mosaic(self, annotation_line, input_shape, jitter=0.3, hue=0.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)
        image_list = []
        box_list = []
        index = 0
        for line in annotation_line:
            line_content = line.split()
            image = Image.open(line_content[0])
            image = cvt_color(image)
            iw, ih = image.size
            box = np.array([np.array(list(map(int, box.split(",")))) for box in line_content[1:]])
            flip = self.rand() < 0.5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]
            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(0.4, 1)
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
            image_list.append(image_data)
            box_list.append(box_data)
        cut_x = int(w * min_offset_x)
        cut_y = int(h * min_offset_y)
        new_image = np.zeros([h, w, 3])
        new_image[:cut_y, :cut_x, :] = image_list[0][:cut_y, :cut_x, :]
        new_image[cut_y:, :cut_x, :] = image_list[1][cut_y:, :cut_x, :]
        new_image[cut_y:, cut_x:, :] = image_list[2][cut_y:, cut_x:, :]
        new_image[:cut_y, cut_x:, :] = image_list[3][:cut_y, cut_x:, :]
        new_image = np.array(new_image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        d_type = new_image.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(d_type)
        lut_sat = np.clip(x * r[1], 0, 255).astype(d_type)
        lut_val = np.clip(x * r[2], 0, 255).astype(d_type)
        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
        new_boxes = self.merge_bbox(box_list, cut_x, cut_y)
        return new_image, new_boxes


class YoloLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, focal_loss=False, alpha=0.25, gamma=2):
        super(YoloLoss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio = 1 * (num_classes / 80)
        self.focal_loss = focal_loss
        self.focal_loss_ratio = 10
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_threshold = 0.5
        self.cuda = torch.cuda.is_available()

    @staticmethod
    def box_c_iou(b1, b2):
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_min = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_min = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half
        intersect_min = torch.max(b1_min, b2_min)
        intersect_max = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_max - intersect_min, torch.zeros_like(intersect_max))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / torch.clamp(union_area, min=1e-6)
        center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), dim=-1)
        enclose_min = torch.min(b1_min, b2_min)
        enclose_max = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_max - enclose_min, torch.zeros_like(intersect_max))
        enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), dim=-1)
        c_iou = iou - 1.0 * center_distance / torch.clamp(enclose_diagonal, min=1e-6)
        v = (4 / (math.pi ** 2)) * torch.pow(
            (torch.atan(b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
                b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
        alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
        c_iou -= alpha * v
        return c_iou

    @staticmethod
    def calculate_iou(_box_a, _box_b):
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

    @staticmethod
    def clip_by_tensor(t: torch.Tensor, t_min: float, t_max: float):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        return (result <= t_max).float() * result + (result > t_max).float() * t_max

    def bce_loss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def forward(self, idx, _input, targets=None):
        bs = _input.size(0)
        in_h = _input.size(2)
        in_w = _input.size(3)
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        prediction = _input.view(bs, len(self.anchors_mask[idx]), self.bbox_attrs, in_h, in_w).permute(
            0, 1, 3, 4, 2).contiguous()
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        y_true, no_obj_mask, _ = self.get_target(idx, targets, scaled_anchors, in_h, in_w)
        no_obj_mask, pred_boxes = self.get_ignore(idx, x, y, h, w, targets, scaled_anchors, in_h, in_w, no_obj_mask)
        if self.cuda:
            y_true = y_true.type_as(x)
            no_obj_mask = no_obj_mask.type_as(x)
        loss = 0
        obj_mask = y_true[..., 4] == 1
        n = torch.sum(obj_mask)
        if n != 0:
            c_iou = self.box_c_iou(pred_boxes, y_true[..., :4]).type_as(x)
            loss_loc = torch.mean((1 - c_iou)[obj_mask])
            loss_cls = torch.mean(self.bce_loss(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio
        if self.focal_loss:
            pos_neg_ratio = torch.where(obj_mask, torch.ones_like(conf) * self.alpha,
                                        torch.ones_like(conf) * (1 - self.alpha))
            hard_easy_ratio = torch.where(obj_mask, torch.ones_like(conf) - conf, conf) ** self.gamma
            loss_conf = torch.mean((self.bce_loss(conf, obj_mask.type_as(conf)) * pos_neg_ratio * hard_easy_ratio)[
                                       no_obj_mask.bool() | obj_mask]) * self.focal_loss_ratio
        else:
            loss_conf = torch.mean(self.bce_loss(conf, obj_mask.type_as(conf))[no_obj_mask.bool() | obj_mask])
        loss += loss_conf * self.balance[idx] * self.obj_ratio
        return loss

    def get_ignore(self, idx, x, y, h, w, targets, scaled_anchors, in_h, in_w, no_obj_mask):
        bs = len(targets)
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[idx])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[idx])), 1, 1).view(y.shape).type_as(x)
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[idx]]
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
                anchor_iou = self.calculate_iou(batch_target, pred_boxes_for_ignore)
                anchor_iou_max, _ = torch.max(anchor_iou, dim=0)
                anchor_iou_max = anchor_iou_max.view(pred_boxes[b].size()[:3])
                no_obj_mask[b][anchor_iou_max > self.ignore_threshold] = 0
        return no_obj_mask, pred_boxes

    def get_target(self, idx, targets, anchors, in_h, in_w):
        bs = len(targets)
        no_obj_mask = torch.ones(bs, len(self.anchors_mask[idx]), in_h, in_w, requires_grad=False)
        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[idx]), in_h, in_w, requires_grad=False)
        y_true = torch.zeros(bs, len(self.anchors_mask[idx]), in_h, in_w, self.bbox_attrs, requires_grad=False)
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
                if best_n not in self.anchors_mask[idx]:
                    continue
                k = self.anchors_mask[idx].index(best_n)
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                c = batch_target[t, 4].long()
                no_obj_mask[b, k, j, i] = 0
                y_true[b, k, j, i, 0] = batch_target[t, 0]
                y_true[b, k, j, i, 1] = batch_target[t, 1]
                y_true[b, k, j, i, 2] = batch_target[t, 2]
                y_true[b, k, j, i, 3] = batch_target[t, 3]
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 5] = 1
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, no_obj_mask, box_loss_scale
