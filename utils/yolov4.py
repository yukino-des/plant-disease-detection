import colorsys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxsim
import os
import time
import torch
from collections import OrderedDict
from PIL import ImageDraw, ImageFont
from torch import nn
from utils.mobilenetv2 import mobilenet_v2
from utils.util import cvt_color, get_anchors, get_classes, preprocess_input, resize_image, show_config, logistic
from utils.bbox import DecodeBox


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetV2, self).__init__()
        self.model = mobilenet_v2(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:14](out3)
        out5 = self.model.features[14:18](out4)
        return out3, out4, out5


def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out,
                           kernel_size=kernel_size, stride=stride, padding=pad, groups=groups, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True)),
    ]))


def conv_dw(filter_in, filter_out, stride=1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),
        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True),
    )


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


def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv_dw(in_filters, filters_list[0]),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, backbone="mobilenetv2", pretrained=False):
        super(YoloBody, self).__init__()
        if backbone == "mobilenetv2":
            self.backbone = MobileNetV2(pretrained=pretrained)
            in_filters = [32, 96, 320]
        else:
            raise ValueError(
                "Unsupported backbone - `{}`, Use mobilenetv2".format(backbone))
        self.conv1 = make_three_conv([512, 1024], in_filters[2])
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512, 1024], 2048)
        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = conv2d(in_filters[1], 256, 1)
        self.make_five_conv1 = make_five_conv([256, 512], 512)
        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = conv2d(in_filters[0], 128, 1)
        self.make_five_conv2 = make_five_conv([128, 256], 256)
        self.yolo_head3 = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)], 128)
        self.down_sample1 = conv_dw(128, 256, stride=2)
        self.make_five_conv3 = make_five_conv([256, 512], 512)
        self.yolo_head2 = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)], 256)
        self.down_sample2 = conv_dw(256, 512, stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)
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


class YOLO(object):
    _defaults = {
        "model_path": "../data/best_epoch_weights.pth",
        "classes_path": "../data/voc_classes.txt",
        "anchors_path": "../data/yolo_anchors.txt",
        "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        "input_shape": [416, 416],
        "backbone": "mobilenetv2",
        "confidence": 0.5,
        "nms_iou": 0.3,
        "letterbox_image": False,
        # todo update `"cuda": True`
        "cuda": False
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name "" + n + """

    def __init__(self, **kwargs):
        self.classes_path = ""
        self.anchors_path = ""
        self.input_shape = []
        self.anchors_mask = []
        self.backbone = ""
        self.model_path = ""
        self.cuda = False
        self.letterbox_image = False
        self.confidence = 0
        self.nms_iou = 0
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors,
                                   self.num_classes,
                                   (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        show_config(**self._defaults)

    def generate(self, onnx=False):
        self.net = YoloBody(self.anchors_mask, self.num_classes, self.backbone)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, crop=False, count=False):
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
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1),
                                                         self.num_classes,
                                                         self.input_shape,
                                                         image_shape,
                                                         self.letterbox_image,
                                                         conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
            if results[0] is None:
                return image, {}
            top_label = np.array(results[0][:, 6], dtype="int32")
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        font = ImageFont.truetype(font="../data/simhei.ttf",
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype("int32"))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], ":", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
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
                print("save crop_" + str(i) + ".png to " + dir_save_path)
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
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            self.bbox_util.non_max_suppression(torch.cat(outputs, 1),
                                               self.num_classes,
                                               self.input_shape,
                                               image_shape,
                                               self.letterbox_image,
                                               conf_thres=self.confidence,
                                               nms_thres=self.nms_iou)
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                self.bbox_util.non_max_suppression(torch.cat(outputs, 1),
                                                   self.num_classes,
                                                   self.input_shape,
                                                   image_shape,
                                                   self.letterbox_image,
                                                   conf_thres=self.confidence,
                                                   nms_thres=self.nms_iou)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        image = cvt_color(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
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
        print("Save to " + heatmap_save_path)

    def convert_to_onnx(self, simplify, model_path):
        self.generate(onnx=True)
        im = torch.zeros(1, 3, *self.input_shape).to("cpu")
        input_layer_names = ["images"]
        output_layer_names = ["output"]
        print(f"Onnx {onnx.__version__}.")
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)
        model_onnx = onnx.load(model_path)
        onnx.checker.check_model(model_onnx)
        if simplify:
            print(f"Onnx-simplifier {onnxsim.__version__}.")
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, "assert check failed"
            onnx.save(model_onnx, model_path)
        print("Save to {}".format(model_path))

    def get_map_txt(self, image_id, image, class_names, maps_out_path):
        f = open(os.path.join(maps_out_path, "detection/" + image_id + ".txt"), "w")
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
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1),
                                                         self.num_classes,
                                                         self.input_shape,
                                                         image_shape,
                                                         self.letterbox_image,
                                                         conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
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
