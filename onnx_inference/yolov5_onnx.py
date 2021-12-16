# coding=utf-8
import sys

from numpy.lib.utils import source
sys.path.append('..')
from utils.general import non_max_suppression

import cv2
import os
import numpy as np
from numpy.core.numeric import outer
import onnxruntime
import torch
import torchvision
import time
import random
from argparse import ArgumentParser
from tqdm import tqdm

class YOLOV5_ONNX(object):
    def __init__(self,onnx_path, shape=(640, 640), class_num=1, batch_size=1):
        '''初始化onnx'''
        device = onnxruntime.get_device()
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.shape = shape
        self.class_num = class_num
        self.batch_size = 1

        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()
        print("device:{}, input name:{}, output name:{}".format(device, self.input_name, self.output_name))

        anchor_list= [[10,13, 16,30, 33,23],[30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]
        self.anchor = np.array(anchor_list).astype(np.float32).reshape(3,-1,2)

        # 推理一次获取输出大小
        input_tensor = np.random.randn(batch_size, 3, shape[0], shape[1]).astype(np.float32)
        input_feed = self.get_input_feed(input_tensor)
        preds = self.onnx_session.run(output_names=self.output_name,input_feed=input_feed)
        self.output_size = [pred.shape[2:4] for pred in preds[1:]]

        stride=[8,16,32]
        area = shape[0] * shape[1]
        self.size = [int(area / stride[0] ** 2), int(area / stride[1] ** 2), int(area / stride[2] ** 2)]

    def get_input_name(self):
        '''获取输入节点名称'''
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self):
        '''获取输出节点名称'''
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_feed(self, image_tensor):
        '''获取输入tensor'''
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_tensor

        return input_feed

    def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True,
                  stride=32):
        '''填充图片使其缩放至new_shape大小'''
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)
    
    def img2input(self, img):
        """
        将图像转换为可以输入模型的格式 
        """
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img).astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img,axis=0)
        return img

    def output2bbox(self, preds):
        """
        将网络的输出转换为bbox格式 
        """
        y = []
        y.append(torch.tensor(preds[1].reshape(preds[1].shape[0], -1, 5 + self.class_num)).sigmoid())
        y.append(torch.tensor(preds[2].reshape(preds[2].shape[0], -1, 5 + self.class_num)).sigmoid())
        y.append(torch.tensor(preds[3].reshape(preds[3].shape[0], -1, 5 + self.class_num)).sigmoid())

        # 处理输出
        grid = []
        stride = [8, 16, 32]
        for f in self.output_size:
            grid.append([[i, j] for j in range(f[0]) for i in range(f[1])])

        size = self.size
        z = []
        for i in range(len(y)):
            src = y[i]
            xy = src[..., 0:2] * 2. - 0.5
            wh = (src[..., 2:4] * 2) ** 2
            dst_xy = []
            dst_wh = []
            for j in range(len(y)):
                dst_xy.append((xy[:, j * size[i]:(j + 1) * size[i], :] + torch.tensor(grid[i])) * stride[i])
                dst_wh.append(wh[:, j * size[i]:(j + 1) * size[i], :] * self.anchor[i][j])
            src[..., 0:2] = torch.from_numpy(np.concatenate((dst_xy[0], dst_xy[1], dst_xy[2]), axis=1))
            src[..., 2:4] = torch.from_numpy(np.concatenate((dst_wh[0], dst_wh[1], dst_wh[2]), axis=1))
            z.append(src.view(1, -1, 5 + self.class_num))

        results = torch.cat(z, 1)
        return results

    def xywh2xyxy(self,x):
        '''将网络输出的[x,y,w,h]格式转换为[x1,y1,x2,y2]格式'''
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        return y

    def clip_coords(self,boxes, img_shape):
        '''查看bbox是否越界'''
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2

    def scale_coords(self,img1_shape, coords, img0_shape, ratio_pad=None):
        '''
        坐标对应到原始图像上，反操作:减去pad，除以最小缩放比例
        :param img1_shape: 输入尺寸
        :param coords: 输入坐标
        :param img0_shape: 映射的尺寸
        :param ratio_pad:
        :return:
        '''
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new,计算缩放比率
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                        img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding ，计算扩充的尺寸
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding，减去x方向上的扩充
        coords[:, [1, 3]] -= pad[1]  # y padding，减去y方向上的扩充
        coords[:, :4] /= gain  # 将box坐标对应到原始图像上
        self.clip_coords(coords, img0_shape)  # 边界检查
        return coords
    
    def inference(self, img, conf_thres=0.25):
        """
        对输入的img进行推理，输出目标的bbox坐标信息
        """
        start = time.time()

        input_feed = self.get_input_feed(img)
        preds = self.onnx_session.run(output_names=self.output_name,input_feed=input_feed)
        results = self.output2bbox(preds)
        results = non_max_suppression(results, conf_thres=conf_thres, iou_thres=0.45)

        end = time.time()
        print("cost time:{}".format(end - start))

        return results

    def test(self, img_path, label_path, conf_thres=0.25):
        """
        测试不同尺度的检测精度，并绘制点阵图
        """
        print("正在处理label")
        source_list = []
        img_list, label_list = os.listdir(img_path), os.listdir(label_path)
        for img_name in tqdm(img_list):
            prefix, suffix = img_name.split('.')
            label_name = prefix + '.txt'
            if label_name in label_list:
                source_list.append((os.path.join(img_path, img_name), os.path.join(label_path, label_name)))
        
        print("正在评估指标")
        for (img_path, label_path) in tqdm(source_list):
            src_img = cv2.imread(img_path)
            src_size = src_img.shape[:2]

            # 预处理、归一化、维度扩张
            img, ratio, (dw, dh) = self.letterbox(src_img, self.shape, stride=32)
            img = self.img2input(img)

            # 前向推理
            results = self.inference(img, conf_thres)


    def detect(self, source_path, show=False, conf_thres=0.25):
        '''
        检测对应路径的图像 
        '''
        # 判断source_path的类型：图片坐标、图片目录或者视频
        source_list = []
        if os.path.isdir(source_path):
            for name in os.listdir(source_path):
                if name.endswith(('jpg', 'png')):
                    source_list.append(os.path.join(source_path, name)) 
        elif source_path.endswith(('jpg', 'png')):
            source_list.append(source_path)
        elif source_path.endswith(('mp4', 'avi')):
            print("暂时不支持视频格式:{}".format(source_path))
        else:
            print("无法识别的格式:{}".format(source_path))

        # 读取图片
        for path in source_list:
            src_img = cv2.imread(path)
            src_size = src_img.shape[:2]

            # 预处理、归一化、维度扩张
            img, ratio, (dw, dh) = self.letterbox(src_img, self.shape, stride=32)
            img = self.img2input(img)

            # 前向推理
            results = self.inference(img, conf_thres)

            #映射到原始图像
            img_shape=img.shape[2:]
            for det in results:  # detections per image
                if det is not None and len(det):
                    det[:, :4] = self.scale_coords(img_shape, det[:, :4],src_size).round()
            if det is not None and len(det):
                self.draw(src_img, det)

    def plot_one_box(self,x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def draw(self,img, boxinfo):
        for *xyxy, conf, cls in boxinfo:
            label = '%s %.2f' % ('image', conf)
            #print('xyxy: ', xyxy)
            #print("img.shape:{}, xyxy:{}".format(img.shape, xyxy))
            self.plot_one_box(xyxy, img, label=label, color=(0, 0, 255), line_thickness=1)

        cv2.namedWindow("dst",0)
        cv2.imshow("dst", img)
        cv2.imwrite("data/res1.jpg",img)
        cv2.waitKey(0)
        # cv2.imencode('.jpg', img)[1].tofile(os.path.join(dst, id + ".jpg"))
        return 0

def main(opt):
    weights, source, img_size, class_num = opt.weights, opt.source, opt.img_size, opt.class_num
    model=YOLOV5_ONNX(onnx_path=weights, shape=(img_size, img_size), class_num=class_num)
    model.detect(img_path=source)

if __name__=="__main__":
    parser = ArgumentParser(usage="python3 {} --weights --source")
    parser.add_argument("--weights", default="./yolov5s.onnx", help="onnx模型路径") 
    parser.add_argument("--source", default="./cat.jpg", help="待检测的图片")
    parser.add_argument("--img_size", default=640, help="模型输入尺寸")
    parser.add_argument("--class_num", type=int, default=80, help="模型输入尺寸")

    main(parser.parse_args())
