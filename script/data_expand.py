# 根据image、background和label对原始数据进行扩充

import cv2
import numpy as np
import os
import argparse
from data_augment import DataAugment
from insulator_detect import preProcess
from tqdm import tqdm

def split(img):
    # description:将绝缘子串从图像中分割出来
    img_gray, img_binary = preProcess(img)
    img[np.where(img_binary == 0)] = 0
    return img

class DataExpand:
    def __init__(self, bbox_path, bg_path, result_path):
        # 原始数据需要以img、label、bg三个目录的形式存放

        np.random.seed(0)        
        bbox_files, bg_files = os.listdir(bbox_path), os.listdir(bg_path)
        
        self.bbox_list = []
        self.bg_list = []
        self.result_path = result_path
        self.augmenter = DataAugment(is_augment=True)

        # 解析原始图片、原始label、背景图片路径和有效性
        for bbox in bbox_files:
            prefix, suffix = bbox.split('.')
            if suffix not in ['jpg', 'png']:
                continue
            self.bbox_list.append(os.path.join(bbox_path, bbox))

        for bg in bg_files:
            prefix, suffix = bg.split('.')
            if suffix in ['jpg', 'png']:
                self.bg_list.append(os.path.join(bg_path, bg))

    def createNewData(self, number=100):
        # description:创造新的样本数据
        for i in tqdm(range(0, number)):
            # 读取并分割绝缘子串，读取背景图片
            index = i % len(self.bbox_list)  
            img = cv2.imread(self.bbox_list[index])
            height, width, _ = img.shape
            img_splited = split(img)

            # 增强图像
            img_splited = self.augmenter.augment(img_splited)
            
            # 融合图像
            index = int(np.random.randint(0, len(self.bg_list)))
            bg = cv2.imread(self.bg_list[index])
            img_new, label_new = self.mergeWithBg(img_splited, bg) 

            # 保存
            cv2.imwrite(os.path.join(self.result_path, "{:0>4d}".format(i)) + ".jpg", img_new)
            np.savetxt(os.path.join(self.result_path, "{:0>4d}".format(i)) + ".txt", label_new)

        print("成功创造新样本{}个，保存在{}".format(i + 1, self.result_path))

    def mergeWithBg(self, bbox, bg):
        # 将bbox和bg融合,返回新的样本和label
        h1, w1, _ = bbox.shape 
        h2, w2, _ = bg.shape
        label = []
        
        # 将bbox缩放到合适的大小,bbox的长边占0.3-0.9的大小
        r1 = max(h1 / h2, w1 / w2)
        r2 = np.random.rand() * 0.6 + 0.3   # 缩放到0.3-0.9       
        ratio = r2 / r1
        h, w = int(ratio * h1), int(ratio * w1)
        bbox = cv2.resize(bbox, (w, h))

        # 随机嵌入bbox
        x1, y1 = np.random.randint(0, w2 - w), np.random.randint(0, h2 - h)
        temp = bg[y1 : y1 + h, x1 : x1 + w]
        temp[(bbox != 0).any(axis=2)] = bbox[(bbox != 0).any(axis=2)]
        #bg[y1 : y1 + h, x1 : x1 + w] = bg[y1 : y1 + h, x1 : x1 + w] * 0.03 + bbox * 0.97
        bg[y1 : y1 + h, x1 : x1 + w] = temp
        label = np.array([[0, (x1 + w / 2) / w2, (y1 + h / 2) / h2, w / w2, h / h2]], )

        return bg, label

    def parseLabel(self, label_path, height, width):
        # 解析label，返回bbox
        bbox = []

        label = np.loadtxt(label_path)
        if len(label.shape) == 1:
            label = label.reshape(1, 5)
        for line in label:
            x, y, w, h = line[1:]
            x1, y1, x2, y2 = int(width * (x - w / 2)), int(height * (y - h / 2)), \
                            int(width * (x + w / 2)), int(height * (y + h / 2))
            bbox.append([x1, y1, x2, y2])
        return bbox

def main(opt):
    input_path, bg_path, output_path, num = opt.input_path, opt.bg_path, opt.output_path, opt.num

    expander = DataExpand(input_path, bg_path, output_path)
    expander.createNewData(num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/home/fenghan/dataset/images", help="input path")
    parser.add_argument("--bg_path", type=str, default="/home/fenghan/dataset/background", help="bg path")
    parser.add_argument("--output_path", type=str, default="/home/fenghan/dataset/augmented/", help="output path")
    parser.add_argument("--num", type=int, default=100, help="output path")

    main(parser.parse_args())
