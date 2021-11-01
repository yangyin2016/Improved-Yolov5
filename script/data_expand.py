# 根据image、background和label对原始数据进行扩充

import cv2
import numpy as np
import os
import argparse
from data_augment import DataAugment


class DataExpand:
    def __init__(self, input_path, bg_path, output_path):
        # 原始数据需要以img、label、bg三个目录的形式存放

        np.random.seed(0)        
        all_files, bg_files = os.listdir(input_path), os.listdir(bg_path)
        
        self.img_list = []
        self.label_list = []
        self.bg_list = []
        self.augmenter = DataAugment(is_augment=True)
        self.output_path = output_path

        # 解析原始图片、原始label、背景图片路径和有效性
        for f in all_files:
            prefix, suffix = f.split('.')
            if suffix not in ['jpg', 'png']:
                continue
            label = prefix + '.txt'
            if label not in all_files:
                continue
            self.img_list.append(os.path.join(input_path, f))
            self.label_list.append(os.path.join(input_path, label))

        for f in bg_files:
            prefix, suffix = f.split('.')
            if suffix in ['jpg', 'png']:
                self.bg_list.append(os.path.join(bg_path, f))

    def createNewData(self, number=100):
        # 创造新的数据样本
        n = 0
        while n < number:
            index_1 = int(np.random.randint(0, len(self.img_list)))
            img = cv2.imread(self.img_list[index_1])
            label_path = self.label_list[index_1]
            height, width, _ = img.shape

            bboxs = self.parseLabel(label_path, height, width)
            for bbox in bboxs:
                if n >= number:
                    break
                n += 1

                # 取出并增强bbox
                img_split = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                img_split = self.augmenter.augment(img_split) 

                # 融合bg
                index_2 = int(np.random.randint(0, len(self.bg_list)))
                bg = cv2.imread(self.bg_list[index_2])
                img_new, label_new = self.mergeWithBg(img_split, bg)

                # save
                cv2.imwrite(os.path.join(self.output_path, "{:0>4d}".format(n)) + ".jpg", img_new)
                np.savetxt(os.path.join(self.output_path, "{:0>4d}".format(n)) + ".txt", label_new)

        print("成功创造新样本{}个，保存在{}".format(n, self.output_path))

    def mergeWithBg(self, bbox, bg):
        # 将bbox和bg融合,返回新的样本和label
        h1, w1, _ = bbox.shape 
        h2, w2, _ = bg.shape
        label = []
        
        # 将bbox缩放到合适的大小,bbox的长边占0.3-0.8的大小
        r1 = max(h1 / h2, w1 / w2)
        r2 = np.random.rand() * 0.5 + 0.3   # 缩放到0.3-0.8       
        ratio = r2 / r1
        h, w = int(ratio * h1), int(ratio * w1)
        bbox = cv2.resize(bbox, (w, h))

        # 随机嵌入bbox
        x1, y1 = np.random.randint(0, w2 - w), np.random.randint(0, h2 - h)
        #bg[y1 : y1 + h, x1 : x1 + w] = bbox 
        bg[y1 : y1 + h, x1 : x1 + w] = bg[y1 : y1 + h, x1 : x1 + w] * 0.03 + bbox * 0.97
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
