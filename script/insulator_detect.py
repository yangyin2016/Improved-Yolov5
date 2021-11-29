import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from argparse import ArgumentParser
from tqdm import tqdm

from insulator import InsulatorItem


class InsulatorDetector:
    # 绝缘子串检测类
    # 使用方法:
    #   detector = InsulatorDetector()
    #   insulator_contour, insulator_points = detector.detect(img, bbox)
    
    def __init__(self):
        self.input_img = None
        self.insulators = []

    def __updateParam(self, img, bbox):
        # 更新检测需要的基本数据
        # @param img 输入的图片
        # @param bbox yolo格式的label

        self.input_img = img.copy()
        self.insulators = []
        if len(bbox.shape) == 1:
            bbox = np.expand_dims(bbox, axis=0)
        # 解析self.bbox内容，将对应的bbox中的绝缘子串切分出来保存在InsulatorItem中
        height, width, _ = self.input_img.shape
        for bx in bbox:
            category, x, y, w, h = bx
            x1, y1, x2, y2 = int(width * (x - w / 2)), int(height * (y - h / 2)), \
                        int(width * (x + w / 2)), int(height * (y + h / 2))
            self.insulators.append(InsulatorItem(img[y1 : y2, x1 : x2].copy(), [category, x1, y1, x2, y2]))
             

    def __identifyCategory(self):
        # 识别绝缘子串的种类

        # 提取中线
        mid_line_param = []
        for i, insulator in enumerate(self.insulators):
            k, b = insulator.reflectKbToGlobal()
            mid_line_param.append([k, b, i])
            # test
            p1 = (int((100 - b) // k), 100)
            p2 = (int((1400 - b) // k), 1400)
            cv2.line(self.input_img, p1, p2, (255, 0, 0), 2)
            plt.figure(figsize=(15, 8))
            plt.imshow(self.input_img)
            plt.show()

        mid_line_param = sorted(mid_line_param)
        
        # 根据共线寻找两段式（这段代码效率很低）
        angle_thresh_1 = 4
        b_thresh = 1000
        for i in range(0, len(mid_line_param)):
            k_i, b_i, index_i = mid_line_param[i]
            angle_i = np.degrees(np.arctan(k_i))
            angle_i = angle_i if angle_i > 0 else 180 + angle_i
            if index_i == -1:
                continue

            for j in range(i + 1, len(mid_line_param)):
                k_j, b_j, index_j = mid_line_param[j]
                angle_j = np.degrees(np.arctan(k_j))
                angle_j = angle_j if angle_j > 0 else 180 + angle_j
                    
                if abs(angle_i - angle_j) < angle_thresh_1 and abs(b_i - b_j) / max(k_i, k_j) < b_thresh:
                    self.insulators[index_i].insulator_type = 2 # 设置为两段
                    self.insulators[index_j].insulator_type = 2 
                    mid_line_param[i][2], mid_line_param[j][2] = -1, -1
          
        # 对于剩下的绝缘子串，继续分类
        angle_thresh_2 = 5
        for index, insulator in enumerate(self.insulators):
            if insulator.insulator_type != -1:
                continue
            angle_l = np.degrees(np.arctan(insulator.binary_left_edge.k))
            angle_r = np.degrees(np.arctan(insulator.binary_right_edge.k))
            angle_l = angle_l if angle_l > 0 else 180 + angle_l
            angle_r = angle_r if angle_r > 0 else 180 + angle_r
            if abs(angle_l - angle_r) < angle_thresh_2:
                self.insulators[index].insulator_type = 1
            else:
                self.insulators[index].insulator_type = 0

    
    def __identifyExposure(self):
        pass

    def __restore(self):
        pass

    def detect(self, img, bbox):
        # 检测图片中的绝缘子串的详细轮廓
        # @param img 输入的彩色绝缘子串图像
        # @param bbox 神经网络的输出，或者数据集中的label，格式为yolo格式
        # @param is_classified 是否已经分类完毕
        # return检测到的绝缘子串精确轮廓，以及包含在其中的点的坐标
        self.__updateParam(img, bbox)
        self.__identifyCategory()
        self.__identifyExposure()

        return [], []
    

"""def identifyCategory(img, show=False):
    # 分析绝缘子和过曝特征
    # return:[insulator_type, exposure_type]
    #   insulator_type:-1表示缺失，0表示变盘径，1表示固定盘径，2表示双段
    #   exposure_type:-1表示严重过曝无法修复，0表示正常，1表示轻微过曝可以修复

    insulator_type = -1
    exposure_type = -1

    # 预处理计算拟合边缘
    gray_img, binary_img = preProcess(img)
    left_edge, right_edge, mid_line_detected = detectEdge(binary_img)
    k_l, b_l, _, _= fitLine(left_edge)
    k_r, b_r, _, _= fitLine(right_edge)

    # 使用基于灰度拟合的边缘，来再次使用canny搜索边缘
    left_edge, right_edge, mid_line_detected = detectEdgeByCanny(img, [k_l, b_l], [k_r, b_r])
    k_l, b_l, left_edge_fitted, variance_l = fitLine(left_edge)
    k_r, b_r, right_edge_fitted, variance_r = fitLine(right_edge)

    # 计算绝缘子串片数
    pieces, gray_curve = calNumOfPieces(gray_img, left_edge, right_edge, mid_line_detected) 
    k_m, b_m, mid_line_fitted, variance_m = fitLine(mid_line_detected)
    if pieces < 9:
        return -1, -1

    # 1. 绝缘子串分类
    # 计算左右边角度差
    angle_l = np.degrees(np.arctan(k_l))
    angle_r = np.degrees(np.arctan(k_r))
    angle_thresh_1old = 5
    angle_diff = 0
    if angle_l * angle_r > 0:
        angle_diff = abs(angle_l - angle_r)
    else:
        angle_diff = 180 - abs(angle_l) - abs(angle_r)
    # 分类
    if angle_diff < angle_thresh_1old:
        insulator_type = 1
    else:
        insulator_type = 0

    # 2. 曝光程度分类
    # 计算片数
    real_pieces = 0
    if insulator_type== 0:
        real_pieces = 23
    elif insulator_type == 1:
        real_pieces = 32
    elif insulator_type == 2:
        pass
    else:
        return -1, -1

    # 分类
    varience_m_threshold = 60
    if (variance_m < varience_m_threshold) and (abs(real_pieces - pieces) < 5):
        exposure_type = 0
    elif (variance_m > varience_m_threshold) and (abs(real_pieces - pieces) < 5):
        exposure_type = 1
    elif (variance_m > varience_m_threshold) and (abs(real_pieces - pieces) > 5):
        exposure_type = -1
    else:
        exposure_type = -1

    # 3. plot
    if show:
        output = ""
        if insulator_type == 0:
            output += "变盘径,"
        elif insulator_type == 1:
            output += "单盘径,"
        elif insulator_type == 2:
            output += "双盘径,"
        else:
            output += "未知,"
        if exposure_type == -1:
            output += "无法恢复"
        elif exposure_type == 0:
            output += "正常"
        elif exposure_type == 1:
            output += "可以恢复"
        print(output)
        quickShow(img, [left_edge, left_edge_fitted, right_edge, right_edge_fitted], show=show)

    return insulator_type, exposure_type

"""
def test(path):
    # 测试绝缘子串分类和曝光分类的正确率 

    img_names = [] 
    for name in os.listdir(path):
        if name.endswith(("jpg", "png")):
            img_names.append(name)

    insulator_precision = 0
    exposure_precision = 0
    insulator_number = 0

    detector = InsulatorDetector()
    for img_name in tqdm(img_names):
        # 读取图片和label
        if not img_name.endswith(("jpg", "png")):
            continue
        prefix, suffix = img_name.split('.')
        label_name = prefix + ".txt"

        img = cv2.imread(os.path.join(path, img_name))
        label = np.loadtxt(os.path.join(path, label_name))

        # 运行检测器
        detector.detect(img, label)

        # 显示结果
        for insulator in detector.insulators:
            insulator_number += 1
            if insulator.insulator_type == insulator.bbox[0]:
                insulator_precision += 1
            else:
                print("分类错误！类别{}被分为{}".format(insulator.bbox[0], insulator.insulator_type))
                plt.figure(figsize=(15, 8))
                plt.subplot(1,2 ,1)
                plt.imshow(insulator.img.color_img)
                plt.subplot(1, 2, 2)
                plt.imshow(insulator.img.binary_img)
                plt.show()


    insulator_precision /= insulator_number
    print("共测试{}个目标，绝缘子串分类准确率:{}".format(insulator_number, insulator_precision))



def main(opt):
    path, output, show = opt.path, opt.output, opt.show
    img_names = os.listdir(path)

    test(path)

    """
    for img_name in tqdm(img_names):
        # 读取图片和label
        if not img_name.endswith(("jpg", "png")):
            continue
        prefix, suffix = img_name.split('.')
        label_name = prefix + ".txt"

        img = cv2.imread(os.path.join(path, img_name))
        label = np.loadtxt(os.path.join(path, label_name))
        print("读取图片{}，读取label{}".format(img_name, label_name))

        detector.test(img, label)

        # 检测
        insulator_contour, insulator_points = detector.detect(img, label)

        # 显示结果
        show_img = img.copy()
        for points in insulator_points:
            for point in points:
                cv2.circle(show_img, point, 1, (0, 0, 255), 1)
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(show_img)
        """

        
if __name__ == "__main__":
    parser = ArgumentParser(description="检测绝缘子串")
    parser.add_argument("--path", type=str, default="/home/fenghan/dataset/detect_dataset/normal", help="图片路径")
    parser.add_argument("--output", type=str, default="output", help="图片保存路径")
    parser.add_argument("--show", action='store_true', help="显示检测效果")

    main(parser.parse_args())

