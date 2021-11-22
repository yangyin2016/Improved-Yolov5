import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from argparse import ArgumentParser
from tqdm import tqdm

from ransac import *
from utils import *

def preProcess(img):
    # description:解析输入的图片路径，给出路径中可用的图像数据
    # return:[灰度图，二值图]

    # 滤波、二值化
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #img_gray = cv2.add(img_gray, -30)
    thresh, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    img_binary = cv2.dilate(img_binary, kernel, iterations=2)
    img_binary = cv2.erode(img_binary, kernel, iterations=2)

    # 寻找最大连通区域,将其余的连通区域清零
    contours, hiterarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    max_idx = -1 if len(area) == 0 else np.argmax(area) 
    for i in range(len(contours)):
        if i != max_idx:
            cv2.fillPoly(img_binary, [contours[i]], 0)

    return img_gray, img_binary

def detectEdge(img_binary):
    # description:分割绝缘子串，提取左右边缘坐标，计算中线坐标
    # return:[左侧边缘坐标，右侧边缘坐标，中线坐标]

    left_edge = []
    right_edge = []

    contours, hiterarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        return left_edge, right_edge

    # 遍历边界点，寻找左右边缘
    total_points = {}
    lower = contours[0][0][0][1]
    upper = contours[0][0][0][1]
    for point in contours[0]:
        x, y = point[0]
        lower = min(lower, y)
        upper = max(upper, y)
        if y not in total_points.keys():
            total_points[y] = []
            total_points[y].append(x) 
        elif len(total_points[y]) == 1:
            if total_points[y][0] < x:
                total_points[y].append(x)
            else:
                total_points[y].append(total_points[y][0])
                total_points[y][0] = x
        else:
            if x < total_points[y][0]:
                total_points[y][0] = x
            elif x > total_points[y][1]:
                total_points[y][1] = x
        
    # 将左右边缘转换成列表的形式
    for row in range(lower, upper + 1):
        if (row not in total_points.keys()) or (len(total_points[row]) != 2):
            continue
        left_edge.append([total_points[row][0], row])
        right_edge.append([total_points[row][1], row])
    
    # 计算中线
    if len(left_edge) != len(right_edge):
        return left_edge, right_edge, []
    n = len(left_edge)
    mid_line_detected = []

    for i in range(n):
        mid_line_detected.append([(left_edge[i][0] + right_edge[i][0]) // 2, \
                (left_edge[i][1] + right_edge[i][1]) // 2])
    return left_edge, right_edge, mid_line_detected
 
def fitLine(line):
    # 拟合线段
    # 返回：直线参数，拟合后的线段坐标，以及拟合线段和原线段的平方差

    n = len(line)
    line_fitted = []
    variance = 0.0
    k, b = randomSampleConsensus(line[int(0.1 * n) : int(0.9 * n)])

    for i in range(n):
        x, y = int((line[i][1] - b) // k), line[i][1]
        line_fitted.append([x, y])
        if i > 0.1 * n and i < 0.9 * n:
            variance += np.square((line[i][0] - x))
    variance /= (0.8 * n)

    return k, b, line_fitted, variance

def split(img):
    # description:将绝缘子串从图像中分割出来

    img_gray, img_binary = preProcess(img)
    img[np.where(img_binary == 0)] = 0
    return img

def calDeviation(width, mid_line_detected, mid_line_fitted):
    # description:计算理论中线与实际中线的偏离程度
    if len(mid_line_detected) != len(mid_line_fitted):
        return 1

    k = 100
    n = len(mid_line_detected)
    variance = 0
    for i in range(n):
        variance += np.square(mid_line_fitted[i][0] - mid_line_detected[i][0]) 
    variance = k * mse / n / np.square(width)

    return variance

def calNumOfPieces(img_gray, left_edge, right_edge, mid_line):
    # description:根据绝缘子串的灰度图和其左右边缘计算其绝缘子片的数量
    # 返回绝缘子串的片数

    # 图像锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    img_gray = cv2.filter2D(img_gray, -1, kernel)

    # 计算灰度曲线并滤波
    mid_line= np.array(mid_line)
    gray_value = img_gray[mid_line[:, 1], mid_line[:, 0]]   # 中线上的灰度值

    # 计算极值点数量
    gray_value_filted = lowPassFilter(gray_value, ratio=0.4)
    peak_index = calNumOfMaximumValue(gray_value, size=10)
    num = peak_index.shape[0]

    return num, gray_value_filted

def identifyCategory(img, show=False):
    # 分析绝缘子和过曝特征
    # return:[insulator_type, exposure_type]
    #   insulator_type:-1表示缺失，0表示变盘径，1表示固定盘径，2表示双段
    #   exposure_type:-1表示严重过曝无法修复，0表示正常，1表示轻微过曝可以修复

    insulator_type = -1
    exposure_type = -1

    # 预处理
    img_gray, img_binary = preProcess(img)
    left_edge, right_edge, mid_line_detected = detectEdge(img_binary) 
    k_m, b_m, mid_line_fitted, variance_m = fitLine(mid_line_detected)
    pieces, gray_curve = calNumOfPieces(img_gray, left_edge, right_edge, mid_line_detected) 
    if pieces < 9:
        return -1, -1
    
    # 拟合边缘
    k_l, b_l, left_edge_fitted, variance_l = fitLine(left_edge)
    k_r, b_r, right_edge_fitted, variance_r = fitLine(right_edge)
    angle_l = np.degrees(np.arctan(k_l))
    angle_r = np.degrees(np.arctan(k_r))

    # 1. 绝缘子串分类
    # 计算左右边角度差
    angle_threshold = 5
    angle_diff = 0
    if angle_l * angle_r > 0:
        angle_diff = abs(angle_l - angle_r)
    else:
        angle_diff = 180 - abs(angle_l) - abs(angle_r)
    # 分类
    if angle_diff < angle_threshold:
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
        quickShow(img, [left_edge, left_edge_fitted, right_edge, right_edge_fitted], show=show)

    return insulator_type, exposure_type

def main(opt):
    path, output, show = opt.path, opt.output, opt.show
    img_names = os.listdir(path)

    N = 0
    n = 0
    for img_name in tqdm(img_names):
        if not img_name.endswith(".jpg"):
            continue
        if img_name.startswith("2"):
            continue
        N += 1
        img = cv2.imread(os.path.join(path, img_name))
        category, exposure_type = identifyCategory(img, show)
        if category == 0 and img_name.startswith("0"):
            n += 1
        elif category == 1 and img_name.startswith("1"):
            n += 1
    print("准确率:{}".format(n / N))
        
if __name__ == "__main__":
    parser = ArgumentParser(description="检测绝缘子串")
    parser.add_argument("--path", type=str, default="test_data/normal", help="图片路径")
    parser.add_argument("--output", type=str, default="output", help="图片保存路径")
    parser.add_argument("--show", action='store_true', help="显示检测效果")

    main(parser.parse_args())

