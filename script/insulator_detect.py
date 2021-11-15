import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from argparse import ArgumentParser

from ransac import line_fitting
from utils import *

def preProcess(img):
    # description:解析输入的图片路径，给出路径中可用的图像数据
    # return:[灰度图，二值图]

    # 滤波、二值化
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
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
 
def fitLine(line, max_iterations=30, stop_at_goal=True, random_seed=0):
    # 拟合线段
    # 返回：直线参数，拟合后的线段坐标

    n = len(line)
    line_fitted = []
    goal_inliers = max(100, int(0.6 * n))
    m = line_fitting(line, threshold=0.01, sample_size=int(0.05 * n), goal_inliers=goal_inliers, \
                    max_iterations=max_iterations, stop_at_goal=stop_at_goal, random_seed=random_seed)
    k = -m[0] / (m[1] + 1e-8)
    b = -m[2] / (m[1] + 1e-8)
    for i in range(n):
        line_fitted.append([int((line[i][1] - b) // k), line[i][1]])

    return k, b, line_fitted

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
    mse = 0
    for i in range(n):
        mse += np.square(mid_line_fitted[i][0] - mid_line_detected[i][0]) 
    mse = k * mse / n / np.square(width)

    return mse

def calNumOfPieces(img_gray, left_edge, right_edge):
    # description:根据绝缘子串的灰度图和其左右边缘计算其绝缘子片的数量

    # 图像锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    img_gray = cv2.filter2D(img_gray, -1, kernel)

    # 计算灰度曲线并滤波
    left_edge = np.array(left_edge)
    right_edge = np.array(right_edge)
    mid_edge = (left_edge + right_edge) // 2

    gray_value = img_gray[mid_edge[:, 1], mid_edge[:, 0]]   # 中线上的灰度值

    # 求导
    gray_value_filted = medianFilter(gray_value, 3)
    num = calNumOfMaximumValue(gray_value, size=10)
    
    print("片数:{}".format(num))
    # 显示
    if True:
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 3, 1)
        plt.imshow(img_gray)
        plt.subplot(1, 3, 2)
        plt.plot(gray_value)
        plt.subplot(1, 3, 3)
        plt.plot(gray_value_filted)
        plt.show()
        
    return 10

def main(opt):
    path, output, show = opt.path, opt.output, opt.show
    img_names = os.listdir(path)

    for img_name in img_names:
        if not img_name.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(path, img_name))
        t1 = timer()

        img_gray, img_binary = preProcess(img)
        left_edge, right_edge, mid_line = detectEdge(img_binary) 
        k, b, mid_line_fitted = fitLine(mid_line)
        calNumOfPieces(img_gray, left_edge, right_edge)

       
        t2 = timer()
        print("total cost timer:{}".format(t2 - t1))

        
if __name__ == "__main__":
    parser = ArgumentParser(description="检测绝缘子串")
    parser.add_argument("--path", type=str, default="test_data/normal", help="图片路径")
    parser.add_argument("--output", type=str, default="output", help="图片保存路径")
    parser.add_argument("--show", action='store_true', help="显示检测效果")

    main(parser.parse_args())

