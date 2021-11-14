import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from argparse import ArgumentParser

from ransac import line_fitting

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

def extractEdge(img_binary):
    # description:分割绝缘子串，提取左右边缘坐标
    # return:[左侧边缘坐标，右侧边缘坐标]

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

    return left_edge, right_edge

def findMidLine(left_edge, right_edge):
    # description:寻找绝缘子串的实际中线与理论中线
    if len(left_edge) != len(right_edge):
        return [], []
    n = len(left_edge)
    mid_line_detected = []
    mid_line_fitted = []
    
    for i in range(n):
        mid_line_detected.append([(left_edge[i][0] + right_edge[i][0]) // 2, \
                (left_edge[i][1] + right_edge[i][1]) // 2])

    # 使用RANSAC拟合
    goal_inliers = max(100, int(0.6 * n))
    m = line_fitting(mid_line_detected, threshold=0.01, sample_size=int(0.05 * n), goal_inliers=goal_inliers, \
                    max_iterations=30, stop_at_goal=True, random_seed=0)
    k = -m[0] / (m[1] + 1e-8)
    b = -m[2] / (m[1] + 1e-8)
    print("k:{}, b:{}".format(k, b))
    for i in range(n):
        mid_line_fitted.append([int((mid_line_detected[i][1] - b) // k), mid_line_detected[i][1]])

    return mid_line_detected, mid_line_fitted

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

    def lowPassFilter(data, ratio=0.2):
        # description:一个简单的低通滤波
        if len(data) == 0:
            return data
        prev_value = data[0]
        data_ret = data.copy()
        for i in range(len(data)):
            data_ret[i] = ratio * data_ret[i] + (1 - ratio) * prev_value
            prev_value = data_ret[i]
        return data_ret

    def derivative(data):
        # description:一个基本的求导功能实现
        if len(data) == 0:
            return data
        data_ret = []
        for i in range(1, len(data) - 1):
            data_ret.append((int(data[i + 1]) - int(data[i - 1])) / 2)
        return data_ret

    # 计算灰度曲线并滤波
    left_edge = np.array(left_edge)
    right_edge = np.array(right_edge)
    mid_edge = (left_edge + right_edge) // 2

    gray_value = img_gray[mid_edge[:, 1], mid_edge[:, 0]]   # 中线上的灰度值
    gray_value_filted = lowPassFilter(gray_value, ratio=0.1)

    # 求导
    deriva = derivative(gray_value_filted) 
        
    return 10

def main(opt):
    path, show = opt.path, opt.show
    img_names = os.listdir(path)

    for img_name in img_names:
        if not img_name.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(path, img_name))
        """t1 = timer()

        img_gray, img_binary = preProcess(img)
        left_edge, right_edge = extractEdge(img_binary) 
        num = calNumOfPieces(img_gray, left_edge, right_edge)
        mid_line_detected, mid_line_fitted = findMidLine(left_edge, right_edge)
        deviation = calDeviation(img.shape[1], mid_line_detected, mid_line_fitted)
        print("devitation:{}".format(deviation))

        t2 = timer()
        print("total cost timer:{}".format(t2 - t1))

        output = "output"
        # 绘图
        for i in range(len(left_edge)):
            cv2.circle(img, left_edge[i], 0, (0, 0, 255), 3)
            cv2.circle(img, right_edge[i], 0, (0, 0, 255), 3)
        for i in range(len(mid_line_detected)):
            cv2.circle(img, mid_line_detected[i], 1, (0, 255, 0), 1)
            cv2.circle(img, mid_line_fitted[i], 1, (255, 0, 0), 1)
        cv2.imwrite(os.path.join(output, img_name), img) 
        """
        img = split(img)
        cv2.imwrite(os.path.join("output", img_name), img) 
        
if __name__ == "__main__":
    parser = ArgumentParser(description="检测绝缘子串")
    parser.add_argument("--path", type=str, default="images", help="图片路径")
    parser.add_argument("--output", type=str, default="output", help="图片保存路径")
    parser.add_argument("--show", action='store_true', help="显示检测效果")

    main(parser.parse_args())

