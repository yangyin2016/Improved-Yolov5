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
 
def fitLine(line, max_iterations=30, stop_at_goal=True, random_seed=0):
    # 拟合线段
    # 返回：直线参数，拟合后的线段坐标，以及拟合线段和原线段的平方差

    line_cropped = line.copy()
    line_cropped = line_cropped[int(0.1 * len(line_cropped)): int(0.9 * len(line_cropped))]
    n = len(line_cropped)
    line_fitted = []
    mse = 0
    goal_inliers = max(100, int(0.6 * n))
    m = line_fitting(line_cropped, threshold=0.01, sample_size=int(0.1 * n), goal_inliers=goal_inliers, \
                    max_iterations=max_iterations, stop_at_goal=stop_at_goal, random_seed=random_seed)
    k = -m[0] / (m[1] + 1e-8)
    b = -m[2] / (m[1] + 1e-8)

    for i in range(len(line)):
        x, y = int((line[i][1] - b) // k), line[i][1]
        line_fitted.append([x, y])
        if i > 0.1 * n and i < 0.9 * n:
            mse += np.square((line[i][0] - x))
    mse /= (0.8 * n)

    return k, b, line_fitted, mse

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

def calNumOfPieces(img_gray, left_edge, right_edge, mid_line):
    # description:根据绝缘子串的灰度图和其左右边缘计算其绝缘子片的数量

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

def identifyCategory(img):
    # -1表示缺失，0表示变盘径，1表示单段固定盘径，2表示双段
    ret = -1

    img_gray, img_binary = preProcess(img)
    left_edge, right_edge, mid_line = detectEdge(img_binary) 
    k_m, b_m, mid_line_fitted, mse_m = fitLine(mid_line)
    num, gray_curve = calNumOfPieces(img_gray, left_edge, right_edge, mid_line) 
    if num < 9:
        return -1, -1
    
    k_l, b_l, left_edge_fitted, mse_l = fitLine(left_edge, random_seed=np.random.randint(0, 1024))
    k_r, b_r, right_edge_fitted, mse_r = fitLine(right_edge, random_seed=np.random.randint(0, 1024))
    angle_l = np.degrees(np.arctan(k_l))
    angle_r = np.degrees(np.arctan(k_r))

    # 绝缘子串分类
    thresh = 5
    diff = 0
    if angle_l * angle_r > 0:
        diff = abs(angle_l - angle_r)
    else:
        diff = 180 - abs(angle_l) - abs(angle_r)

    if diff < thresh:
        ret = 1
        print("片数:{},角度差:{},类别为固定盘径,中线偏差:{}".format(num, diff, mse_m))
    else:
        ret = 0
        print("片数:{},角度差:{},类别为变盘径,中线偏差:{}".format(num, diff, mse_m))

    # 曝光程度分类
    base = 0
    exposure_type = 0
    if ret == 0:
        base = 23
    elif ret == 1:
        base = 32
    else:
        return -1, -1

    thresh_mid = 60
    if (mse_m < thresh_mid) and (abs(base - num) < 5):
        exposure_type = 0
    elif (mse_m > thresh_mid) and (abs(base - num) < 5):
        exposure_type = 1
    elif (mse_m > thresh_mid) and (abs(base - num) > 5):
        exposure_type = -1
    else:
        exposure_type = -1

    # 打印输出
    if exposure_type == 0:
        print("正常图片");
    elif exposure_type == -1:
        print("无法修复")
    else:
        print("可以修复")

    # plot
    for i in range(len(left_edge)):
        cv2.circle(img, left_edge[i], 1, (0, 255, 0), 1)
        cv2.circle(img, left_edge_fitted[i], 1, (255, 0, 0), 1)
        cv2.circle(img, right_edge[i], 1, (0, 255, 0), 1)
        cv2.circle(img, right_edge_fitted[i], 1, (255, 0, 0), 1)
        cv2.circle(img, mid_line[i], 1, (255, 0, 0), 1)
        pass
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(img_binary, "binary")
    plt.subplot(1, 3, 3)
    plt.plot(gray_curve)
    #plt.show()

    if diff < thresh:
        return 1, exposure_type
    else:
        return 0, exposure_type

def main(opt):
    path, output, show = opt.path, opt.output, opt.show
    img_names = os.listdir(path)

    N = 0
    n = 0
    for img_name in img_names:
        if not img_name.endswith(".jpg"):
            continue
        if img_name.startswith("2"):
            continue
        N += 1
        img = cv2.imread(os.path.join(path, img_name))
        category, exposure_type = identifyCategory(img)
        if exposure_type == 1:
            n += 1
    print("准确率:{}".format(n / N))
        
if __name__ == "__main__":
    parser = ArgumentParser(description="检测绝缘子串")
    parser.add_argument("--path", type=str, default="test_data/normal", help="图片路径")
    parser.add_argument("--output", type=str, default="output", help="图片保存路径")
    parser.add_argument("--show", action='store_true', help="显示检测效果")

    main(parser.parse_args())

