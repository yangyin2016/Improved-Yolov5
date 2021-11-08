import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from argparse import ArgumentParser

def preProcess(img):
    # description:解析输入的图片路径，给出路径中可用的图像数据
    # return:[灰度图，二值图]

    # 二值化
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

def main(opt):
    path, show = opt.path, opt.show
    img_names = os.listdir(path)

    for img_name in img_names:
        img = cv2.imread(os.path.join(path, img_name))
        img_gray, img_binary = preProcess(img)
        t1 = timer()
        left_edge, right_edge = extractEdge(img_binary) 
        t2 = timer()
        print("cost timer:{}".format(t2 - t1))

        for i in range(len(left_edge)):
            cv2.circle(img, left_edge[i], 0, (255, 0, 0), 0)
            cv2.circle(img, right_edge[i], 0, (255, 0, 0), 0)

        if show:
            plt.subplot(2, 2, 1)
            plt.imshow(img)
            plt.subplot(2, 2, 2)
            plt.imshow(img_gray, cmap="gray")
            plt.subplot(2, 2, 3)
            plt.imshow(img_binary, cmap="binary")
            plt.show()

if __name__ == "__main__":
    parser = ArgumentParser(description="检测绝缘子串")
    parser.add_argument("--path", type=str, default="images", help="图片路径")
    parser.add_argument("--show", action='store_true', help="显示检测效果")

    main(parser.parse_args())

