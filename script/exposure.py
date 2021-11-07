# 本文将实现了模拟过曝行为的函数

import cv2
import numpy as np
from timeit import default_timer as timer

def createExposureKernel(radius=50, strength=100, expand_dim=True):
    # description:创造一个曝光核，可以直接添加在图像上模仿曝光效果
    # param:
    #   radius:曝光半径
    #   strength:曝光强度
    #   expand_dim:是否扩充维度
    # return:大小为(2 * radius + 1, 2 * radius + 1)的矩阵

    # 1.生成二维坐标图
    X = np.linspace(-radius, radius, 2 * radius + 1)
    Y = np.linspace(-radius, radius, 2 * radius + 1)
    xv, yv = np.meshgrid(X, Y)

    # 2. 根据过曝模拟公式生成尺度
    ratio = 1 - np.sqrt(xv**2 + yv**2) / radius
    ratio = np.where(ratio<0, 0, ratio)  

    # 3.生成kernel并返回
    kernel = strength * ratio
    if expand_dim:
        kernel = np.expand_dims(kernel, axis=2)
    return kernel.astype(np.uint16)

def addExposure(img, center, radius, strength):
    # description:在图像的指定位置添加指定半径和强度的过曝
    # return:返回过曝后的图片

    # 检测有效输入
    center_x, center_y = center
    if (center_x < radius) or (center_y < radius) or \
            (center_x + radius >= img.shape[1]) or (center_y + radius >= img.shape[0]):
        return img

    # 添加曝光
    exposure_kernel = createExposureKernel(radius, strength)
    img = img.astype(np.uint16)
    img[center_y - radius : center_y + radius + 1, center_x - radius : center_x + radius + 1] += exposure_kernel 
    img = np.where(img>255, 255, img)

    return img.astype(np.uint8)

if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    h, w, _ = img.shape
    img = addExposure(img, (300, 300), 100, 400)
    cv2.imshow("img", img)
    cv2.imwrite("output.jpg", img)
    cv2.waitKey(0)
