# 本文将实现了模拟过曝行为的函数

import cv2
import os
import numpy as np
from timeit import default_timer as timer
from argparse import ArgumentParser

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
    if center_x < 0 or center_x >= img.shape[1] or center_y < 0 or center_y >= img.shape[0]:
        return img

    # 如果半径过大，则对其进行裁剪
    exposure_kernel = createExposureKernel(radius, strength)
    x1, x2 = center_x -radius, center_x + radius + 1
    y1, y2 = center_y - radius, center_y + radius + 1
    if x1 < 0:
        exposure_kernel = np.delete(exposure_kernel, range(0, -x1), axis=1)
        x1 = 0
    if y1 < 0:
        exposure_kernel = np.delete(exposure_kernel, range(0, -y1), axis=0)
        y1 = 0
    if x2 > img.shape[1]:
        exposure_kernel = np.delete(exposure_kernel, range(img.shape[1] - x1, exposure_kernel.shape[1]), axis=1)
        x2 = img.shape[1]
    if y2 > img.shape[0]:
        exposure_kernel = np.delete(exposure_kernel, range(img.shape[0] - y1, exposure_kernel.shape[0]), axis=0)
        y2 = img.shape[0]
    
    # 添加曝光
    img = img.astype(np.uint16)
    img[y1 : y2, x1 : x2] += exposure_kernel
    img[img>255] = 255

    return img.astype(np.uint8)

def main(opt):
    input_path, output_path, num = opt.input_path, opt.output_path, opt.num

    np.random.seed(0)
    data_name = os.listdir(input_path) 
    n = 0
    for name in data_name:
        if not name.endswith((".jpg", ".png")):
            continue
        path = os.path.join(input_path, name)
        img = cv2.imread(path)
        height, width, _ = img.shape 

        for i in range(0, num):
            img_exposured = img.copy()

            # 随机位置、随机半径、随机强度
            center_x, center_y = np.random.randint(0, width), np.random.randint(0, height)
            radius = int(np.random.randint(min(height, width) // 2, min(height, width) * 1.3))
            strength = int(np.random.randint(255, 400))
            img_exposured = addExposure(img_exposured, (center_x, center_y), radius, strength)

            # 保存
            prefix, suffix = name.split('.')
            output_name = os.path.join(output_path, prefix + "_{}.".format(i) + suffix)
            cv2.imwrite(output_name, img_exposured)
            print("输入位置:{},曝光位置:{},曝光半径:{},曝光强度:{},输出位置:{}".format(
                input_path, (center_x, center_y), radius, strength, output_name))
            n += 1
    print("一共处理{}张图片".format(n))

if __name__ == "__main__":
    parser = ArgumentParser("给图片添加曝光效果")
    parser.add_argument("--input_path", type=str, default="./test_data", help="原始图片路径")
    parser.add_argument("--output_path", type=str, default="output", help="输出图片的路径")
    parser.add_argument("--num", type=int, default=1, help="对每张图片的过曝次数")
    main(parser.parse_args())
