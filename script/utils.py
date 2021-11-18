# 一些基本算法的实现
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

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

def medianFilter(data, step=3):
    # description:中值滤波 
    return signal.medfilt(data, step) 

def derivative(data):
    # description:一个基本的求导功能实现
    if len(data) == 0:
        return data
    data_ret = []
    for i in range(1, len(data) - 1):
        data_ret.append((int(data[i + 1]) - int(data[i - 1])) / 2)
    return data_ret

def calNumOfMaximumValue(data, size=10):
    # 计算极大值点的数量
    peak_indexes = signal.argrelextrema(data, np.greater, order=size)  
    return peak_indexes[0]

def uShow(data):
    # 显示图片
    n = len(data)
    plt.figure(figsize=(15, 7))         

    for i in range(0, n):
        plt.subplot(1, n, i + 1)
        if type(data[i]) == np.ndarray:
            plt.imshow(data[i])
        elif type(data[i]) == list:
            plt.plot(data[i])
    plt.imshow()
