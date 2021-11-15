# 一些基本算法的实现
import scipy.signal as signal

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


