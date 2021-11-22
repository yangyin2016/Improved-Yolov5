"""
代码引用自https://github.com/falcondai/py-ransac.git
"""
import numpy as np
import random

def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, 
            max_iterations, stop_at_goal=True, random_seed=None, show=False):
    """
    description:对数据进行ransac变换
    param:
        data：要拟合的数据
        estimate:
        is_inlier:函数，判断是否是inline点
        sample_size:每次随机采样的数目
        goal_inliers:目标直线的最小点数
        max_iterations:最大迭代次数
        stop_at_goal:寻找到符合目标的直线时立即返回
    return:
        ?,cos(x), sin(x)
    """

    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        if show:
            img = np.zeros((500, 500, 3), np.uint8)
            img.fill(255)
            inliers = [] 
            outliers = []

        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                if show:
                    inliers.append(data[j])
                ic += 1
            else:
                if show:
                    outliers.append(data[j])

        if show:
            print("iterations:{}, model:{}".format(i + 1, ic))
            k = - m[0] / (m[1] + 1e-9)
            b = - m[2] / (m[1] + 1e-9)
            for point in data:
                x, y = int((point[1] - b) // k), point[1]
                cv2.circle(img, (x, y), 1, (255, 0, 0), 1)
            for point in outliers:
                cv2.circle(img, point, 1, (0, 0, 255), 1)
            for point in inliers:
                cv2.circle(img, point, 1, (0, 255, 0), 1)

            plt.figure(figsize=(15, 7))
            plt.imshow(img)
            plt.show()

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    if i == max_iterations:
        print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic

def augment(xys):
    axy = np.ones((len(xys), 3))
    axy[:, :2] = xys
    return axy

def estimate(xys):
    axy = augment(xys[:2])
    return np.linalg.svd(axy)[-1][-1, :]

def is_inlier(coeffs, xy, threshold):
    """
    判断是否是inline点
    """
    #return np.abs(coeffs.dot(augment([xy]).T)) < threshold
    k = - coeffs[0] / (coeffs[1] + 1e-9)
    b = - coeffs[2] / (coeffs[1] + 1e-9)
    return (abs(k * xy[0] - xy[1] + b) / np.sqrt(np.square(k) + 1)) < threshold

def randomSampleConsensus(points):
    """
    description:拟合points中的直线点，寻找出最接近的直线参数
    param:
        points:离散点的坐标，格式为[[x1, y1], [x2, y2],...]
    return:
        直线的参数，格式为k,b
    """
    sample_size = 2
    threshold = 2
    goal_inliers = int(0.4 * len(points))
    max_iterations = 30

    m, _ = run_ransac(points, estimate, lambda x, y: is_inlier(x, y, threshold), sample_size=sample_size, 
            goal_inliers=goal_inliers, max_iterations=max_iterations, stop_at_goal=True, random_seed=None)

    k = - m[0] / (m[1] + 1e-9)
    b = - m[2] / (m[1] + 1e-9)

    return k, b

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    import cv2
    import sys
    pass
