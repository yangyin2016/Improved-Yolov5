import cv2
import numpy as np
from ransac import randomSampleConsensus
from utils import medianFilter, lowPassFilter, calNumOfMaximumValue

class ImageItem:
    # 存放寻找绝缘子串过程中图像的基本类型
    # 使用方法：
    #   it1 = ImageItem()
    #   gray_img = it1.gray_img
    def __init__(self, img):
        # 在初始化的同时计算灰度图、二值图和canny检测图
        self.color_img = img.copy()
        self.gray_img, self.binary_img, self.canny_img = self.__preProcess(self.color_img)
    
    def __preProcess(self, img, clear_no_max_area=True, min_area=100):
        # 滤波、二值化
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        binary_img = cv2.dilate(binary_img, kernel, iterations=2)
        binary_img = cv2.erode(binary_img, kernel, iterations=2)

        contours, hiterarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 如果clear_no_max_area为True，将其非最大的连通区域清零
        if clear_no_max_area:
            area = [cv2.contourArea(contours[i]) for i in range(len(contours))]
            for i in range(len(contours)):
                if i != np.argmax(area):
                    cv2.fillPoly(binary_img, [contours[i]], 0)

        # 如果clear_no_max_area为False，则清除面积小于min_area的区域
        else:
            for i in range(len(contours)):
                if cv2.contourArea(contours[i]) < min_area:
                    cv2.fillPoly(binary_img, [contours[i]], 0)

        # canny边缘检测，保存检测结果
        thresh_min = 10
        thresh_max = 50
        blur_img = cv2.GaussianBlur(img, (5, 5), 0)
        canny_img = cv2.Canny(blur_img, thresh_max, thresh_max)

        return gray_img, binary_img, canny_img


class EdgeItem:
    # 存放寻找绝缘子串过程中，其轮廓特征的数据
    def __init__(self, points, ignore_ratio=0.1):
        # 在初始化的同时拟合三个曲线
        self.detected_points = points.copy()
        self.k, self.b, self.fitted_points, self.variance = self.__fitLine(self.detected_points, ignore_ratio)

    def __fitLine(self, points, ignore_ratio):
        # 拟合线段并计算两者的方差
        # @param ignore_ratio 线段两端容易受到干扰，因此在计算方差过程中将忽略线段两端ignore_ratio比例的长度

        size = len(points)
        fitted_poinst = []
        variance = 0.0
        k, b = randomSampleConsensus(points[int(ignore_ratio * size) : int((1 - ignore_ratio) * size)])

        for i in range(size):
            x, y = int((points[i][1] - b) // k), points[i][1]
            fitted_poinst.append([x, y])
            if i > ignore_ratio * size and i < (1 - ignore_ratio) * size:
                variance += np.square((points[i][0] - x))
        variance /= ((1 - 2 * ignore_ratio) * size)

        return k, b, fitted_poinst, variance


class InsulatorItem:
    # 每个类实例表示在图片中的一个绝缘子串目标

    def __init__(self, img, bbox):
        # @param img 基于神经网络或者数据集label分割出的绝缘子串图片
        # @param bbox 格式为[category,x1,y1,x2,y2]

        self.img = ImageItem(img)
        self.bbox = bbox.copy()
        self.insulator_type = -1
        self.exposure_type = -1

        # 从self.img中提取绝缘子串的轮廓特征
        self.binary_left_edge, self.binary_right_edge, self.binary_mid_line = self.__detectEdgeByColor()
        l, r, m = self.__detectEdgeByColor()
        self.binary_left_edge, self.binary_right_edge, self.binary_mid_line = EdgeItem(l), EdgeItem(r), EdgeItem(m)

        l, r, m = self.__detectEdgeByCanny(left_limit=[self.binary_left_edge.k, self.binary_left_edge.b], 
                right_limit=[self.binary_right_edge.k, self.binary_right_edge.b], thresh=3)
        self.canny_left_edge, self.canny_right_edge, self.canny_mid_line = EdgeItem(l), EdgeItem(r), EdgeItem(m)

        # 提取灰度特征
        self.gray_curve_peak_index, self.gray_curve = self.__detectGrayCurve()

    def __detectEdgeByColor(self):
        left_edge = []
        right_edge = []
        mid_line = []
        binary_img = self.img.binary_img
        contours, hiterarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 遍历边界点，寻找左右边缘
        total_points = {}
        lower = np.inf
        upper = -np.inf
        for contour in contours:
            for point in contour:
                x, y = point[0]
                if x == 0 or y == 0:
                    continue
                lower = min(lower, y)
                upper = max(upper, y)
                if y not in total_points.keys():
                    total_points[y] = [x]
                elif len(total_points[y]) == 1:
                    total_points[y].append(x)
                    if total_points[y][0] > total_points[y][1]:
                        total_points[y][0], total_points[y][1] = total_points[y][1], total_points[y][0]
                else:
                    if x < total_points[y][0]:
                        total_points[y][0] = x
                    elif x > total_points[y][1]:
                        total_points[y][1] = x

        # 将total_points中的边缘点转换成列表的形式，同时计算中线坐标
        index = 0
        for row in range(lower, upper + 1):
            if (row not in total_points.keys()) or (len(total_points[row]) != 2):
                continue
            left_edge.append([total_points[row][0], row])
            right_edge.append([total_points[row][1], row])
            mid_line.append([(left_edge[index][0] + right_edge[index][0]) // 2, row])
            index += 1

        return left_edge, right_edge, mid_line

    def __detectEdgeByCanny(self, left_limit, right_limit, thresh=3):
        k_l, b_l = left_limit
        k_r, b_r = right_limit
        img = self.img.color_img
        img_canny = cv2.GaussianBlur(img, (5, 5), 0)
        img_canny = cv2.Canny(img_canny, 10, 50)
        left_edge = []
        right_edge = []
        mid_line = []

        h, w, _ = img.shape
        for row in range(h):
            for col1 in range(w):
                if img_canny[row, col1] == 255:
                    if (row - b_l) // k_l > col1 + thresh:
                        continue
                    left_edge.append([col1, row])
                    break
            for col2 in reversed(range(w)):
                if img_canny[row, col2] == 255:
                    if (row - b_r) // k_r < col2 - thresh:
                        continue
                    right_edge.append([col2, row])
                    break
            mid_line.append([int((col1 + col2) // 2), row]) 

        # 滤波
        left_edge = np.array(left_edge)
        left_edge[:, 0] = medianFilter(left_edge[:, 0])
        right_edge = np.array(right_edge)
        right_edge[:, 0] = medianFilter(right_edge[:, 0])
        mid_line = np.array(mid_line)
        mid_line[:, 0] = medianFilter(mid_line[:, 0])

        return left_edge.tolist(), right_edge.tolist(), mid_line.tolist()
     
    def __detectGrayCurve(self):
        # 检测绝缘子串的绝缘子片片数

        # 图像锐化
        gray_img = self.img.gray_img
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        gray_img_sharpened = cv2.filter2D(gray_img, -1, kernel)

        # 计算灰度曲线并滤波
        line = np.array(self.canny_mid_line.detected_points)
        gray_curve = gray_img[line[:, 1], line[:, 0]]   # 中线上的灰度值

        # 计算极值点数量
        gray_curve_filted = lowPassFilter(gray_curve, ratio=0.4)
        peak_index = calNumOfMaximumValue(gray_curve_filted, size=10)
        peak_index = peak_index.shape[0]

        return peak_index, gray_curve_filted

    def reflectKbToGlobal(self):
        # 将k和b从局部图片映射到全局图片
        k, b = self.binary_mid_line.k, self.binary_mid_line.b
        x1, y1 = self.bbox[1: 3]
        b = b - y1 - k * x1
        return k, b

