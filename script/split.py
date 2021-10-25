"""
本文件实现了从bbox中提取出绝缘子串的精确轮廓，并针对光照丢失效果做一定的补偿
"""
import cv2
import sys
import numpy as np
from timeit import default_timer as timer

"""
绝缘子串分为两种，一种是变盘径式，一种是固定盘径式
"""

class InsulatorSplit:
    def __init__(self):
        pass
    
    def preProcess(self, img):
        """
        description:对图像进行预处理操作
        param：
            img：输入的numpy类型的图像
        return:
            检测到的绝缘子串的二值图像
        """
        # 灰度化后要做均衡化处理,并且降低图像亮度，便于二值化
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        frame = cv2.equalizeHist(frame)
        frame = np.uint8(np.clip((cv2.add(1*frame,-50)), 0, 255))

        # 二值化
        thresh, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_OTSU)
        frame = 255 - frame

        # 形态学操作,先用开运算滤波，再开运算过滤掉塔杆等影响，再闭运算填充绝缘子串的内部噪声点
        kernel = np.ones((3, 3), np.uint8)
        frame = cv2.dilate(frame, kernel, iterations = 2)
        frame = cv2.erode(frame, kernel, iterations = 2)

        frame = cv2.erode(frame, kernel, iterations = 9)
        frame = cv2.dilate(frame, kernel, iterations = 9)

        frame = cv2.dilate(frame, kernel, iterations = 9)
        frame = cv2.erode(frame, kernel, iterations = 9)

        # 寻找最大连通区域并标记
        contours, hiterarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        area = []
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))
            #img = cv2.drawContours(img, contours, i, (0, 0, 255))

        max_idx = -1 if len(area) == 0 else np.argmax(area) 
        for i in range(len(contours)):
            if i != max_idx:
                cv2.fillPoly(frame, [contours[i]], 0)

        return frame

    def findEdge(self, img):
        """
        description:寻找绝缘子串的左右轮廓
        param：
            img：绝缘子串的轮廓二值图像
        return：
            left_edge:绝缘子串左轮廓
            right_edge:绝缘子串右轮廓
            mid_line:绝缘子串的中线
        """
        # 寻找绝缘子串的左右轮廓
        original_left_edge = []
        original_right_edge = []
        original_mid_line = []

        temp_img = np.zeros(img.shape, np.uint8)
        for row in range(img.shape[0]):
            # 寻找左轮廓
            for col1 in range(img.shape[1]):
                if img[row][col1] == 255:
                    original_left_edge.append([col1, row])
                    break

            # 寻找右轮廓
            for col2 in range(img.shape[1] - 1, -1, -1):
                if img[row][col2] == 255:
                    original_right_edge.append([col2, row])
                    break
            original_mid_line.append([(col1 + col2) // 2, row])
            cv2.circle(temp_img, ((col1 + col2) // 2, row), 0, 255, 1)

        test = cv2.dilate(temp_img, np.ones((5, 5), np.uint8), iterations=1)

        # 使用霍夫变换来检测直线
        lines_detected = cv2.HoughLinesP(temp_img, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
        normal_points = []  # 检测到的正常的中线区域
        defect_points = []  # 没有检测到的区域视为缺陷区域
        for line in lines_detected:
            x1, y1, x2, y2 = line[0]
            if x1 != x2:
                k = (y2 - y1) / (x2 - x1)
                for x in range(x1, x2 + 1):
                    normal_points.append([x, int(y1 + k * (x - x1))]) 
            else:
                for x in range(x1, x2 + 1):
                    normal_points.append([x, int(y1 + k * (x - x1))]) 

        # 最小二乘拟合
        line_param = cv2.fitLine(np.array(normal_points), cv2.DIST_L2, 0, 0.01, 0.01)
        k = (line_param[1] / line_param[0])[0]
        b = (line_param[3] - k * line_param[2])[0]

        # 计算坐标
        final_left_edge = original_left_edge
        final_mid_lines = []
        final_right_edge = []
        
        for row in range(original_mid_line[0][1], original_mid_line[len(original_mid_line) - 1][1]):
            final_mid_lines.append([int((row - b) // k), row])

        for i in range(len(final_left_edge)):
            x, y = final_left_edge[i] 
            if i >= len(final_mid_lines):
                break
            new_x ,new_y= 2 * final_mid_lines[i][0] - final_left_edge[i][0], i
            if new_x >= img.shape[1]:
                new_x = img.shape[1]
            final_right_edge.append([new_x, new_y])
        return original_left_edge, original_right_edge,original_mid_line 

    def split(self, img):
        """
        description:对绝缘子串详细轮廓进行分割
        param：
            img：输入的numpy类型的图像
        return:
        """
        original = cv2.imread("images/001.png")
        t1 = timer()
        binary_frame = self.preProcess(img)
        left_edge, right_edge, lines = self.findEdge(binary_frame)
        t2 = timer()
        print("cost time:{}ms".format((t2 - t1) * 1000))

        # 绘制边缘
        for i in range(len(left_edge)):
            cv2.circle(img, left_edge[i], 1, (0, 0, 255), 1)
            cv2.circle(original, left_edge[i], 1, (0, 0, 255), 1)
            if i < len(right_edge):
                cv2.circle(img, right_edge[i], 1, (0, 0, 255), 1)
                cv2.circle(original, right_edge[i], 1, (0, 0, 255), 1)
        for line in lines:
            cv2.circle(img, line, 0, (0, 0, 255), 0)
            cv2.circle(original, line, 0, (0, 0, 255), 0)

        # 绘制
        """points = [[[140, 615], [143, 677]], [[130, 425], [136, 546]], [[127, 350], [130, 419]], [[149, 753], [149, 688]]]
        for point in points:
            cv2.line(img, point[0], point[1], (0, 0, 255), thickness=1)
            cv2.line(original, point[0], point[1], (0, 0, 255), thickness=1)
         """
       
        result = binary_frame
        cv2.imshow("img", img)
        cv2.imshow("original", original)
        cv2.imwrite("result.png", img)
        cv2.imwrite("original.png", original)
        if cv2.waitKey(0) == ord('q'):
            exit(0)
        
        return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: {} [images]".format(sys.argv[0]))
        exit(-1)

    spliter = InsulatorSplit() 
    img = cv2.imread(sys.argv[1])
    result = spliter.split(img)

