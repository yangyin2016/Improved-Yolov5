import cv2
import os
import sys
import numpy as np
import random

class DataAugment:
    """
    分割绝缘子串，然后将其添加到新的背景图片中
    """
    def __init__(self):
        pass

    def split(self, img_path, label_path):
        """
        分割绝缘子串
        """
        img = cv2.imread(img_path)
        label = np.loadtxt(label_path, np.float, ndmin=2)
        height, width, channel = img.shape

        # 处理所有的目标
        result = []
        for bbox in label:
            # 计算bbox坐标
            center_x = bbox[1] * width
            center_y = bbox[2] * height
            bbox_width = bbox[3] * width 
            bbox_height = bbox[4] * height

            left_top_position = (int(center_x - bbox_width / 2), int(center_y - bbox_height / 2))
            right_bottom_position = (int(center_x + bbox_width / 2), int(center_y + bbox_height / 2))

            # 分割图片
            img_splited = img[left_top_position[1]:right_bottom_position[1] + 1, 
                    left_top_position[0]:right_bottom_position[0] + 1] 
            result.append(img_splited)
        return result
    
    def merge(self, img, background, new_left_top):
        """
        抠图，融合
        """
        #background[new_left_top[1]:new_left_top[1] + img.shape[0],
        #    new_left_top[0]:new_left_top[0] + img.shape[1]] = img

        # 转换到hsv空间，过滤背景
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([78, 43, 46])
        upper_blue = np.array([110, 255, 255])
        lower_white = np.array([0, 0, 221])
        upper_white = np.array([180, 30, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        mask.fill(255)
        mask[mask_blue == mask_white] = 0

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # 插入图片
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                if mask[row, col] == 0:
                    background[new_left_top[1] + row, new_left_top[0] + col] = img[row, col]

        return background
    
    def augment(self, file_path, background_path, output_path):
        """
        输入要输入的图片的目录，分割增强后存放在output中
        """
        np.random.seed(0)
        # 处理输入文件
        train_txt = open(file_path)
        train_files = train_txt.readlines()

        # 处理背景文件
        background_files = os.listdir(background_path)
        index = 0 

        # 处理输出位置
        output_path = os.path.join(os.getcwd(), output_path)

        # 提取目标，进行融合
        for one_file in train_files:
            one_file = one_file.replace("\n", "")

            # 解析图片与label
            if not one_file.endswith(("jpg", "png")):
                continue
            file_name = one_file.split('.')[0]
            img_name = one_file
            label_name = file_name + ".txt"

            # 分割
            objs = self.split(img_name, label_name)
            for obj in objs:
                if obj.shape[0] * obj.shape[1] < 5000:
                    continue

                # 计算新图片名称
                new_img_path = os.path.join(output_path, "{:0>4d}".format(index) + ".jpg")
                new_label_path = new_img_path.replace("jpg", "txt")

                # 读取背景图片，计算融合尺寸
                b_img = cv2.imread(os.path.join(background_path, background_files[index]))
                ratio = min(b_img.shape[0] / obj.shape[0], 
                        b_img.shape[1] / obj.shape[1]) * (np.random.random() * 0.5 + 0.5) 
                new_shape = (int(obj.shape[0] * ratio), int(obj.shape[1] * ratio))
                obj = cv2.resize(obj, (new_shape[1], new_shape[0])) 
                # 计算融合后的bbox
                range_x = b_img.shape[1] - new_shape[1]
                range_y = b_img.shape[0] - new_shape[0]

                new_left_top = (int(np.random.random() * range_x), int(np.random.random() * range_y))
                label_txt = np.zeros((1, 5), dtype=np.float)
                label_txt[0][1] = (new_left_top[0] + new_shape[1] / 2) / b_img.shape[1]
                label_txt[0][2] = (new_left_top[1] + new_shape[0] / 2) / b_img.shape[0]
                label_txt[0][3] = (new_shape[1]) / b_img.shape[1]
                label_txt[0][4] = (new_shape[0]) / b_img.shape[0]

                # 计算bbox坐标
                bbox = label_txt[0]
                height, width, _ = b_img.shape
                center_x = bbox[1] * width
                center_y = bbox[2] * height
                bbox_width = bbox[3] * width 
                bbox_height = bbox[4] * height

                left_top_position = (int(center_x - bbox_width / 2), int(center_y - bbox_height / 2))
                right_bottom_position = (int(center_x + bbox_width / 2), int(center_y + bbox_height / 2))

                # 把背景抠出来，然后融合
                b_img = self.merge(obj, b_img, left_top_position)            

                # 显示效果
                #img = cv2.rectangle(b_img, left_top_position, right_bottom_position, (0, 0, 255))
                img = b_img
                cv2.imshow("b_img", img)
                if cv2.waitKey(0) == ord('s'):
                    print("skip")
                    continue
                cv2.imwrite(new_img_path, b_img)
                np.savetxt(new_label_path, label_txt)
                index += 1
                print("saved") 

if __name__ == "__main__":
    data_augment = DataAugment()

    data_augment.augment("/home/fenghan/datasets/insulator/new_insulator/train.txt","background"  , "output")
