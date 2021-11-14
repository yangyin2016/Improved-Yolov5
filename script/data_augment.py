import numpy as np
import cv2

class DataAugment:
    def __init__(self, is_augment=True):
        self.is_augment = is_augment

    def augment(self, img):
        new_img = img
        if not self.is_augment:
            return img

        # 1.x轴翻转
        if np.random.rand() > 0.5:
            new_img = cv2.flip(new_img, 0)  

        # 2.y轴翻转
        if np.random.rand() > 0.5:
            new_img = cv2.flip(new_img, 1)  

        # 3.长宽比例随机缩放(0.7 - 1.3)
        h, w, _ = new_img.shape
        ratio = (np.random.rand() * 0.6) + 0.7
        h = int(h * ratio)
        new_img = cv2.resize(new_img, (w, h))
        
        new_img = new_img.astype(np.uint16)

        # 4.随机对比度变化
        alpha = np.random.uniform(0.7, 1.3)
        new_img = (new_img * alpha)
        new_img = np.where(new_img>255, 255, new_img)
        new_img = np.where(new_img<0, 0, new_img)

        # 5.随机亮度变化
        #alpha = int(np.random.uniform(-10, 64))
        #new_img += alpha
        #new_img = np.where(new_img>255, 255, new_img)
        #new_img = np.where(new_img<0, 0, new_img)

        # 6.随机饱和度变换
        alpha = np.random.uniform(0.7, 1.3)
        new_img[1] = new_img[1] * alpha
        new_img = np.where(new_img>255, 255, new_img)
        new_img = np.where(new_img<0, 0, new_img)

        # 7.添加椒盐噪声
        #noise = np.random.random((new_img.shape[:2])) 
        #alpha = np.random.uniform(0.98, 1.0)
        #new_img[noise>alpha] = 255
        #alpha = np.random.uniform(0.98, 1.0)
        #new_img[noise>alpha] = 0

        return new_img.astype(np.uint8)
