import numpy as np
import cv2

class DataAugment:
    def __init__(self, is_augment=True):
        self.is_augment = is_augment

    def randomFlip(self, img, axis_x=True, axis_y=True):
        # description:反转
        # 1.x轴翻转
        if axis_x and (np.random.rand() > 0.5):
            img = cv2.flip(img, 0)  

        # 2.y轴翻转
        if axis_y and (np.random.rand() > 0.5):
            img = cv2.flip(img, 1)  

        return img
    
    def randomResize(self, img, lower=0.7, upper=1.3):
        # 随机修改长宽比例
        h, w, _ = img.shape
        ratio = (np.random.rand() * (upper - lower)) + lower
        h = int(h * ratio)
        img = cv2.resize(img, (w, h))

        return img

    def randomBrightness(self, img, lower=0, upper=30, is_splited=True):
        # 随机亮度
        value = np.random.randint(lower, upper)

        img = img.astype(np.uint16)
        if is_splited:
            img[(img != 0).any(axis=2)] += value
        else:
            img += value
        img[img > 255] = 255

        return img.astype(np.uint8)

    def randomContrast(self, img, lower=0.7, upper=1.3):
        # 随机对比度
        ratio = (np.random.rand() * (upper - lower)) + lower

        img = img.astype(np.uint16)
        img = img * ratio
        img[img > 255] = 255

        return img.astype(np.uint8)

    def randomSaturation(self, img, lower=0.7, upper=1.3):
        # 随机饱和度
        ratio = (np.random.rand() * (upper - lower)) + lower

        img = img.astype(np.uint16)
        img[1] = img[1] * ratio
        img[img > 255] = 255

        return img.astype(np.uint8)

    def randomSpNoise(self, img, lower=0.02, upper=0.98):
        # 随机添加椒盐噪声
        ratio = np.random.rand()
        lower = ratio * lower
        upper = upper + ratio * (1 - upper) 

        value = np.random.random((img.shape[:2]))
        new_img = img.copy()
        new_img[value<lower] = 0
        new_img[value>upper] =255
        new_img[(img == 0).all(axis=2)] = 0

        return new_img

    def randomGaussNoise(self, img, mean=0, var=0.001):
        # 随机添加高斯噪声
        ratio = np.random.rand()
        var = ratio * var

        new_img = img.copy()
        new_img = new_img.astype(np.uint16)
        noise = np.expand_dims(np.random.normal(mean, var**0.5, img.shape[:2]), axis=2)
        new_img += (255 * noise).astype(np.uint16)

        new_img[new_img<0] = 0
        new_img[new_img>255] = 255
        new_img[(img == 0).all(axis=2)] = 0

        return new_img.astype(np.uint8)

    def whiteBalance(self, img):
        # 白平衡
        b, g, r = cv2.split(img)
        b_avg, g_avg, r_avg = np.mean(np.mean(img, axis=0), axis=0)
        k = (b_avg + g_avg + r_avg) / 3

        r = cv2.addWeighted(r, k / r_avg, 0, 0, 0)
        g = cv2.addWeighted(g, k / g_avg, 0, 0, 0)
        b = cv2.addWeighted(b, k / b_avg, 0, 0, 0)

        return cv2.merge([b, g, r])

    def augment(self, img):
        new_img = img
        if not self.is_augment:
            return img

        img = self.randomFlip(img, axis_x=True, axis_y=True)
        img = self.randomResize(img, lower=0.65, upper=1.35)
        img = self.randomBrightness(img, lower=0, upper=50, is_splited=True)
        img = self.randomContrast(img, lower=0.7, upper=1.3)
        img = self.randomSaturation(img, lower=0.7, upper=1.3)
        img = self.randomSpNoise(img, lower=0.025, upper=0.975)
        img = self.randomGaussNoise(img, mean=0, var=0.0025)
        if np.random.rand() > 0.5:
            img = self.whiteBalance(img)

        return img
