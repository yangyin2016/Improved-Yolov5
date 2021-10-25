import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

class ResumeImage:
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.mean = np.array(mean)
        self.std= np.array(std)
        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(640),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ])

    def random_bright(self, img, u=100):
        """
        随机亮度
        """
        if np.random.random() > 0:
            alpha = np.random.uniform(-u, u) / 255
            img += alpha
            img = img.clamp(min=-1.0, max=1.0)
        return img
    def random_contrast(self, img, lower=0.5, upper=1.5):
        """
        随机对比度
        """
        if np.random.random() > 0:
            alpha = np.random.uniform(lower, upper)
            result = img * alpha
            result= result.clamp(min=-1.0, max=1.0)
        return result
    def random_saturation(self, img, lower=0.5, upper=1.5):
        if np.random.random() > 0:
            alpha = np.random.uniform(lower, upper)
            img[1] = img[1] * alpha;
            img[1] = img[1].clamp(min=-1.0, max=1.0)
        return img

    def augment(self, img, random_bright=True):
        """
        增强
        """
        img = self.trans(img)
        #img = self.random_contrast(img, lower=0.1, upper=1.5)
        img = self.random_saturation(img)
        return img

    def toImages(self, img):
        """
        转换成图片格式
        """
        img = img.numpy().transpose((1, 2, 0))
        img = self.std * img + self.mean
        return np.array(img * 255, dtype=np.uint8)

if __name__ == "__main__":
    data = ResumeImage()
    img = cv2.imread("001.jpg")

    result = data.augment(img)
    result = data.toImages(result)

    cv2.imshow("img", result)
    cv2.imwrite("result.jpg", result)
    if cv2.waitKey(0) == 'q':
        exit(0)
