#%%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint

class SlotModel:
    def __init__(self, canvas, x, y, r, width, height):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.r = r
        self.w = width
        self.h = height
    
    @property
    def image(self):
        print("kontol")
        img = self.canvas * 0
        x, y, r, w, h = self.x, self.y, self.r, self.w, self.h 
        bl = x - w//2
        br = x + w//2
        bt = y - h//2
        bb = y + w//2
        cv2.circle(img, (bl + r, bt + r), r, (0, 255, 0), -1)
        cv2.circle(img, (br - r, bt + r), r, (0, 255, 0), -1)
        cv2.circle(img, (bl + r, bb - r), r, (0, 255, 0), -1)
        cv2.circle(img, (br - r, bb - r), r, (0, 255, 0), -1)
        cv2.rectangle(img, (bl + r, bt), (br - r, bb), (0, 255, 0), -1)
        cv2.rectangle(img, (bl, bt + r), (br, bb - r), (0, 255, 0), -1)
        return img

class MetaAnalyzer:
    def __init__(self, path):
        self.__path = path
        self.__image = self.load_image(self.__path)
        self.__contours = self.get_contours(self.__image)
        # self.__rectangles = self.draw_rectangles()
        # self.test_model()
        self.fit_ellipse()

    @staticmethod
    def load_image(path):
        print(f"Loading Image from {path}")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img[:2000, :]
        plt.imshow(img)
        plt.show()
        return img

    @staticmethod
    def get_contours(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res_img = img*0
        min_size = 300
        ret, im = cv2.threshold(img_gray, 70, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if c.shape[0] > min_size ]
        print(f"Using contour detection algorithm for size > {min_size}")
        for c in contours:
            cv2.drawContours(res_img, c, -1, (0,randint(100,255),randint(100,255)),2)
        plt.imshow(res_img)
        plt.show()
        return contours
    
    def test_model(self):
        img_gray = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        model = SlotModel(self.__image, 500, 500, 50, 400, 400)
        slot = model.image
        print(slot.shape)
        res_img = self.__image*0
        plt.imshow(slot)
        plt.show()

    def fit_ellipse(self):
        img = self.__image
        contours = self.__contours
        res_img = img * 0
        fit_result = []
        for cnt in contours:
            ellipse = cv2.fitEllipse(cnt)
            fit_result.append(ellipse)
            cv2.ellipse(res_img,ellipse,(0,randint(100,255),randint(100,255)),4)
        print("Fitting ellipse")
        plt.imshow(res_img)
        plt.show()            
        size_w = [e[1][0] for e in fit_result]
        size_h = [e[1][1] for e in fit_result]
        angle = [e[2] for e in fit_result]

        print("avg_w: ", sum(size_w)/len(size_w))
        print("avg_h: ", sum(size_h)/len(size_h))
        print("avg_w: ", sum(angle)/len(angle))

    def draw_rectangles(self):
        img = self.__image
        contours = self.__contours
        res_img = img * 0
        for c in contours:
            min_y = min([p[0][0] for p in c])
            max_y = max([p[0][0] for p in c])
            min_x = min([p[0][1] for p in c])
            max_x = max([p[0][1] for p in c])
            cv2.rectangle(res_img, (min_y, min_x), (max_y, max_x), (0, 255, 0), 2)
        plt.imshow(res_img)
        plt.show()            


def main():
    path = os.path.abspath("/home/stnav/jensen-lab/sample/Align_P300L135_003.tif")
    analysis = MetaAnalyzer(path)

if __name__ == '__main__':
    main()

# %%
