import cv2
import numpy as np
import os

img = cv2.imread(r'C:\Users\19093\Desktop\OSV\data\chinese_data\genuine\001\001_1.png')


kernel = np.ones((2,2), np.uint8)

# 图像的腐蚀，默认迭代次数
img = cv2.erode(img, kernel, iterations=5)

cv2.imshow("1", img)

# 图像的膨胀，默认迭代次数
img = cv2.erode(img, kernel, iterations=5)


cv2.imshow("2", img)

cv2.waitKey()


