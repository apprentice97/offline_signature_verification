#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import cv2 as cv
import numpy as np


# In[2]:


import data_pro_aug.processing


# In[3]:


img = cv.imread(r'generate\img_gray.png')


# In[4]:


cv2.imshow('ima_gray.png', img)


# In[5]:


print(img.shape)
img = img[:,:,0]
print(img.shape)


# In[6]:


from matplotlib import pyplot as plt

def plot_hist(img):
    img_hist = np.histogram(img.ravel(),256,[0,256])
    hist = img_hist[0]
    plt.bar(np.arange(256),hist)
    plt.show()

plot_hist(img)


# In[7]:


def plt_hist2(img):
    plt.hist(img.ravel(), 256, [0, 256])
    # 参数bin256表示256个柱子，[0,256]表示范围
    plt.show()
    
plt_hist2(img)


# In[8]:


# 基于灰度直方图的灰度均值和标准差
mean, std = cv2.meanStdDev(img)
print('灰度均值：\n',mean,'\n灰度标准差:\n',std)


# In[9]:


import math
def get_entropy(img_):
    x, y = img_.shape[0:2]
    img_ = cv2.resize(img_, (100, 100)) # 缩小的目的是加快计算速度
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    img = np.array(img_)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k =  float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res
print('灰度图像的信息熵：',get_entropy(img))


# In[10]:


img2 = cv.imread(r'generate/img_binary_inv.png',1)
img2 = img2[:,:,0]
moments = cv2.moments(img2)
hu_moments = cv2.HuMoments(moments)
print('二值图像取反的几何矩：\n',hu_moments)

center=(moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
print('重心：',center)

a = moments['m20']/moments['m00'] - center[0]*center[0]
b = moments['m11']/moments['m00'] - center[0]*center[1]
c = moments['m02']/moments['m00'] - center[1]*center[1]
theta = cv2.fastAtan2(2*b,(a - c))/2
print('方向角度：',theta)

# 此部分为连通域定位并且框起来，对于汉字这种分割开的无法定位，失败
# img2 = img2.astype(np.uint8)
# retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img2)
# print((stats[1][0],stats[1][1]),(stats[1][0]+stats[1][2],stats[1][1]+stats[1][3]))
# img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
# img3 = cv2.rectangle(img2,(stats[1][0],stats[1][1]),(stats[1][0]+stats[1][2],stats[1][1]+stats[1][3]),(0,255,0),2)
# cv2.imwrite(r"generate\img_test.png", img3)


# In[ ]:




