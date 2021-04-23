import sys
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

import cv2
import time
import itertools
import random

from sklearn.utils import shuffle

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dropout
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime


# 根据申请GPU资源
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# 数据集参数
split1 = 100    # 训练集是从1-100个人
split2 = 130    # 验证集是从100-130个人
final_threshold = 0     # 测试集是从130-150个人
path_g = "../data/chinese_sig/G/"  # 真签名的保存路径
path_f = "../data/chinese_sig/F/"  # 随机伪造签名的保存路径
path_h = "../data/chinese_sig/H/"  # 刻意伪造签名的保存路径

# 模型参数
img_h, img_w = 150, 220 # 读入图片的长和宽
epochs = 100  # 训练多少个epoch结束
batch_sz = 32  # 一个epoch的大小
learning_rate = 1e-4    # 学习率
rho = 0.9   # RMSProp 优化器的衰退因子
es_patience = 10
factor = 0.1
rl_patience = 2
min_learning_rate = 1e-10
save_weights_only = True

# 输出参数
model_save_path = '../weight/weights12_aug/'  # 模型的保存路径
if not os.path.exists(model_save_path): # 如果模型保存路径不存在，则创建改文件夹
    os.mkdir(model_save_path)


# 将控制台输出存入指定文件中
class Logger(object):
    def __init__(self, stream=sys.stdout):
        self.terminal = stream
        self.log = open(model_save_path + "default.log", 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger(stream=sys.stdout)
print("\n\n--------------------------")
print(datetime.datetime.now().strftime("%Y/%m/%d/ %H:%M"))
