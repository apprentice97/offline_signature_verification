import data_pro_aug.processing
from data_pro_aug.augmentation import *
import tensorflow as tf
import time
import os
import datetime


# 用于数据预处理和数据增强
if __name__ == '__main__':

    path_train_forge = "../data/chinese_data_aug_w/forge/"
    path_train_genuine = "../data/chinese_data_aug_w/genuine/"

    paths = [path_train_forge, path_train_genuine]

    print(datetime.datetime.now())
    for path in paths:
        dirs = os.listdir(path)
        for dir_ in dirs:
            dir_path = os.path.join(path, dir_)
            dir_path_pro = '../data/' + 'chinese_data_aug_b' + dir_path[26:] + '/'
            dir_path_aug = '../data/' + 'chinese_data_aug_b' + dir_path[26:] + '/'
            print(dir_path, 'processing')
            print(dir_path_pro, 'processing')
            data_pro_aug.processing.run_dir(dir_path, dir_path_pro)
            # print(dir_path, 'augmentation')
            # data_augmentation.run_dir(dir_path_pro, dir_path_aug)
    print(datetime.datetime.now())