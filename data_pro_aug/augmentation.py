from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import shutil
import cv2
import numpy as np


# 添加黑色边框
def add_black_border(img_path, scale):
    ret = cv2.imread(img_path)
    height, width = ret.shape[0], ret.shape[1]
    new_height, new_width = int(height*scale), int(width*scale)

    ans = np.zeros((new_height, new_width), dtype=np.uint8)
    ans = cv2.cvtColor(ans, cv2.COLOR_GRAY2BGR)

    h_add = int(height * (scale - 1) / 2.0)
    w_add = int(width * (scale - 1) / 2.0)

    for i in range(height):
        for j in range(width):
            ans[i + h_add, j + w_add] = ret[i][j]
    # cv2.imwrite(img_path, ans)

    ret, img = cv2.threshold(ans, 128, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(img_path, ans)
    return


def run_picture(img_dir, img_name, save_to_dir, save_prefix="", size=20):
    if save_prefix == "":
        save_prefix = img_name

    data_gen = ImageDataGenerator(
        rotation_range=20,          # 图片随机转动的角度
        width_shift_range=0.1,      # 图片随机水平偏移的幅度
        height_shift_range=0.1,     # 图片随机竖直偏移的幅度
        shear_range=0.2,            # 浮点数，剪切强度（逆时针方向的剪切变换角度）
        zoom_range=[1, 1.5],      # 缩放比例
        fill_mode='constant')

    img = load_img(os.path.join(img_dir, img_name))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    if not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir)

    i = 0
    for _ in data_gen.flow(x, batch_size=1, save_to_dir=save_to_dir, save_prefix=save_prefix, save_format='png'):
        i += 1
        if i > size:
            break


def run_dir(img_dir, save_to_dir):
    if os.path.exists(save_to_dir):
        shutil.rmtree(save_to_dir)
    os.makedirs(save_to_dir)

    for _, _, img_names in os.walk(img_dir):
        for img_name in img_names:
            # add_black_border(os.path.join(img_dir, img_name), 1.5)
            run_picture(img_dir, img_name, save_to_dir)


if __name__ == '__main__':
    run_dir('processing', 'data_augmentation')
