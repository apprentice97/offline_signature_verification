import os
import cv2
import data_pro_aug.processing_method as mt
from skimage import morphology
import shutil
import numpy as np


# 添加黑色边框
def add_black_border(img_path, scale, img):
    # ret = cv2.imread(img_path)
    ret = img
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
    return ans


def run_picture(img_dir, img_name, save_to_dir, save_name="", select=None):
    """
    传入图片的路径，返回预处理后的图片
    0. 原图片 img_row.png
    1. 倾斜校正 img_incline_correct.png
    2. 去噪声 img_remove_noise.png
    3. 转换成灰度图片 img_gray.png
    4. 二值化 img_binary.png
    5. 二值化且取反 img_binary_inv.png
    6. 去背景 img_remove_bg.png
    7. 细化字体提取骨骼 img_skeleton.png
    8. 归一化（缩放至300x150大小） img_normal.png
    9. 腐蚀
    10. 膨胀
    """

    if save_name == "":
        save_name = img_name

    if select is None:
        select = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 读入图片
    img = cv2.imread(os.path.join(img_dir, img_name))
    # cv2.imshow("img_row", img_row)
    # cv2.imwrite(r"generate\img_row.png", img)

    kernel = np.ones((2, 2), np.uint8)

    if select[9]:
        # 图像的腐蚀，默认迭代次数
        img = cv2.erode(img, kernel, iterations=3)

    if select[10]:
        # 图像的膨胀，默认迭代次数
        img = cv2.erode(img, kernel, iterations=3)

    if select[1]:
        # 倾斜校正
        img = mt.get_correct(img)
        # cv2.imshow("img_incline_correct",img_incline_correct)
        # cv2.imwrite(r"generate\img_incline_correct.png", img)

    if select[2]:
            # 去噪声
            img = mt.method_3(img)
            # cv2.imshow("img_remove_noise",img_remove_noise)
            # cv2.imwrite(r"generate\img_remove_noise.png", img)

    if select[3]:
        # 转换成灰度图片
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("img_gray", img_row)
        # cv2.imwrite(r"generate\img_gray.png", img)

    if select[4]:
        # 二值化
        ret1, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        # cv2.imshow("img_binary", img_binary)
        # cv2.imwrite(r"generate\img_binary.png", img)

    if select[5]:
        # 二值化且反转图像
        ret2, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow("img_binary_inv", img)
        # cv2.imwrite(r"generate\img_binary_inv.png", img)

    if select[6]:
        # 去背景
        img = mt.remove_background(img)
        # cv2.imshow("img_remove_bg", img)
        # cv2.imwrite(r"generate\img_remove_bg.png", img)

    if select[7]:
        # 细化处理
        img = mt.img_255_to_1(img)
        img =morphology.skeletonize(img)
        img = mt.img_1_to_255(img)
        # cv2.imshow("img_thin", img)
        # cv2.imwrite(r"generate\img_skeleton.png", img)

    if select[8]:
        # 归一化 采用双线性插值法
        img = cv2.resize(img, (300, 150))
        # cv2.imshow("img_remove_bg", img)
        # cv2.imwrite(r"generate\img_normal.png", img)



    # cv2.imwrite(os.path.join(save_to_dir, save_name), img)
    return img


def run_dir(img_dir, save_to_dir):
    if os.path.exists(save_to_dir):
        shutil.rmtree(save_to_dir)
    os.makedirs(save_to_dir)

    for _, _, img_names in os.walk(img_dir):
        for img_name in img_names:
            img = run_picture(img_dir, img_name, save_to_dir)
            # img = add_black_border(os.path.join(save_to_dir, img_name), 1.5, img_no_border)
            ret, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
            cv2.imwrite(os.path.join(save_to_dir, img_name), img)


if __name__ == '__main__':
    print("hello")
    # run_dir('data', 'processing')
    path = r"Hindi"
    paths = os.listdir(path)
    path_list = []
    for i in paths:
        if os.path.isdir(os.path.join(path, i)):
            path_list.append(os.path.join(path, i))
    for i in path_list:
        out = 'Hindi_processing' + i[5:]
        print(i, out)
        run_dir(i, out)