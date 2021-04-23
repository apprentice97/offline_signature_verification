import cv2
import cv2 as cv
import numpy as np


def method_2(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary


def method_3(image):
    blurred = cv.pyrMeanShiftFiltering(image, 50, 100)
    # gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    # t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return blurred


def bi_demo(image):  # 双边滤波
    dst = cv.bilateralFilter(image, 0, 100, 15)
    cv.namedWindow("bi_demo", cv.WINDOW_NORMAL)
    cv.imshow("bi_demo", dst)


# 去除背景
def remove_background(img):
    ret = img
    height, width = ret.shape
    prospect, background = 255, 0
    height_top = 0
    height_bottom = height
    width_left = 0
    width_right = width
    # for i in range(height):
    #     for j in range(width):
    #         print(ret[i, j], end=" ")
    #     print("")

    border = 5

    for i in range(border, height - border):
        for j in range(border, width - border):
            if height_top == 0 and ret[i, j]:
                height_top = i
            if ret[i, j]:
                height_bottom = i

    for j in range(border, width - border):
        for i in range(border, height - border):
            if width_left == 0 and ret[i, j]:
                width_left = j
            if ret[i, j]:
                width_right = j

    return ret[height_top - 3:height_bottom + 3, width_left - 3:width_right + 3]


# 细化处理
def thin(image):

    array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
             1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    # array = [1 - x for x in array]
    h, w = image.shape
    iThin = image

    for i in range(h):
        for j in range(w):
            if image[i, j] == 0:
                a = [1] * 9
                for k in range(3):
                    for l in range(3):
                        # 如果3*3矩阵的点不在边界且这些值为零，也就是黑色的点
                        if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and iThin[i - 1 + k, j - 1 + l] == 0:
                            a[k * 3 + l] = 0
                sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                # print(sum)
                # print(len(array))
                # 然后根据array表，对ithin的那一点进行赋值。
                iThin[i, j] = array[sum] * 255
    return iThin


# 将图片转换成从[0,1]转换成[0,255]
def img_1_to_255(img):
    img = img.astype('uint8')
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            img[i, j] = img[i, j] * 255
    return img


# 将图片转换成从[0,255]转换成[0,1]
def img_255_to_1(img):
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img[i,j]:
                img[i, j] = 1
    return img
'''
    interpolation - 插值方法。共有5种：

    １）INTER_NEAREST - 最近邻插值法

    ２）INTER_LINEAR - 双线性插值法（默认）

    ３）INTER_AREA - 基于局部像素的重采样（resampling using pixel area relation）。对于图像抽取（image decimation）来说，这可能是一个更好的方法。但如果是放大图像时，它和最近邻法的效果类似。

    ４）INTER_CUBIC - 基于4x4像素邻域的3次插值法

    ５）INTER_LANCZOS4 - 基于8x8像素邻域的Lanczos插值
'''
# img_row = cv2.resize(img_row, (int(300), int(50)))


# 图片旋转
def rotate_bound(image, angle):
    # 获取宽高
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 提取旋转矩阵 sin cos
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像的新边界尺寸
    nW = int((h * sin) + (w * cos))
    #     nH = int((h * cos) + (w * sin))
    nH = h

    # 调整旋转矩阵
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# 获取图片旋转角度
def get_minAreaRect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    return cv2.minAreaRect(coords)


def get_correct(img):
    if img is None:
        image_path = r"data\4.png"
        image = cv2.imread(image_path)
    image = img
    angle = get_minAreaRect(image)[-1]
    if angle < -45:
        angle += 90
    rotated = rotate_bound(image, angle)

    # cv2.putText(rotated, "angle: {:.2f} ".format(angle),
    #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the output image
    # print("[INFO] angle: {:.3f}".format(angle))
    # cv2.imshow("imput", image)
    # cv2.imshow("output", rotated)
    # cv2.waitKey(0)
    return rotated
