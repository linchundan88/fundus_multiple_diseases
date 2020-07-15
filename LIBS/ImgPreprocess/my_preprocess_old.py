'''
眼底图像预处理模块
my_fundus_image_quality() 调用shell C++ Opencv 图像质量判断  Gradable Retinal Image
my_detect_fundus_image() 用一个很简单的算法 判断是否眼底图像
process_img() 单一文件进行预处理，接收参数，调用my_preprocess或者convertImage
convertImage()  Kaggle竞赛的预处理代码
my_preprocess() 结合了HoughCircles检测圆，和Kaggle预处理方法cv2.addWeighted(image1, 4, cv2.GaussianBlur
'''


import cv2
import numpy as np
import os
import uuid
import sys
sys.path.append("..")


DEL_PADDING_RATIO = 0.02  #used for del_black_or_white
CROP_PADDING_RATIO = 0.02  #used for my_crop_xyr

# del_black_or_white margin
THRETHOLD_LOW = 7
THRETHOLD_HIGH = 180

# HoughCircles
MIN_REDIUS_RATIO = 0.33
MAX_REDIUS_RATIO = 0.6

#illegal image
IMG_SMALL_THRETHOLD = 80

def del_black_or_white(img1):
    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=-1)

    height, width = img1.shape[:2]

    (left, bottom) = (0, 0)
    (right, top) = (width, height)

    padding = int(min(width, height) * DEL_PADDING_RATIO)


    for i in range(width):
        array1 = img1[:, i, :]  #array1.shape[1]=3 RGB
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            left = i
            break
    left = max(0, left-padding)

    for i in range(width - 1, 0 - 1, -1):
        array1 = img1[:, i, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            right = i
            break
    right = min(width, right + padding)

    for i in range(height):
        array1 = img1[i, :, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            bottom = i
            break
    bottom = max(0, bottom - padding)

    for i in range(height - 1, 0 - 1, -1):
        array1 = img1[i, :, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            top = i
            break
    top = min(height, top + padding)

    img2 = img1[bottom:top, left:right, :]

    return img2

def add_black_margin(img_source, add_black_pixel_ratio = 0.05):
    if isinstance(img_source, str):
        try:
            img1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        img1 = img_source

    if img1 is None:
        raise Exception("image file error:" + img_source)

    height, width = img1.shape[:2]

    add_black_pixel = int(min(height, width) * add_black_pixel_ratio)

    img_h = np.zeros((add_black_pixel, width, 3))
    img_v = np.zeros((height + add_black_pixel*2, add_black_pixel, 3))

    img1 = np.concatenate((img_h, img1, img_h), axis=0)
    img1 = np.concatenate((img_v, img1, img_v), axis=1)

    return img1



#预处理只是切割  DR1准确率明显下降
#检测圆，如果检测不到假设圆心在中心， 然后内圆外涂黑，内圆内保留，然后裁剪
# 青光眼 crop_circle_ratio少一点，多删除边缘

def my_preprocess(img_source, crop_size, crop_circle_ratio=0.96):
    if isinstance(img_source, str):
        try:
            img1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data: 19 extraneous bytes before marker 0xc4
            return None
    else:
        img1 = img_source

    img1 = del_black_or_white(img1)

    original_width = img1.shape[1]
    original_height = img1.shape[0]
    crop_ratio = 1200/min(original_width, original_height)  #缩放成最短边长1200

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=crop_ratio, fy=crop_ratio, interpolation=cv2.INTER_AREA)
    img1 = cv2.resize(img1, None, fx=crop_ratio, fy=crop_ratio, interpolation=cv2.INTER_AREA)

    width = img1.shape[1]
    height = img1.shape[0]

    myMinWidthHeight = min(width, height)  #最短边长1600 宽和高的最小,并不是所有的图片宽>高 train/22054_left.jpeg 相反

    myMinRadius = myMinWidthHeight // 2 - 120
    # 图像本来就横向裁剪，所以要加
    myMaxRadius = myMinWidthHeight // 2 + 180

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=450, param1=120, param2=60, minRadius=myMinRadius,
                               maxRadius=myMaxRadius)

    (x, y, r) = (0, 0, 0)

    found_circle = False

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if (circles is not None) and (len(circles == 1)):
            #有些圆心位置很离谱 25.Hard exudates/chen_liang quan_05041954_clq19540405_557410.jpg
            # x width, y height
            x1, y1, r1 = circles[0]
            if x1 > (2/5 * width) and x1 < (3/5 * width) \
                    and y1 > (2/5 * height) and y1 < (3/5 * height):
                x, y, r = circles[0]
                found_circle = True


    if not found_circle:  # 检测不到圆  根据像素分布 假设圆心在图像中心
        x = img1.shape[1] // 2
        y = img1.shape[0] // 2

        temp_x = img1[int(img1.shape[0] / 2), :, :].sum(1)
        r = int((temp_x > temp_x.mean() / 10).sum() / 2)

    (image_height, image_width) = (img1.shape[0], img1.shape[1])


    #region 裁剪图像
    # 根据半径裁减  判断高是否够  防止超过边界,所以0和width
    # 第一个是高,第二个是宽  r是半径
    img_padding = 12

    image_left = max(0, x - r - img_padding)
    image_right = min(x + r + img_padding, image_width - 1)
    image_bottom = max(0, y - r - img_padding)
    image_top = min(y + r + img_padding, image_height - 1)

    if image_width >= image_height:  # 图像宽比高大
        if image_height >= 2 * (r + img_padding):
            # 图像比圆大
            img1 = img1[image_bottom: image_top, image_left:image_right]
        else:
            # 因为图像高度不够,图像被垂直剪切
            img1 = img1[:, image_left:image_right]
    else:  # 图像宽比高小
        if image_width >= 2 * (r + img_padding):
            # 图像比圆大
            img1 = img1[image_bottom: image_top, image_left:image_right]
        else:
            img1 = img1[image_bottom:image_top, :]


    #endregion
    scale = 400  #image1 shape 大概 800 足够了， 如果500,image尺寸大于1100
    s = scale * 1.0 / r
    image1 = cv2.resize(img1, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)  # fx,fy为图像x,y方向的缩放比例

    image2 = np.zeros(image1.shape)  # 生成一个空彩色图像 image2=numpy.zeros([2,3]) 创建一个数组

    cv2.circle(image2, (image1.shape[1] // 2, image1.shape[0] // 2), int(scale * crop_circle_ratio), (1, 1, 1), -1, 8, 0)

    image3 = cv2.addWeighted(image1, 4, cv2.GaussianBlur(image1, (0, 0), scale / 30), -4,
                             128) * image2 + 128 * (1 - image2)

    # image3 = add_black_margin(image3)

    image3 = cv2.resize(image3, (crop_size, crop_size))

    return image3


