'''
 Created:2019/4/6
 Modified:2019/5/7
'''

import heapq
import cv2
import numpy as np
from LIBS.ImgPreprocess.my_image_helper import image_to_square
import os

THRETHOLD_FILESIZE = 5*1024

def cut_oct_2d(img_file, image_size=None):
    if isinstance(img_file, str):
        size = os.path.getsize(img_file)
        if size < THRETHOLD_FILESIZE:
            return None

        img1 = cv2.imread(img_file)
    else:
        img1 = img_file

    # img1.shape:(885,512,3)  (height,width,3)

    height, width = img1.shape[:-1]

    if height > width: #TOPCON
        array_sum = np.sum(img1, axis=(1, 2))  # shape (885)

        top_n = heapq.nlargest(1, range(len(array_sum)), array_sum.take)

        bottom = max(0, top_n[0] - (img1.shape[1] // 2))
        top = min(img1.shape[0], top_n[0] + (img1.shape[1] // 2))

        img_crop = img1[bottom:top, :, :]

        if img_crop.shape[0] < img_crop.shape[1]:  # width < height
            add_black_pixel = img_crop.shape[1] - img_crop.shape[0]
            img_height_0 = np.zeros((add_black_pixel, img_crop.shape[1], 3))

            img_crop = np.concatenate((img_height_0, img_crop), axis=0)
        elif img_crop.shape[0] > img_crop.shape[1]:
            add_black_pixel = img_crop.shape[0] - img_crop.shape[1]
            img_width_0 = np.zeros((img_crop.shape[0], 3), add_black_pixel)
            img_crop = np.concatenate((img_width_0, img_crop), axis=1)

        if image_size is not None:
            img_crop = cv2.resize(img_crop, (image_size, image_size))

        return img_crop

    else:   #Carl Zeiss Jena
        return image_to_square(img1, image_size)


if __name__ ==  '__main__':
    img_file1 = '/media/ubuntu/data1/jst/OCT/全层黄斑裂孔/全层黄斑裂孔 TOPCON/CGC19530117_20161114_103724_3DOCT00_L_062.jpg'
    img1 = cut_oct_2d(img_file=img_file1, image_size=384)
    cv2.imwrite('1.jpg', img1)

    img_file2 = '/media/ubuntu/data1/jst/OCT/全层黄斑裂孔/全层黄斑裂孔 蔡司/chen_gui hua__CGH19511228_19511228_Female_HD 5 Line Raster_20161212145600_OD_20190415172714.bmp'
    img2 = cut_oct_2d(img_file=img_file2, image_size=384)
    cv2.imwrite('2.jpg', img2)

    print('OK')

