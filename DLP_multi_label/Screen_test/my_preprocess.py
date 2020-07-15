import os
from LIBS.ImgPreprocess import my_preprocess
import multiprocessing
import cv2

from LIBS.ImgPreprocess.my_preprocess_dir import do_process_dir


dir_original = '/media/ubuntu/data2/其它数据集/外部测试集/original'
dir_preprocess = '/media/ubuntu/data2/其它数据集/外部测试集/preprocess384'

do_process_dir(dir_original, dir_preprocess, image_size=384,
               add_black_pixel_ratio=0.02)


