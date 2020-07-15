'''
将一个目录所有图像文件进行预处理，保存到另一个目录，多进程调用my_preprocess的process_img
OpenCV multi_process sometime encounter errors, so...
'''

import os
from LIBS.ImgPreprocess import my_preprocess
import multiprocessing
import cv2
import numpy as np
from LIBS.ImgPreprocess.my_image_helper import get_green_channel

def do_process_dir(source_dir, dest_dir, image_size=299,
                   convert_jpg=False, add_black_pixel_ratio=0.07):

    print('preprocess start')

    if source_dir.endswith('/'):
        source_dir = source_dir[:-1]
    if dest_dir.endswith('/'):
        dest_dir = dest_dir[:-1]

    for dir_path, subpaths, files in os.walk(source_dir, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)

            filename, file_extension = os.path.splitext(img_file_source)

            if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                print('file ext name:', f)
                continue

            img_file_dest = img_file_source.replace(source_dir, dest_dir)
            if convert_jpg:
                fname, fename = os.path.splitext(img_file_dest)
                img_file_dest = fname + '.jpg'

            if os.path.exists(img_file_dest):
                continue

            if not os.path.exists(os.path.dirname(img_file_dest)):
                os.makedirs(os.path.dirname(img_file_dest))

            process_img(img_file_source, img_file_dest, image_size, add_black_pixel_ratio=add_black_pixel_ratio)

    delete_small_files(dest_dir)

    print('preprocess end')


def do_process_dir_G_channel(source_dir, dest_dir, image_size=299):
    print('preprocess start')

    # 去掉最后一个字符
    if source_dir.endswith('/'):
        source_dir = source_dir[:-1]
    if dest_dir.endswith('/'):
        dest_dir = dest_dir[:-1]

    for dir_path, subpaths, files in os.walk(source_dir, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)

            filename, file_extension = os.path.splitext(img_file_source)
            if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                print('file ext name:', f)
                continue

            # 如果目的文件已经存在
            img_file_dest = img_file_source.replace(source_dir, dest_dir)
            if os.path.exists(img_file_dest):
                continue
            if not os.path.exists(os.path.dirname(img_file_dest)):
                os.makedirs(os.path.dirname(img_file_dest))

            get_green_channel(img_file_source, img_file_dest)
            print(img_file_dest)

    print('preprocess end')


def do_resize_dir(source_dir, dest_dir, image_size=299, convert_jpg=False, add_black_pixel_ratio=0):
    print('preprocess start')

    # 去掉最后一个字符
    if source_dir.endswith('/'):
        source_dir = source_dir[:-1]
    if dest_dir.endswith('/'):
        dest_dir = dest_dir[:-1]

    for dir_path, subpaths, files in os.walk(source_dir, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)

            filename, file_extension = os.path.splitext(img_file_source)

            if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                print('file ext name:', f)
                continue

            img_file_dest = img_file_source.replace(source_dir, dest_dir)
            if convert_jpg:
                fname, fename = os.path.splitext(img_file_dest)
                img_file_dest = fname + '.jpg'

            if os.path.exists(img_file_dest):
                continue

            if not os.path.exists(os.path.dirname(img_file_dest)):
                os.makedirs(os.path.dirname(img_file_dest))

            img1 = cv2.imread(img_file_source)
            if add_black_pixel_ratio>0 :
                img1 = my_preprocess.add_black_margin(img1, add_black_pixel_ratio)
            img1 = cv2.resize(img1, (image_size, image_size))

            print(img_file_dest)
            cv2.imwrite(img_file_dest, img1)

    print('preprocess end')

#删除无效文件，预处理后个别文件错误
def delete_small_files(dir, SMALL_SIZE_THRETHOLD = 2048):

    for dir_path, subpaths, files in os.walk(dir, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)

            filename, file_extension = os.path.splitext(img_file_source)

            if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                print('file ext name:', f)
                continue

            size = os.path.getsize(img_file_source)
            if size < SMALL_SIZE_THRETHOLD:
                print('small file:', img_file_source)
                os.remove(img_file_source)


# want use multi process, however opencv do not support multi process very well.
def process_img(img_source, img_file_dest, crop_size, add_black_pixel_ratio, SMALL_SIZE=2048):
    try:
        size = os.path.getsize(img_source)
        if size < SMALL_SIZE:
            print('error:', img_source)
            return None

        img = cv2.imread(img_source)
    except:
        # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
        print('error:', img_source)
        return None

    if img is not None:
        image1 = my_preprocess.do_preprocess(img, crop_size=crop_size, add_black_pixel_ratio=add_black_pixel_ratio)
        if image1 is not None:
            cv2.imwrite(img_file_dest, image1)
            print(img_file_dest)
        else:
            print('error:', img_source)  # file error

    else:  # file not exists or other errors
        print('error:', img_source)


'''

#opencv do not support mylti_precesses well
def do_process_dir_multi_processes(source_dir, dest_dir, list_imagesize=[299]):
    print('preprocess start')
    # 去掉最后一个字符
    if source_dir.endswith('/'):
        source_dir = source_dir[:-1]
    if dest_dir.endswith('/'):
        dest_dir = dest_dir[:-1]

    pool = multiprocessing.Pool(processes=5)

    # for i_image_size in [299]:
    for i_image_size in list_imagesize:
        # 由于有多级目录
        for dir_path, subpaths, files in os.walk(source_dir, False):
            dir_path_pre = dir_path.replace(source_dir, dest_dir)

            for f in files:
                img_file_source = os.path.join(dir_path, f)
                img_file_dest = os.path.join(dir_path_pre, f)
                # 如果目的文件已经存在
                if os.path.exists(img_file_dest):
                    continue

                filename, file_extension = os.path.splitext(img_file_source)

                if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                    print('file ext name:', f)
                    continue

                dirname, _ = os.path.split(img_file_dest)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                pool.apply_async(my_preprocess.process_img, (img_file_source, img_file_dest, i_image_size))

                print(img_file_dest)

    pool.close()
    pool.join()

    print('preprocess end')

'''


if __name__ == '__main__':

    dir_original = '/media/ubuntu/data1/外部测试集_已标注_DR1/original'
    dir_preprocess = '/media/ubuntu/data1/外部测试集_已标注_DR1/preprocess512'

    dir_original = '/media/ubuntu/data1/测试集0/original'
    dir_preprocess = '/media/ubuntu/data1/测试集0/preprocess512'

    do_process_dir(dir_original, dir_preprocess, image_size=512)

    exit(0)

    dir_original = '/media/ubuntu/data1/新生血管/preprocess384'
    dir_preprocess = '/media/ubuntu/data1/新生血管/preprocess384_G_channel'

    # dir_original = '/media/ubuntu/data1/子类/1/测试集_已标注/original'
    # dir_original = '/media/ubuntu/data1/子类/1/测试集_已标注/preprocess384'
    # dir_preprocess = '/media/ubuntu/data1/子类/1/测试集_已标注/preprocess384_G_channel'


    do_process_dir_G_channel(dir_original, dir_preprocess)

    # source_dir = '/home/jsiec/disk1/PACS/公开数据集/IDRID/test/original/'

    # source_dir = '/home/jsiec/disk1/PACS/DR-粗标/original/粗标-谢/json/yes/2018_05_10/LXQPH01-S262'

    # dest_dir = '/home/jsiec/disk1/PACS/公开数据集/IDRID/test/process'
    # do_process_dir(source_dir, dest_dir, list_imagesize=[512])

    # exit(0)

    '''
    source_dir = '/home/jsiec/disk2/pic_new_2018_08_24/=Fundus-All-add/'
    dest_dir = '/home/jsiec/disk2/pic_new_2018_08_24/=Fundus-All-add_preprocess384/'
    do_process_dir(source_dir, dest_dir, list_imagesize=[384])

    dest_dir = '/home/jsiec/disk2/pic_new_2018_08_24/=Fundus-All-add_preprocess299/'
    do_process_dir(source_dir, dest_dir, list_imagesize=[299])

    exit(0)
    source_dir = '/home/jsiec/disk1/PACS/DR-粗标/original/'

    # source_dir = '/home/jsiec/disk1/PACS/DR-粗标/original/粗标-谢/json/yes/2018_05_10/LXQPH01-S262'

    dest_dir = '/home/jsiec/disk1/PACS/DR-粗标/preprocess512/'
    do_process_dir(source_dir, dest_dir, list_imagesize=[512])


    dest_dir = '/home/jsiec/disk1/PACS/DR-粗标/preprocess299'
    do_process_dir(source_dir, dest_dir, list_imagesize=[299])

    '''
