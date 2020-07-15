'''

'''

import os
from LIBS.ImgPreprocess import my_preprocess_old

import cv2


def do_process_dir(source_dir, dest_dir, image_size=299):
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

            process_img(img_file_source, img_file_dest, image_size)

    delete_small_files(dest_dir)

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
def process_img(img_source, img_file_dest, crop_size, SMALL_SIZE=2048):
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
        image1 = my_preprocess_old.my_preprocess(img, crop_size=crop_size)
        if image1 is not None:
            cv2.imwrite(img_file_dest, image1)
            print(img_file_dest)
        else:
            print('error:', img_source)  # file error

    else:  # file not exists or other errors
        print('error:', img_source)



if __name__ == '__main__':

    dir_original = '/media/ubuntu/data1/测试集0/original'
    dir_preprocess = '/media/ubuntu/data1/测试集0/preprocess512_old'

    dir_original = '/media/ubuntu/data1/DR1_2019_5_5/original'
    dir_preprocess = '/media/ubuntu/data1/DR1_2019_5_5/preprocess512_old'


    do_process_dir(dir_original, dir_preprocess, image_size=512)


