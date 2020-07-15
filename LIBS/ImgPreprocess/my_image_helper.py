
import numpy as np
import math
import cv2
import os

# detect vessel G channel
from LIBS.ImgPreprocess.my_image_norm import input_norm


def get_green_channel(img1, img_file_dest=None):
    if isinstance(img1, str):
        img1 = cv2.imread(img1)

    img1 = img1[:, :, 1] # BGR
    img1 = np.expand_dims(img1, axis=-1)

    img_zero = np.zeros(img1.shape)
    img2 = np.concatenate((img_zero, img1, img_zero), axis=-1)

    if img_file_dest is not None:
        cv2.imwrite(img_file_dest, img2)
    else:
        return img2


# crop fovea from 512 image, train model is for 299*299
def my_crop_fovea(img1, center_x, center_y):
    center_x = center_x * (img1.shape[1] / 299)
    center_y = center_y * (img1.shape[0] / 299)

    left = center_x - (img1.shape[1] * 1/5)
    left = int(max(round(left), 0))
    right = center_x + (img1.shape[1] * 1 / 5)
    right = int(min(round(right), img1.shape[1]))
    bottom = center_y - (img1.shape[0] * 1 / 5)
    bottom = int(max(round(bottom), 0))
    top = center_y + (img1.shape[0] * 1 / 5)
    top = int(min(round(top), img1.shape[1]))

    img2 = img1[bottom:top, left:right]

    img2 = image_to_square(img2)

    return img2


def resize_images_dir(source_dir='', dest_dir='', imgsize=299):
    if not source_dir.endswith('/'):
        source_dir += '/'
    if not dest_dir.endswith('/'):
        dest_dir += '/'

    for dir_path, subpaths, files in os.walk(source_dir, False):
        for f in files:
            image_file_source = os.path.join(dir_path, f)

            file_base, file_ext = os.path.splitext(image_file_source)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue

            img1 = cv2.imread(image_file_source)
            if img1 is None:
                print('error file:', image_file_source)

            img1 = cv2.resize(img1, (imgsize, imgsize))

            image_file_dest = image_file_source.replace(source_dir, dest_dir)
            if not os.path.exists(os.path.dirname(image_file_dest)):
                os.makedirs(os.path.dirname(image_file_dest))

            cv2.imwrite(image_file_dest, img1)

            print(image_file_source)

# 1.square, 2.resize
def image_to_square(image1, image_size=None, grayscale=False):
    if isinstance(image1, str):
        image1 = cv2.imread(image1)

    height, width = image1.shape[:-1]

    if width > height:
        #original size can be odd or even number,
        padding_top = math.floor((width - height) / 2)
        padding_bottom = math.ceil((width - height) / 2)

        image_padding_top = np.zeros((padding_top, width, 3), dtype=np.uint8)
        image_padding_bottom = np.zeros((padding_bottom, width, 3), dtype=np.uint8)

        image1 = np.concatenate((image_padding_top,image1,image_padding_bottom), axis=0)
    elif width < height:
        padding_left = math.floor((height - width) / 2)
        padding_right = math.ceil((height - width) / 2)

        image_padding_left = np.zeros((height, padding_left, 3), dtype=np.uint8)
        image_padding_right = np.zeros((height, padding_right, 3), dtype=np.uint8)

        image1 = np.concatenate((image_padding_left, image1, image_padding_right), axis=1)


    if image_size is not None:
        height, width = image1.shape[:-1] #image1 is square now

        if height > image_size:
            image1 = cv2.resize(image1, (image_size, image_size))
        elif height < image_size:
            if image_size > width:
                padding_left = math.floor((image_size - width) / 2)
                padding_right = math.ceil((image_size - width) / 2)

                image_padding_left = np.zeros((height, padding_left, 3), dtype=np.uint8)
                image_padding_right = np.zeros((height, padding_right, 3), dtype=np.uint8)

                image1 = np.concatenate((image_padding_left, image1, image_padding_right), axis=1)
                height, width = image1.shape[:-1]

            if image_size > height:
                padding_top = math.floor((image_size - height) / 2)
                padding_bottom = math.ceil((image_size - height) / 2)

                image_padding_top = np.zeros((padding_top, width, 3), dtype=np.uint8)
                image_padding_bottom = np.zeros((padding_bottom, width, 3), dtype=np.uint8)

                image1 = np.concatenate((image_padding_top, image1, image_padding_bottom), axis=0)
                height, width = img1.shape[:-1]

    if grayscale:
        #cv2.cvtColor only support unsigned int (8U, 16U) or 32 bit float (32F).
        # image_output = np.uint8(image_output)
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    return image1


def image_to_square_dir(source_dir, dest_dir, image_size=None, grayscale=False):
    if not source_dir.endswith('/'):
        source_dir = source_dir + '/'
    if not dest_dir.endswith('/'):
        dest_dir = dest_dir + '/'


    for dir_path, subpaths, files in os.walk(source_dir, False):
        for f in files:
            image_file_source = os.path.join(dir_path, f)

            file_base, file_ext = os.path.splitext(image_file_source)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue

            image_converted = image_to_square(image_file_source, image_size=image_size,
                                              grayscale=grayscale)

            image_file_dest = image_file_source.replace(source_dir, dest_dir)

            if not os.path.exists(os.path.dirname(image_file_dest)):
                os.makedirs(os.path.dirname(image_file_dest))

            print(image_file_dest)

            cv2.imwrite(image_file_dest, image_converted)

# 加载一个或者图像文件或者图像文件列表 到 一个list  (384,384,3)  my_images_aug使用
def load_resize_images(image_files, image_shape=None, grayscale=False):
    list_image = []

    if isinstance(image_files, list):   # list of image files
        for image_file in image_files:
            image_file = image_file.strip()

            if grayscale:
                image1 = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            else:
                image1 = cv2.imread(image_file)

            try:
                if (image_shape is not None) and (image1.shape[:2] != image_shape[:2]):
                        image1 = cv2.resize(image1, image_shape[:2])
            except:
                raise Exception("Invalid image:" + image_file)

            if image1 is None:
                raise Exception("Invalid image:" + image_file)

            if image1.ndim == 2:
                image1 = np.expand_dims(image1, axis=-1)

            list_image.append(image1)
    else:
        if isinstance(image_files, str):
            if grayscale:
                image1 = cv2.imread(image_files, cv2.IMREAD_GRAYSCALE)
            else:
                image1 = cv2.imread(image_files)
        else:
            if grayscale and image_files.ndim == 3:
                image1 = cv2.cvtColor(image_files, cv2.COLOR_BGR2GRAY)
            else:
                image1 = image_files

        try:
            if (image_shape is not None) and (image1.shape[:2] != image_shape[:2]):
                image1 = cv2.resize(image1, image_shape[:2])
        except:
            raise Exception("Invalid image:" + image_files)

        if image1 is None:
            raise Exception("Invalid image:" + image_files)

        if image1.ndim == 2:
            image1 = np.expand_dims(image1, axis=-1)

        list_image.append(image1)

    return list_image


def crop_image(img1, bottom, top, left, right):
    if isinstance(img1, str):
        img1 = cv2.imread(img1)

    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=-1)

    img2 = img1[bottom:top, left:right, :]

    return img2

def my_gen_img_tensor(file, image_shape=(299, 299, 3), imgaug_seq=None, imgaug_times=1):
    images = load_resize_images(file, image_shape)

    if imgaug_seq is None:
        x_test = images
    else:
        images_times = []
        for _ in range(imgaug_times):
            images_times += images

        x_test = imgaug_seq.augment_images(images_times)

    x_test = np.asarray(x_test, dtype=np.float16)
    x_test = input_norm(x_test)

    return x_test

if __name__ ==  '__main__':
    img1 = np.ones((50, 100, 3))
    img1 = img1 * 255
    cv2.imwrite('/tmp1/111.jpg', img1)

    img2 = image_to_square(img1, imgsize=150)
    cv2.imwrite('/tmp1/122.jpg', img2)
    exit(0)


