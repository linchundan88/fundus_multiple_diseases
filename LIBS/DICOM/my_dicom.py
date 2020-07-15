'''
get_dicom:
    input dicom file, output 3d ndarray,

get_npy
    used by 3d dicom imgaug, input list of npy files, output list of [128,128,64]

convert_dicom_array
    input 3d ndarray, output 3d ndarray, subsampling z axis,
    transpose(OCT), crop, resize, subsampling z axis,
'''

import os
import pydicom
import numpy as np
import cv2
from imgaug import augmenters as iaa
import math

# input dicom file, output 3d ndarray, transposed
def get_dicom_array(filename):
    # OCT: (885, 512, 128), Fundus image:(2000, 2992, 3)
    ds2 = pydicom.dcmread(filename)
    array1 = ds2.pixel_array
    return array1

def get_npy(files_batch):
    list_image = []

    for image_file in files_batch:
        array1 = np.load(image_file)  # shape (128,128,128)
        list_image.append(array1)

    return list_image

def array3d_transpose_crop(array1, transponse=True, crop=True, img_size=128,
                           z_interval=0):

    if transponse:
        # shape (885, 512, 128) (height, width, slices)
        # array1.shape  (128,885,512)   Channel First(Slice)
        array1 = array1.transpose(1, 2, 0)

    if not crop:
        return array1

    if z_interval > 0: # only keep some z channels
        for i in range(array1.shape[2]):
            if i % z_interval == 0:
                tmp_array = np.expand_dims(array1[:, :, i], axis=-1)

                if i == 0:
                    array_subsampoling_z = tmp_array
                else:
                    array_subsampoling_z = np.concatenate((array_subsampoling_z, tmp_array), axis=-1)
    else:
        array_subsampoling_z = array1

    # region Vertical cutting
    array_tmp = np.sum(array_subsampoling_z, axis=(1, 2)) #shape (885)
    max1 = np.max(array_tmp)

    top = 0
    bottom = array_tmp.shape[0] - 1

    for i in range(array_tmp.shape[0]):
        if array_tmp[i] > max1 / 6:
            top = i
            break
    top = max(0, top - 50)

    for i in range(array_tmp.shape[0]-1, -1, -1):
        if array_tmp[i] > max1 / 6:
            bottom = i
            break
    bottom = min(bottom + 20, array_tmp[i])

    array_cropping = array_subsampoling_z[top:bottom, :, :]

    #endregion

    #zero padding if necessary, in order to get square
    min_border = min(array_subsampoling_z.shape[:-1])

    if array_cropping.shape[0] < min_border:
        total_margion = min_border-array_cropping.shape[0]
        top_margin = math.floor(total_margion/2)
        array_temp_top = np.zeros((top_margin, array_cropping.shape[1], array_cropping.shape[2]))
        array_cropping = np.concatenate((array_temp_top, array_cropping), axis=0)
        bottom_margin = math.ceil(total_margion/2)
        array_temp_bottom = np.zeros((bottom_margin, array_cropping.shape[1], array_cropping.shape[2]))
        array_cropping = np.concatenate((array_cropping, array_temp_bottom), axis=0)

    if img_size is not None:
        array_resize = None
        for i in range(array_cropping.shape[2]):
            img1 = cv2.resize(array_cropping[:, :, i], (img_size, img_size))
            img1 = np.expand_dims(img1, axis=-1)

            if array_resize is None:
                array_resize = img1
            else:
                array_resize = np.concatenate((array_resize, img1), axis=-1)

        return array_resize
    else:
        return array_cropping


# OCT file size bigger than THRETHOLD_FILESIZE
THRETHOLD_FILESIZE = 5*1024*1024

def is_oct_file(img_file_source):
    _, file_extension = os.path.splitext(img_file_source)
    if file_extension.upper() not in ['.DCM', '.DICOM']:
        return False

    if os.path.getsize(img_file_source) < THRETHOLD_FILESIZE:  # 5MB
        return False

    try:
        array1 = get_dicom_array(img_file_source)
    except:
        return False

    if array1.shape[2] == 3:  # color fundus image
        return False
    else:
        return True


# 将包含DICOM目录的每一个文件，存成许多切片图像文件, 给人看的
def dicom_save_dirs(source_dir, dir_dest_base, transpose=True, crop=True,
                    save_npy=True, img_size=128, z_interval=0):

    for dir_path, subpaths, files in os.walk(source_dir, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)

            if not is_oct_file(img_file_source):
                print('not a OCT dicom file:', img_file_source)
                continue
            else:
                print('processing OCT file:{}'.format(img_file_source))

            array1 = get_dicom_array(img_file_source)
            array1 = array3d_transpose_crop(array1, transponse=transpose,
                                    crop=crop, img_size=img_size, z_interval=z_interval)

            _, filename = os.path.split(img_file_source)
            filename_stem = os.path.splitext(filename)[0]

            dir_dest = os.path.join(dir_dest_base, filename_stem)

            pat_id = f.replace('.dcm', '')
            pat_id = pat_id.replace('.DCM', '')
            pat_id = pat_id.replace('.dicom', '')
            pat_id = pat_id.replace('.DICOM', '')

            if save_npy:
                filename = os.path.join(dir_dest, pat_id + '.img.npy')
                if not os.path.exists(os.path.dirname(filename)):
                    os.makedirs(os.path.dirname(filename))
                np.save(filename, array1)

            for i in range(array1.shape[-1]):  #channel last, slice last
                filename = os.path.join(dir_dest, 'i_' + str(i) + '.png')
                if not os.path.exists(os.path.dirname(filename)):
                    os.makedirs(os.path.dirname(filename))
                cv2.imwrite(filename, array1[:, :, i])



