import csv
import os
import uuid
import pandas as pd
import cv2
from LIBS.ImgPreprocess.my_image_helper import load_resize_images
from LIBS.ImgPreprocess.my_image_object_boundary import get_boundry


def imgaug_seg_masks(filename_csv_images_masks, dir_dest, imgaug_seq, aug_times=10):
    df = pd.read_csv(filename_csv_images_masks)

    for _, row in df.iterrows():
        filename_image = row["images"].strip()
        filename_mask = row["masks"].strip()

        list_images = load_resize_images(filename_image)  # 训练文件列表
        list_masks = load_resize_images(filename_mask, grayscale=True)

        for _ in range(aug_times):
            seq_det = imgaug_seq.to_deterministic()

            x_train = seq_det.augment_images(list_images)
            y_train = seq_det.augment_images(list_masks)

            str_uuid = str(uuid.uuid1())
            file_dst_image = os.path.join(dir_dest, str_uuid+'_img.jpg')
            file_dst_mask = os.path.join(dir_dest, str_uuid + '_msk.jpg')

            if not os.path.exists(os.path.dirname(file_dst_image)):
                os.makedirs(os.path.dirname(file_dst_image))

            cv2.imwrite(file_dst_image, x_train[0])
            cv2.imwrite(file_dst_mask, y_train[0])


def convert_masks_pascal(filename_csv_images_masks, filename_csv_pascal):
    df = pd.read_csv(filename_csv_images_masks)

    with open(filename_csv_pascal, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')

        for _, row in df.iterrows():
            filename_image = row["images"].strip()
            filename_mask = row["masks"].strip()

            print(filename_image)

            (left, right, bottom, top) = get_boundry(filename_mask)
            csv_writer.writerow([filename_image, left, right, bottom, top, 'OpticDisc'])


if __name__ == '__main__':
    filename_csv_images_masks = 'images_masks_before_refuge.csv'
    filename_csv_pascal = 'test1.csv'


    from imgaug import augmenters as iaa
    sometimes = lambda aug: iaa.Sometimes(0.96, aug)
    # sometimes1 = lambda aug: iaa.Sometimes(0.96, aug)
    imgaug_seq = iaa.Sequential([
        # iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        # sometimes(iaa.CropAndPad(
        #     percent=(-0.04, 0.04),
        #     pad_mode=ia.ALL,
        #     pad_cval=(0, 255)
        # )),
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.Flipud(0.2),  # horizontally flip 50% of the images
        # iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0,
        # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
        # sometimes(iaa.Crop(percent=(0, 0.1))),  # crop images by 0-10% of their height/width
        # shuortcut for CropAndPad

        # improve or worsen the contrast  If PCH is set to true, the process happens channel-wise with possibly different S.
        # sometimes1(iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5), ),
        # change brightness of images (by -5 to 5 of original value)
        # sometimes1(iaa.Add((-6, 6), per_channel=0.5),),
        sometimes(iaa.Affine(
            # scale={"x": (0.92, 1.08), "y": (0.92, 1.08)},
            # scale images to 80-120% of their size, individually per axis
            # Translation Shifts the pixels of the image by the specified amounts in the x and y directions
            translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
            rotate=(-180, 180),  # rotate by -10 to +10 degrees
            # shear=(-16, 16),  # shear by -16 to +16 degrees
            # order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            # cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
    ])

    imgaug_seg_masks(filename_csv_images_masks, '/tmp2/a', imgaug_seq=imgaug_seq)

    print('OK')