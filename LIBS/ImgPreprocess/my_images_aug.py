'''

imgs_aug 输入 list_image_files

img_aug_seg 输入 list_image_files, list_masks_files
img_aug_seg_multiclass

'''

# import keras.preprocessing.image
import cv2
from LIBS.ImgPreprocess import my_preprocess, my_image_helper
from imgaug import augmenters as iaa

# img_aug 只增镪 images
# 可以使用多种变换方式(mode) :img_aug_mode_rotate_flip 1: rotate and crop,
#  2:don't rotate, 3:don't flip(left right eye)

def imgs_aug(list_image_files, image_shape=(299, 299, 3), train_or_valid='train',
             img_aug_mode_rotate_flip=1,
             img_aug_mode_contrast=False):
    # list_image a list of 3d numpy array (height,weight,channel)
    # (batch, size, size , 3) 一维数组batch个元素 (size, size , 3)
    list_images = my_image_helper.load_resize_images(list_image_files, image_shape)

    if train_or_valid == 'valid':
        return list_images

    if train_or_valid == 'test':
        return list_images

    if train_or_valid == 'train':
        if img_aug_mode_rotate_flip == 1:
            sometimes = lambda aug: iaa.Sometimes(0.96, aug)
            # https://github.com/aleju/imgaug/blob/master/images/examples_crop.jpg
            # https://github.com/aleju/imgaug/blob/master/images/examples_affine_translate.jpg
            seq = iaa.Sequential([
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
                    translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-10, 10),  # rotate by -10 to +10 degrees
                    # shear=(-16, 16),  # shear by -16 to +16 degrees
                    # order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    # cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
            ])

        #  don't rotate
        elif img_aug_mode_rotate_flip == 2:
            sometimes = lambda aug: iaa.Sometimes(0.96, aug)
            # sometimes1 = lambda aug: iaa.Sometimes(0.96, aug)
            seq = iaa.Sequential([
                # iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
                # sometimes(iaa.CropAndPad(
                #     percent=(-0.04, 0.04),
                #     pad_mode=ia.ALL,
                #     pad_cval=(0, 255)
                # )),
                iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                iaa.Flipud(0.2),  # vertically  flip 50% of the images
                # iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0,
                # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                # sometimes(iaa.Crop(percent=(0, 0.1))),  # crop images by 0-10% of their height/width
                # shuortcut for CropAndPad

                # improve or worsen the contrast  If PCH is set to true, the process happens channel-wise with possibly different S.
                # sometimes1(iaa.ContrastNormalization((0.92, 1.08), per_channel=0.5), ),
                # change brightness of images (by -5 to 5 of original value)
                # sometimes(iaa.Add((-5, 5), per_channel=0.5),),
                sometimes(iaa.Affine(
                    # scale={"x": (0.92, 1.08), "y": (0.92, 1.08)},
                    # scale images to 80-120% of their size, individually per axis
                    # Translation Shifts the pixels of the image by the specified amounts in the x and y directions
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    # translate by -20 to +20 percent (per axis)
                    # rotate=(0, 360),  # rotate by -45 to +45 degrees
                    # shear=(-16, 16),  # shear by -16 to +16 degrees
                    # order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    # cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
            ])

        #Fliplr(0.5) don't rotate, don't  flip
        elif img_aug_mode_rotate_flip == 3:
            sometimes = lambda aug: iaa.Sometimes(0.96, aug)
            # sometimes1 = lambda aug: iaa.Sometimes(0.96, aug)
            seq = iaa.Sequential([
                # iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
                # sometimes(iaa.CropAndPad(
                #     percent=(-0.04, 0.04),
                #     pad_mode=ia.ALL,
                #     pad_cval=(0, 255)
                # )),
                # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                # iaa.Flipud(0.2),  # vertically  flip 50% of the images
                # iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0,
                # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                # sometimes(iaa.Crop(percent=(0, 0.1))),  # crop images by 0-10% of their height/width
                # shuortcut for CropAndPad

                # improve or worsen the contrast  If PCH is set to true, the process happens channel-wise with possibly different S.
                # sometimes1(iaa.ContrastNormalization((0.92, 1.08), per_channel=0.5), ),
                # change brightness of images (by -5 to 5 of original value)
                # sometimes(iaa.Add((-5, 5), per_channel=0.5),),
                sometimes(iaa.Affine(
                    # scale={"x": (0.92, 1.08), "y": (0.92, 1.08)},
                    # scale images to 80-120% of their size, individually per axis
                    # Translation Shifts the pixels of the image by the specified amounts in the x and y directions
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    # translate by -20 to +20 percent (per axis)
                    # rotate=(0, 360),  # rotate by -45 to +45 degrees
                    # shear=(-16, 16),  # shear by -16 to +16 degrees
                    # order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    # cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
            ])


        images_aug = seq.augment_images(list_images)

        # 做一个对比度的变换
        if img_aug_mode_contrast:
            sometimes1 = lambda aug: iaa.Sometimes(0.96, aug)
            seq2 = iaa.Sequential([
                sometimes1(iaa.ContrastNormalization((0.92, 1.08), per_channel=0.5), ),
                # change brightness of images (by -5 to 5 of original value)
                # sometimes1(iaa.Add((-6, 6), per_channel=0.5),), #不能加这句，否则要坏
            ])

            images_aug = seq2.augment_images(images_aug)

        return images_aug


# img_aug_seg 同时增镪 images，和masks
def img_aug_seg(list_images_files, list_masks_files, image_shape, train_or_valid='train', img_aug_mode_rotate_flip=1,
                img_aug_mode_contrast=False):
    # list_images a list of 3d numpy array (height,weight,channel)
    # (batch, size, size , 3) 一维数组batch个元素 (size, size , 3)
    list_images = my_image_helper.load_resize_images(list_images_files, image_shape)  # 训练文件列表

    if train_or_valid in ['train', 'valid']:
        # (batch, size, size , 1) 一维数组batch个元素 (size, size , 1) mask标注文件列
        list_masks = my_image_helper.load_resize_images(list_masks_files, image_shape, grayscale=True)

    if train_or_valid == 'valid':  # 不做变换直接返回, 验证有标注的
        return list_images, list_masks

    # 标注图像增强以后，0,255,会有一些1,2,3,4
    if train_or_valid == 'train':
        #不同img_aug_mode 定义 不同的 seq1
        # 1, don't rotate , 2 do rotate  (3,4) change contrast
        if img_aug_mode_rotate_flip == 1:
            sometimes = lambda aug: iaa.Sometimes(0.96, aug)
            seq1 = iaa.Sequential([
                # iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
                iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                iaa.Flipud(0.2),  # vertically  flip 50% of the images

                # iaa.Crop(px=(0, 10)),

                # sometimes(iaa.ContrastNormalization((0.92, 1.08), per_channel=0.5), ),
                sometimes(iaa.Affine(
                    # scale={"x": (0.92, 1.08), "y": (0.92, 1.08)},
                    translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(0, 360),  # rotate by -45 to +45 degrees
                )),
            ])

        if img_aug_mode_rotate_flip == 2:
            sometimes = lambda aug: iaa.Sometimes(0.96, aug)
            seq1 = iaa.Sequential([
                # iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
                iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                iaa.Flipud(0.2),  # vertically  flip 50% of the images

                # iaa.Crop(px=(0, 6)),

                # sometimes(iaa.ContrastNormalization((0.92, 1.08), per_channel=0.5), ),
                sometimes(iaa.Affine(
                    # scale={"x": (0.92, 1.08), "y": (0.92, 1.08)},
                    translate_percent={"x": (-0.06, 0.06), "y": (-0.05, 0.05)},
                    # translate by -20 to +20 percent (per axis)
                    # rotate=(0, 360),  # rotate by -45 to +45 degrees
                )),
            ])

        if img_aug_mode_rotate_flip == 3:
            sometimes = lambda aug: iaa.Sometimes(0.96, aug)
            seq1 = iaa.Sequential([
                # iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
                # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                # iaa.Flipud(0.2),  # vertically  flip 50% of the images

                # iaa.Crop(px=(0, 6)),

                # sometimes(iaa.ContrastNormalization((0.92, 1.08), per_channel=0.5), ),
                sometimes(iaa.Affine(
                    # scale={"x": (0.92, 1.08), "y": (0.92, 1.08)},
                    translate_percent={"x": (-0.06, 0.06), "y": (-0.05, 0.05)},
                    # translate by -20 to +20 percent (per axis)
                    # rotate=(0, 360),  # rotate by -45 to +45 degrees
                )),
            ])

        # The deterministic sequence will always apply the exactly same effects to the images.
        # 两次之间的变换一样， 但是每次批量内变换不一样
        seq_det = seq1.to_deterministic()

        images_aug = seq_det.augment_images(list_images)
        images_aug_annotations = seq_det.augment_images(list_masks)

        # #图像可以在做一个变换，标注不用了，因为标注是黑白值
        if img_aug_mode_contrast:
            sometimes1 = lambda aug: iaa.Sometimes(0.96, aug)
            seq2 = iaa.Sequential([
                sometimes1(iaa.ContrastNormalization((0.92, 1.08), per_channel=0.5), ),
                # change brightness of images (by -5 to 5 of original value)
                # sometimes1(iaa.Add((-6, 6), per_channel=0.5),), #不能加这句，否则要坏
            ])

            images_aug = seq2.augment_images(images_aug)


        return images_aug, images_aug_annotations

def img_aug_seg_test(list_images_files,  image_shape):
    # list_images a list of 3d numpy array (height,weight,channel)
    # (batch, size, size , 3) 一维数组batch个元素 (size, size , 3)
    list_images = my_image_helper.load_resize_images(list_images_files, image_shape)  # 训练文件列表

    # 不做变换直接返回， 测试无标注
    return list_images



def img_aug_seg_multiclass(list_images_files, list_masks_files, image_shape, train_or_valid='train', img_aug_mode_rotate_flip=1,
                           img_aug_mode_contrast=False):
    # list_image a list of 3d numpy array (height,weight,channel)
    # (batch, size, size , 3) 一维数组batch个元素 (size, size , 3)
    list_images = my_image_helper.load_resize_images(list_images_files, image_shape)  # 训练文件列表

    if train_or_valid == 'test':  # 不做变换直接返回， 测试无标注
        return list_images

    if train_or_valid in ['train', 'valid']:
        # (batch, size, size , 1) 一维数组batch个元素 (size, size , 1) mask标注文件列
        list_masks = my_image_helper.load_resize_images(list_masks_files, image_shape, grayscale=True)

    if train_or_valid == 'valid':  # 不做变换直接返回, 验证有标注的
        return list_images, list_masks

    # 标注图像增强以后，0,255,会有一些1,2,3,4
    if train_or_valid == 'train':
        #不同img_aug_mode 定义 不同的 seq1
        # 1, don't rotate , 2 do rotate  (3,4) change contrast
        if img_aug_mode_rotate_flip == 1:
            sometimes = lambda aug: iaa.Sometimes(0.96, aug)
            seq1 = iaa.Sequential([
                # iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
                iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                iaa.Flipud(0.2),  # vertically  flip 50% of the images

                # iaa.Crop(px=(0, 10)),

                # sometimes(iaa.ContrastNormalization((0.92, 1.08), per_channel=0.5), ),
                sometimes(iaa.Affine(
                    # scale={"x": (0.92, 1.08), "y": (0.92, 1.08)},
                    translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(0, 360),  # rotate by -45 to +45 degrees
                )),
            ])

        if img_aug_mode_rotate_flip == 2:
            sometimes = lambda aug: iaa.Sometimes(0.96, aug)
            seq1 = iaa.Sequential([
                # iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
                iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                iaa.Flipud(0.2),  # vertically  flip 50% of the images

                # iaa.Crop(px=(0, 6)),

                # sometimes(iaa.ContrastNormalization((0.92, 1.08), per_channel=0.5), ),
                sometimes(iaa.Affine(
                    # scale={"x": (0.92, 1.08), "y": (0.92, 1.08)},
                    translate_percent={"x": (-0.06, 0.06), "y": (-0.05, 0.05)},
                    # translate by -20 to +20 percent (per axis)
                    # rotate=(0, 360),  # rotate by -45 to +45 degrees
                )),
            ])

        if img_aug_mode_rotate_flip == 3:
            sometimes = lambda aug: iaa.Sometimes(0.96, aug)
            seq1 = iaa.Sequential([
                # iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
                # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                # iaa.Flipud(0.2),  # vertically  flip 50% of the images

                # iaa.Crop(px=(0, 6)),

                # sometimes(iaa.ContrastNormalization((0.92, 1.08), per_channel=0.5), ),
                sometimes(iaa.Affine(
                    # scale={"x": (0.92, 1.08), "y": (0.92, 1.08)},
                    translate_percent={"x": (-0.06, 0.06), "y": (-0.05, 0.05)},
                    # translate by -20 to +20 percent (per axis)
                    # rotate=(0, 360),  # rotate by -45 to +45 degrees
                )),
            ])

        # The deterministic sequence will always apply the exactly same effects to the images.
        # 两次之间的变换一样， 但是每次批量内变换不一样
        seq_det = seq1.to_deterministic()

        images_aug = seq_det.augment_images(list_images)

        images_aug_annotations = []
        for file_mask in list_masks:
            list_mask = [file_mask]
            tmp_1 = seq_det.augment_images(list_mask)
            # cv2.imwrite('/tmp/000.jpg', tmp_1[0]) #test
            images_aug_annotations.append(tmp_1[0])

        # #图像可以在做一个变换，标注不用了，因为标注是黑白值
        if img_aug_mode_contrast:
            sometimes1 = lambda aug: iaa.Sometimes(0.96, aug)
            seq2 = iaa.Sequential([
                sometimes1(iaa.ContrastNormalization((0.92, 1.08), per_channel=0.5), ),
                # change brightness of images (by -5 to 5 of original value)
                # sometimes1(iaa.Add((-6, 6), per_channel=0.5),), #不能加这句，否则要坏
            ])

            images_aug = seq2.augment_images(images_aug)


        return images_aug, images_aug_annotations

#test imgaug
if __name__ == '__main__':
    file1 = '/home/jsiec/disk1/PACS/DR-病灶精标/preprocess_384/本院_补充漏标/2018_04_09本院/00-100153/LHP19610102_20170105_160218_Color_R_001.jpg'

    x = imgs_aug(file1, image_shape=(299, 299, 3), img_aug_mode_rotate_flip=2)

    cv2.imwrite('/tmp/0000.jpg', x[0])
    print('OK')


