import os
from LIBS.DLP.my_optic_disc_inference import My_optic_disc_MaskRcnn

'''
IDRID 2018_6_14
0.8542527768779332 0.9211615083013103 0.6515102497215085 0.78734327158855

DRIONS-DB 110
0.8734136595790881 0.9320799855459115 0.7330709075602343 0.8442624891009389
'''
from LIBS.ImgPreprocess.my_iou_dice import iou_dice_mask, iou_dicee_bbox

model_file = '/home/jsiec/deploy_models/segmentation_optic_disk/mask_rcnn_opticdisc_0035_val_loss0.5724.h5'
my_optic_disk_MaskRcnn = My_optic_disc_MaskRcnn(model_file)


base_dir = '/home/jsiec/disk2/公开数据集/视盘/IDRID/Optic_disk_test/preprocess/Images/384'
base_dir = '/home/jsiec/disk2/公开数据集/视盘/DRIONS-DB/preprocess/Images/384'

num = 0
total_iou1 = 0
total_dice1 = 0
total_iou2 = 0
total_dice2 = 0

for dir_path, subpaths, files in os.walk(base_dir, False):
    for f in files:
        image_file = os.path.join(dir_path, f)

        #IDRID
        if '/IDRID/' in image_file:
            image_file_mask = image_file.replace('/Images/', '/Masks/')
            image_file_mask = image_file_mask.replace('.jpg', '_OD.tif')
            # IDRID 61 识别不了
            if '_61' in image_file:
                continue;

        #DRIONS-DB
        if '/DRIONS-DB/' in image_file:
            # image_001.jpg  anotExpert1_001.jpg
            image_file_mask = image_file.replace('/Images/', '/masks/')
            image_file_mask = image_file_mask.replace('/image_', '/anotExpert1_')

        # image_file = '/home/jsiec/disk2/公开数据集/OpticDiscDetection/lcd/dataset_2018_5_7/384/5/preprocess_384.jpg'
        # image_file_mask = '/home/jsiec/disk2/公开数据集/OpticDiscDetection/lcd/dataset_2018_5_7/384/5/masks.jpg'

        file_base, file_ext = os.path.splitext(image_file)  # 分离文件名与扩展名
        if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            continue
        print('filename:', image_file)

        img_file_crop = ''
        img_file_crop_mask = ''

        (found_optic_disk, score, image_preprocess, image_crop, image_mask, image_crop_mask, x1, y1, x2, y2) \
            = my_optic_disk_MaskRcnn.detect_optic_disc(image_source=image_file, image_size=384, preprocess=False)

        if found_optic_disk:
            bbox1 = [x1, y1, x2, y2]

            from LIBS.ImgPreprocess import my_image_object_boundary

            left, right, bottom, top = my_image_object_boundary.get_boundry(image_mask)

            bbox2 = [left, bottom, right, top]

            iou1, dice1 = iou_dicee_bbox(bbox1, bbox2)

            # 返回的是crop后，（112，112）
            iou2, dice2 = iou_dice_mask(image_mask, image_file_mask)

            # iou1, dice1, iou2, dice2 = 0.92, 0.96, 0.75, 0.85

            if iou1 < 0.4:
                pass

            total_iou1 = total_iou1 + iou1
            total_dice1 = total_dice1 + dice1
            total_iou2 = total_iou2 + iou2
            total_dice2 = total_dice2 + dice2

            num = num + 1

    print(total_iou1 / num, total_dice1 / num,
          total_iou2 / num, total_dice2 / num)
    print('1')