
import os
from OpticDiscDetection.Mask_RCNN.optic_disc.helper.my_seg_optic_disc import seg_optic_disc, optic_disc_draw_circle, crop_posterior
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
# Import Mask RCNN
from OpticDiscDetection.Mask_RCNN.optic_disc.helper.my_optic_disc_config import OpticDiscConfig
from OpticDiscDetection.Mask_RCNN.mrcnn import model as modellib

#region Configurations and loading model
config = OpticDiscConfig()
config.display()

config.IMAGES_PER_GPU = 1
config.BATCH_SIZE = 1   # len(image) must match BATCH_SIZE

model = modellib.MaskRCNN(mode="inference", config=config)

# weights_path = '/home/ubuntu/dlp/deploy_models_new/segmentation_optic_disk/mask_rcnn_opticdisc_0018_0.234.h5'
# weights_path = '/home/ubuntu/dlp/deploy_models/ROP/segmentation_optic_disk/mask_rcnn_opticdisc_ROP_0018_val_0.245.h5'
weights_path = '/home/ubuntu/dlp/deploy_models/ROP/segmentation_optic_disc/mask_rcnn_opticdisc_0022_loss0.2510.h5'

model.load_weights(weights_path, by_name=True)
#endregion

dir_original = '/media/ubuntu/data1/ROP项目/ROP训练图集汇总_20200104_PLUS+对照/original'
dir_preprocess = '/media/ubuntu/data1/ROP项目/ROP训练图集汇总_20200104_PLUS+对照/preprocess384'
dir_crop_optic_disc = '/media/ubuntu/data1/ROP项目/ROP训练图集汇总_20200104_PLUS+对照/crop_optic_disc_crop_circle'
dir_draw_circle = '/media/ubuntu/data1/ROP项目/ROP训练图集汇总_20200104_PLUS+对照/draw_circle'
dir_dest = '/media/ubuntu/data1/ROP项目/ROP训练图集汇总_20200104_PLUS+对照/optic_disc_seg'
dir_dest_error = '/media/ubuntu/data1/ROP项目/ROP训练图集汇总_20200104_PLUS+对照/optic_disc_seg_error'

for dir_path, subpaths, files in os.walk(dir_preprocess, False):
    for f in files:
        img_file_source = os.path.join(dir_path, f)

        filename, file_extension = os.path.splitext(img_file_source)
        if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
            print('file ext name:', f)
            continue

        img_file_dest = img_file_source.replace(dir_preprocess, dir_dest)
        image_shape = (384, 384, 1)

        (confidence, img_file_mask, circle_center, circle_diameter) = seg_optic_disc(model, img_file_source,
                img_file_dest, image_shape=image_shape, return_optic_disc_postition=True)
        if confidence is not None:
            print('detect optic disc successfully! ', img_file_source)

            img_draw_circle = optic_disc_draw_circle(img_file_source, circle_center, circle_diameter, diameter_times=3)
            img_file_draw_circle = img_file_source.replace(dir_preprocess, dir_draw_circle)
            if not os.path.exists(os.path.dirname(img_file_draw_circle)):
                os.makedirs(os.path.dirname(img_file_draw_circle))
            cv2.imwrite(img_file_draw_circle, img_draw_circle)

            img_crop_optic_disc = crop_posterior(img_file_source, circle_center, circle_diameter,
                    diameter_times=3, image_size=299, crop_circle=True)
            img_file_crop_optic_disc = img_file_source.replace(dir_preprocess, dir_crop_optic_disc)
            if not os.path.exists(os.path.dirname(img_file_crop_optic_disc)):
                os.makedirs(os.path.dirname(img_file_crop_optic_disc))
            cv2.imwrite(img_file_crop_optic_disc, img_crop_optic_disc)

        else:
            print('detect optic disc fail! ', img_file_source)
            img_file_original = img_file_source.replace(dir_preprocess, dir_original)
            img_file_dest_error = img_file_source.replace(dir_preprocess, dir_dest_error)
            #
            if not os.path.exists(os.path.dirname(img_file_dest_error)):
                os.makedirs(os.path.dirname(img_file_dest_error))
            import shutil
            shutil.copy(img_file_original, img_file_dest_error)

print('OK')

#