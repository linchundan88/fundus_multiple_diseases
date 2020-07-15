"""
Mask R-CNN

"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("../")

from OpticDiscDetection.Mask_RCNN.optic_disc.helper.my_optic_disc_config import OpticDiscConfig, OpticDiscDataset
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from OpticDiscDetection.Mask_RCNN.mrcnn import model as modellib


def train(model):

    datafile_type = 'dataset3' #including ROP data
    filename_csv = os.path.join(os.path.abspath('..'),
                'optic_disc', 'datafiles', datafile_type, 'images_masks.csv')

    from LIBS.DataPreprocess.my_data import split_dataset
    train_image_files, train_mask_files, valid_image_files, valid_mask_files = split_dataset(
        filename_csv, valid_ratio=0.1, random_state=2223, field_columns=['images', 'masks'])

    # Training dataset.
    dataset_train = OpticDiscDataset()
    dataset_train.load_OpticDisc(train_image_files)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = OpticDiscDataset()
    dataset_val.load_OpticDisc(valid_image_files)
    dataset_val.prepare()

    #region  Image augmentation
    sometimes = lambda aug: iaa.Sometimes(0.96, aug)
    imgaug_seq = iaa.Sequential([
        # iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.Flipud(0.2),  # horizontally flip 10% of the images

        iaa.Crop(px=(0, 20)),

        sometimes(iaa.Affine(
            scale={"x": (0.92, 1.08), "y": (0.92, 1.08)},
            translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
            # translate by -20 to +20 percent (per axis)
            rotate=(0, 360),  # rotate by -45 to +45 degrees
        )),
    ])

    # endregion

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE1,
                epochs=10, #20
                augmentation=imgaug_seq,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE2,
                epochs=40,
                augmentation=imgaug_seq,
                layers='all')


# Create model
#  Configurations
config = OpticDiscConfig()
config.display()

model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir='/tmp4/OpticDiscMaskRCNN')

WEIGHTS_PATH = os.path.join(sys.path[0], 'weights', 'mask_rcnn_opticdisc_0018_0.234.h5')
model.load_weights(WEIGHTS_PATH, by_name=True)

# Training
train(model)

print('OK')

