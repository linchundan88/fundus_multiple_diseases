
import os
import sys
import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from OpticDiscDetection.Mask_RCNN.mrcnn.config import Config
from OpticDiscDetection.Mask_RCNN.mrcnn import utils


class OpticDiscConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "OpticDisc"

    BACKBONE = "resnet50"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 8

    #  Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384
    #
    # # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (16, 32, 64, 128)  # anchor side in pixels
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_ANCHOR_SCALES = (8, 16, 32, 48, 64)
    # MEAN_PIXEL = np.array([127, 127, 127])
    #
    # # Reduce training ROIs per image because the images are small and have
    # # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + OpticDisc

    # # Number of training steps per epoch
    STEPS_PER_EPOCH = 120
    #
    VALIDATION_STEPS = 12
    #
    # # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

class OpticDiscDataset(utils.Dataset):

    def load_OpticDisc(self, image_files):
        # Add classes. We have only one class to add.
        self.add_class("OpticDisc", 1, "OpticDisc")

        for image_file in image_files:
            self.add_image(
                "OpticDisc",
                image_id=image_file,  # use file name as a unique image id
                path=image_file)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        image_info = self.image_info[image_id]
        # image_info['id'] == image_info['path']

        image_mask_file = image_info['path']

        #region 眼底图像文件 到 mask 文件名的转换
        if '/DRIONS-DB/' in image_mask_file:
            image_mask_file = image_mask_file.replace('Images/', '/masks/')
            # anotExpert1_006.jpg image_006.jpg

            temp_dir = os.path.dirname(image_mask_file)
            temp_filename = os.path.basename(image_mask_file)
            temp_filename = temp_filename.replace('image_', 'anotExpert1_')
            image_mask_file = os.path.join(temp_dir, temp_filename)

        elif '/IDRID/' in image_mask_file:
            image_mask_file = image_mask_file.replace('Images/', '/masks/')
            #   IDRiD_01.jpg  IDRiD_01_OD.tif

            temp_dir = os.path.dirname(image_mask_file)
            temp_filename = os.path.basename(image_mask_file)
            temp_filename = temp_filename.replace('.jpg', '_OD.tif')
            image_mask_file = os.path.join(temp_dir, temp_filename)

        elif '/ljw/' in image_mask_file or '/Drishti-GS1_files/' in image_mask_file:
            image_mask_file = image_mask_file.replace('Images/', '/masks/')
            #drishtiGS_001.png  drishtiGS_001.png
            # drishtiGS_ and ljw the same

        elif '/Refuge/' in image_mask_file:
            image_mask_file = image_mask_file.replace('images/', '/masks/')
            image_mask_file = image_mask_file.replace('.jpg', '.bmp')

        else:
            # '/lcd/  ,dataset3_screen2 , # '/lcd/' in image_mask_file:
            # masks.jpg preprocess_384.jpg
            image_mask_file = image_mask_file.replace('images/', '/masks/')

            image_mask_file = image_mask_file.replace('preprocess_384.jpg', 'masks.jpg')
            image_mask_file = image_mask_file.replace('preprocess_512.jpg', 'masks.jpg')
            image_mask_file = image_mask_file.replace('preprocess_448.jpg', 'masks.jpg')
            image_mask_file = image_mask_file.replace('preprocess_224.jpg', 'masks.jpg')

        #endregion

        image_mask = cv2.imread(image_mask_file)
        if image_mask.ndim == 3:
            image_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)  #shape (384,384)

        # 二值化  Second output is our thresholded image
        threthold = 10
        ret, img_thresh1 = cv2.threshold(image_mask, threthold, 255, cv2.THRESH_BINARY)
        image2 = np.expand_dims(img_thresh1, axis=-1)  #shape (384,384,1)
        image3 = image2 > threthold


        return image3, np.ones([image2.shape[-1]], dtype=np.int32)

