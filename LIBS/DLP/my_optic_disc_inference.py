import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""  #force Keras to use CPU
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))

import keras
import cv2
import numpy as np
from LIBS.ImgPreprocess import my_preprocess


#region RetinaNet
'''
from retinanet_keras.keras_retinanet.models.resnet import custom_objects
from retinanet_keras.keras_retinanet.utils.image import preprocess_image, resize_image

class My_optic_disc_RetinaNet:

    def __init__(self, model_file):
        if isinstance(model_file, str):
            # load retinanet model
            # model_file = '/home/jsiec/deploy_models/detection_optic_disk/converted_resnet50_csv_24.h5'
            print('loading model:', model_file)
            self.model = keras.models.load_model(model_file, custom_objects=custom_objects, compile=False)
            print('load model complete:', model_file)
        else:
            self.model = model_file


    def detect_optic_disc(self, image_source, image_size=384, preprocess = True):

        # preprocess image for network
        if preprocess:
            image_preprocess = my_preprocess.do_preprocess(image_source, crop_size=image_size)
        else:
            # 不做预处理 裁剪， 为了计算IOU,DICE和原来的Mask匹配
            if isinstance(image_source, str):
                image_preprocess = cv2.imread(image_source)
                if image_preprocess.shape[1] != 384:
                    image_preprocess = cv2.resize(image_preprocess, (384, 384) )
            else:
                image_preprocess = image_source

        image = image_preprocess
        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=image_size, max_side=image_size)

        # process image
        # import time
        # start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        # print("processing time: ", time.time() - start)


        # Where boxes are shaped (None, None, 4) 例如(1,300,4) (for (x1, y1, x2, y2)),
        # scores is shaped (None, None) 例如(1,300) (classification score)
        # labels is shaped (None, None)  例如(1,300) (label corresponding to the score).


        for i in range(len(boxes[0])):
            if labels[0][i] == 0:  # 是所属类别
                score = scores[0][i]  # 置信度

                x1, y1, x2, y2 = boxes[0][i]
                print(score, x1, y1, x2, y2)

                found_optic_disc = True
                break  # 只要概率最高的
            else:
                found_optic_disc = False
                return found_optic_disc, image_preprocess, None, None, None, None, None, None

        if x2-x1 < 10 or y2-y1 < 10 or x2-x1 > 250 or y2-y1 > 250:
            found_optic_disc = False
            return found_optic_disc, image_preprocess, None, None, None, None, None, None

        center_x = (x2 + x1) // 2
        center_y = (y2 + y1) // 2
        r = max((x2-x1)//2, (y2-y1)//2)
        r = r + 40

        # 正方形
        left = int(max(0, center_x - r))
        right = int(min(image.shape[1], center_x + r))
        bottom = int(max(0, center_y - r))
        top = int(min(image.shape[1], center_y + r))

        image_crop = image_preprocess[bottom:top, left:right]
        image_crop = cv2.resize(image_crop, (112, 112))

        return found_optic_disc, image_preprocess, image_crop, score, x1, y1, x2, y2

'''
#endregion RetinaNet


#region Mask_Rcnn

from OpticDiscDetection.Mask_RCNN.mrcnn.config import Config
from OpticDiscDetection.Mask_RCNN.mrcnn import model as modellib


class InferenceConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "OpticDisc"

    BACKBONE = "resnet50"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    #  len(image) must match BATCH_SIZE
    BATCH_SIZE = 1

    #  Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384
    #
    # # Use smaller anchors because our image and objects are small
    # # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    # RPN_ANCHOR_SCALES = (16, 32, 64, 128)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    #
    # MEAN_PIXEL = np.array([127, 127, 127])
    #
    # # Reduce training ROIs per image because the images are small and have
    # # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + OpticDisk

    # # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7


class My_optic_disc_MaskRcnn:
    model = None

    def __init__(self, model_file):
        if isinstance(model_file, str):
            # load retinanet model
            inference_config = InferenceConfig()
            # config.BATCH_SIZE = 1  # len(image) must match BATCH_SIZE

            print('loading model:', model_file)
            model = modellib.MaskRCNN(mode="inference", config=inference_config)

            model.load_weights(model_file, by_name=True)
            print('load model complete:', model_file)

            self.model = model
        else:
            self.model = model_file


    def detect_optic_disc(self, image_source, image_size=384, preprocess = True):
        if preprocess:
            image_preprocess = my_preprocess.do_preprocess(image_source, crop_size=image_size,
                                add_black_pixel_ratio=0.05)
        else:
            # 不做预处理 裁剪， 为了计算IOU,DICE和原来的Mask匹配
            if isinstance(image_source, str):
                image_preprocess = cv2.imread(image_source)
                if image_preprocess.shape[1] != 384:
                    image_preprocess = cv2.resize(image_preprocess, (384, 384))
            else:
                image_preprocess = image_source

        image_preprocess = image_preprocess.astype(np.uint8)

        image_preprocess_RGB = cv2.cvtColor(image_preprocess, cv2.COLOR_BGR2RGB)

        # image_preprocess_RGB = np.load("/tmp/A.npy")

        # Run detection
        results = self.model.detect([image_preprocess_RGB], verbose=1)
        r = results[0]   #batch size = 1

        if len(r['class_ids']) > 0 and r['class_ids'][0] == 1:
            found_optic_disc = True
            score = r['scores'][0]

            y1, x1, y2, x2 = r['rois'][0]
            # cv2.rectangle(image_preprocess, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # cv2.imwrite('test1.jpg', image_preprocess)

            np_zero = np.zeros((384, 384, 1))
            np_ono = np.ones((384, 384, 1))
            np_255 = 255 * np_ono

            #img_masks shape (384,384,1) 可能多个masks
            temp_image = np.expand_dims(r['masks'][:, :, 0], axis=-1)
            img_masks = np.where(temp_image, np_255, np_zero)
            img_masks = img_masks.astype(np.uint8)

            #region 剪切图像 视盘区域，和掩码

            center_x = (x2 + x1) // 2
            center_y = (y2 + y1) // 2
            r = max((x2-x1)//2, (y2-y1)//2)
            r = r + 40

            # 正方形
            left = int(max(0, center_x - r))
            right = int(min(image_preprocess.shape[1], center_x + r))
            bottom = int(max(0, center_y - r))
            top = int(min(image_preprocess.shape[1], center_y + r))

            image_crop = image_preprocess[bottom:top, left:right]
            image_crop = cv2.resize(image_crop, (112, 112))

            image_crop_mask = img_masks[bottom:top, left:right]
            #image_crop_mask shape (112,112)
            image_crop_mask = cv2.resize(image_crop_mask, (112, 112))

            # endregion剪切图像

            return found_optic_disc, score, image_preprocess, image_crop, img_masks, image_crop_mask, x1, y1, x2, y2

        return False, 0, image_preprocess, None, None, None, 0, 0, 0, 0

#endregion Mask_Rcnn

