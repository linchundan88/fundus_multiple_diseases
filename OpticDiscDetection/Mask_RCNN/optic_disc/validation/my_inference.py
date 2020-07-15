
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from OpticDiscDetection.Mask_RCNN.optic_disc.helper import OpticDiscConfig
from OpticDiscDetection.Mask_RCNN.mrcnn import model as modellib

#region Configurations and loading model
config = OpticDiscConfig()
config.display()

config.IMAGES_PER_GPU = 1
config.BATCH_SIZE = 1   # len(image) must match BATCH_SIZE

model = modellib.MaskRCNN(mode="inference", config=config)

# weights_path = '/home/ubuntu/dlp/deploy_models_new/segmentation_optic_disk/mask_rcnn_opticdisc_0018_0.234.h5'
weights_path = '/home/ubuntu/dlp/deploy_models/ROP/segmentation_optic_disk/mask_rcnn_opticdisc_ROP_0018_val_0.245.h5'
model.load_weights(weights_path, by_name=True)
#endregion


#region image, preprocess
file_name_source = os.path.join(sys.path[0], 'image1.jpg')

from LIBS.ImgPreprocess import my_preprocess
image_preprocess = my_preprocess.do_preprocess(file_name_source, 384)

# cv2.imwrite('aaa.jpg', image_preprocess)

from OpticDiscDetection.Mask_RCNN.optic_disc.helper.my_seg_optic_disc import seg_optic_disc
file_name_mask = os.path.join(sys.path[0], 'mask4.jpg')
(confidence, img_dest) = seg_optic_disc(model, image_preprocess,
        os.path.join(sys.path[0], file_name_mask))



print('OK')