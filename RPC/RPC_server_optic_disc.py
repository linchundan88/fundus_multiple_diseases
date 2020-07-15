'''
    RPC classification server
    发布的服务 Retinanet detect_optic_disc
    发布的服务 Mask RCNN detect_optic_disc_mask
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""  #force Keras to use CPU
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))

from LIBS.DLP.my_optic_disc_inference import My_optic_disc_MaskRcnn
# from LIBS.DLP.my_optic_disc_inference import My_optic_disc_RetinaNet, My_optic_disc_MaskRcnn
from xmlrpc.server import SimpleXMLRPCServer

import cv2
import uuid
import my_config
dir_models = my_config.dir_deploy_models
BASE_DIR_SAVE = my_config.dir_optic_disc

'''
def detect_optic_disc(file_source_512, preprocess=False):
    img_file_crop = ''

    (found_optic_disc, image_preprocess_384, image_crop, score, x1, y1, x2, y2) = \
        my_optic_disc_RetinaNet.detect_optic_disc(file_source_512, image_size=384,
                          preprocess=preprocess)

    if found_optic_disc:
        str_uuid = str(uuid.uuid1())
        img_file_crop =os.path.join(BASE_DIR_SAVE, str_uuid + '_OD_retinanet.jpg')
        cv2.imwrite(img_file_crop, image_crop)

    return found_optic_disc, img_file_crop
'''

def detect_optic_disc_mask(file_source_512, preprocess=False):
    img_file_crop = ''
    img_file_crop_mask = ''

    (found_optic_disc, score, image_preprocess, image_crop, img_masks, image_crop_mask, x1, y1, x2, y2) \
        = my_optic_disc_MaskRcnn.detect_optic_disc(image_source=file_source_512, image_size=384,
                                                   preprocess=preprocess)

    if found_optic_disc:
        str_uuid = str(uuid.uuid1())

        img_file_crop = os.path.join(BASE_DIR_SAVE, str_uuid + '_OD_maskrcnn.jpg')
        cv2.imwrite(img_file_crop, image_crop)

        img_file_crop_mask = os.path.join(BASE_DIR_SAVE, str_uuid + '_OD_Mask.jpg')
        # image_crop_mask shape (112,112)
        cv2.imwrite(img_file_crop_mask, image_crop_mask)

    return found_optic_disc, img_file_crop, img_file_crop_mask


# command prarmeter  only port number (no type as RPC_server_single_class.py)
if len(sys.argv) != 2:  # sys.argv[0]  exe file itself
    port = 21000
else:
    port = int(sys.argv[1])

model_file = dir_models + 'detection_optic_disk/converted_resnet50_csv_24.h5'
# my_optic_disc_RetinaNet = My_optic_disc_RetinaNet(model_file)


model_file = dir_models + 'segmentation_optic_disc/mask_rcnn_opticdisc_ROP_0022_loss0.2510.h5'

my_optic_disc_MaskRcnn = My_optic_disc_MaskRcnn(model_file)

#region test mode
if my_config.debug_mode:
    img_source = '/tmp1/brvo.jpg'

    # img_file_crop, img_file_crop_mask: 112*112
    found_optic_disc, img_file_crop, img_file_crop_mask = detect_optic_disc_mask(img_source, preprocess=True)

    # img_file_crop: image:112*112
    # found_optic_disc, img_file_crop = detect_optic_disc(img_source)

    print('OK')
#endregion

server = SimpleXMLRPCServer(("localhost", port))
print("Listening on port: ", str(port))
# server.register_function(detect_optic_disc, "detect_optic_disc")
server.register_function(detect_optic_disc_mask, "detect_optic_disc_mask")
server.serve_forever()

