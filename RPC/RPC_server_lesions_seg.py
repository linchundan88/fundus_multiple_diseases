'''
DR lesion segmentation server
'''

import cv2
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""  #force Keras to use CPU

from xmlrpc.server import SimpleXMLRPCServer
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))

from keras.models import load_model
from LIBS.DataPreprocess import my_images_generator
from LIBS.ImgPreprocess import my_preprocess
import uuid

import my_config

#region load models during startup

file_save_path = '/tmp'
dir_models = my_config.dir_deploy_models

model_file_Hemorrhages = os.path.join(dir_models, 'lesions_seg/Hemorrhages-042-0.2920_val0.3097.hdf5')
# model_lesion = load_model(model_name_lesion, custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})
print('prepare load model ' + model_file_Hemorrhages)
model_Hemorrhages = load_model(model_file_Hemorrhages, compile=False)

model_file_HardExudates = os.path.join(dir_models, 'lesions_seg/HardExudates-011-0.299_val0.4188.hdf5')
print('prepare load model ' + model_file_HardExudates)
model_HardExudates = load_model(model_file_HardExudates, compile=False)

model_file_SoftExudates = os.path.join(dir_models, 'lesions_seg/SoftExudates-080-0.313_val0.3971.hdf5')
print('prepare load model ' + model_file_SoftExudates)
model_SoftExudates = load_model(model_file_SoftExudates, compile=False)

# model_file_LaserSpot = os.path.join(base_dir_models, 'LaserSpot-079-0.5179_val0.7846.hdf5') #add none
model_file_LaserSpot = os.path.join(dir_models, 'lesions_seg/LaserSpot-010-0.1702_val0.4477.hdf5') #add none
print('prepare load model ' + model_file_LaserSpot)
model_LaserSpot = load_model(model_file_LaserSpot, compile=False)

# model_file_FibrousProliferation = os.path.join(base_dir_models, 'FibrousProliferation-035-0.5156_val0.6734.hdf5')  #add none
model_file_FibrousProliferation = os.path.join(dir_models, 'lesions_seg/FibrousProliferation-018-0.4544.hdf5')  #add none
print('prepare load model ' + model_file_FibrousProliferation)
model_FibrousProliferation = load_model(model_file_FibrousProliferation, compile=False)

#endregion


def get_img_process(filename_source, image_size=384):
    image_shape = (image_size, image_size)
    # preprocess don't add black
    image1 = my_preprocess.do_preprocess(filename_source, image_size, add_black_pixel_ratio=0)
    filename_preprocess = os.path.join(file_save_path, str(uuid.uuid1()) + '.jpg')
    cv2.imwrite(filename_preprocess, image1)
    img_preprocess_seg = cv2.imread(filename_preprocess)  # (384,384,3)

    train_image_files = []
    train_image_files.append(filename_preprocess)

    gen1 = my_images_generator.my_Generator_seg_test(list_images=train_image_files,
                                                     image_shape=image_shape)
    x = gen1.__next__()

    return img_preprocess_seg, x

def img_seg_one_type(lesion_type, x):
    if lesion_type == 'Hemorrhages':
        model_lesion = model_Hemorrhages
    elif lesion_type == 'HardExudates':
        model_lesion = model_HardExudates
    elif lesion_type == 'SoftExudates':
        model_lesion = model_SoftExudates
    elif lesion_type == 'LaserSpot':
        model_lesion = model_LaserSpot
    elif lesion_type == 'FibrousProliferation':
        model_lesion = model_FibrousProliferation

    img_pred = model_lesion.predict_on_batch(x)    #1,384,384,1 batch,height,width,channels

    img_pred = img_pred[0, :, :, 0]  #(1,384,384,1)  (384,384)
    # img_pred = img_pred[0].reshape(img_pred[0].shape[0], img_pred[0].shape[1])
    img_pred = img_pred > 0.5
    img_pred = img_pred.astype(np.int8)    #0, 1
    img_pred = img_pred * 255

    return img_pred

def predict_lesions_seg(filename_source):
    # 返回预处理的文件名，和用来predict的tensor(x/255等)
    img_preprocess_seg, x = get_img_process(filename_source)

    # 原始图像和所有病变混合的图像
    img_blend_lesion = np.copy(img_preprocess_seg)
    # 用来声称病变颜色
    image_ones_2dim = np.ones((img_preprocess_seg.shape[0], img_preprocess_seg.shape[1]))

    dict_lesions = {}   #返回的数据结构 字典

    for lesion_type in ['Hemorrhages', 'HardExudates', 'SoftExudates',
            'LaserSpot',  ] :   #'FibrousProliferation'
        img_pred_lesion = img_seg_one_type(lesion_type, x)

        #region 生成病变区域的图像文件
        filename_lesion = os.path.join(file_save_path,
                    lesion_type + '_' + str(uuid.uuid4()) + '.jpg')

        cv2.imwrite(filename_lesion, img_pred_lesion)

        # 激光班假阳，先不用
        if lesion_type == 'LaserSpot':
            continue

        dict_lesions[lesion_type] = filename_lesion
        #endregion

        #region 在一张图像上 生成混合原图各种病变的图像

        # 周围0，病变1
        img_op = img_pred_lesion // 255
        img_op_3dim = np.stack((img_op, img_op, img_op),
                                       axis=-1)  # (384,384,3) 维度匹配才可以broadcast相乘

        # 周围-1，病变0
        img_op_reverse = img_op - 1
        # 周围1，病变0
        img_op_reverse = (-1) * img_op_reverse
        img_op_reverse_3dim = np.stack((img_op_reverse, img_op_reverse, img_op_reverse), axis=-1)  #(384,384,3) 维度匹配才可以broadcast相乘

        # 只保留非病变区域
        img_blend_lesion = img_blend_lesion * img_op_reverse_3dim

        # #确定每种病变的颜色 B G R  (0,0,255)
        if lesion_type == 'LaserSpot':   #黄色
            np_B = 0 * image_ones_2dim
            np_G = 255 * image_ones_2dim
            np_R = 255 * image_ones_2dim
        if lesion_type == 'FibrousProliferation':   #绿色
            np_B = 60 * image_ones_2dim
            np_G = 255 * image_ones_2dim
            np_R = 60 * image_ones_2dim
        if lesion_type == 'Hemorrhages':
            np_B = 0 * image_ones_2dim
            np_G = 0 * image_ones_2dim
            np_R = 255 * image_ones_2dim
        if lesion_type == 'HardExudates':
            np_B = 255 * image_ones_2dim
            np_G = 255 * image_ones_2dim
            np_R = 255 * image_ones_2dim
        if lesion_type == 'SoftExudates':
            np_B = 188 * image_ones_2dim
            np_G = 188 * image_ones_2dim
            np_R = 188 * image_ones_2dim

        if lesion_type not in ['Hemorrhages','HardExudates', 'SoftExudates',
                               'LaserSpot', 'FibrousProliferation']:
            np_B = 0 * image_ones_2dim
            np_G = 0 * image_ones_2dim
            np_R = 0 * image_ones_2dim
            continue

        img_lesion_color_3dim = np.stack((np_B, np_G, np_R), axis=-1)

        img_blend_lesion = img_blend_lesion + img_lesion_color_3dim * img_op_3dim

        #endregion

    filename_blend_lesions = os.path.join(file_save_path,
                                    'all_' + str(uuid.uuid4()) + '.jpg')

    cv2.imwrite(filename_blend_lesions, img_blend_lesion)
    dict_lesions['all_lesions'] = filename_blend_lesions

    return dict_lesions

#region test mode
if my_config.debug_mode:
    filename = 'IDRiD_51.jpg'
    dict1 = predict_lesions_seg(filename)

#endregion

# command prarmeter  only port number (no type as RPC_server_single_class.py)
if len(sys.argv) != 2:  # sys.argv[0]  exe file itself
    port = 22000
else:
    port = int(sys.argv[1])

server = SimpleXMLRPCServer(("localhost", port))
print("Listening on port: ", str(port))
server.register_function(predict_lesions_seg, "predict_lesions_seg")
server.serve_forever()

