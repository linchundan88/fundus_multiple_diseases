'''
    RPC Service for CAM  and Grad-CAM++
    support both Multi-Class and Multi_Label
'''

import os
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from xmlrpc.server import SimpleXMLRPCServer
from keras.layers import *
from keras.models import Model
import keras
import cv2
import uuid

import LIBS.ImgPreprocess.my_image_helper
from LIBS.ImgPreprocess import my_preprocess
from scipy.ndimage.interpolation import zoom

from LIBS.CNN_Models.Utils.my_utils import get_last_conv_layer_name, get_last_conv_layer_number
from vis.visualization import visualize_cam

import my_config
DIR_MODELS = my_config.dir_deploy_models

BASE_DIR_SAVE = os.path.join(my_config.dir_heatmap, 'CAM')
if not os.path.exists(BASE_DIR_SAVE):
    os.makedirs(BASE_DIR_SAVE)


# Modify the model to output both original output and the output of last_conv_layer
# and return last_layer_weights (values after global average pooling)
def get_CNN_model(model_file1, last_layer=-1):
    if isinstance(model_file1, str):
        model = keras.models.load_model(model_file1, compile=False)
    else:
        model = model_file1

    # get the last conv layer before global average pooling
    for i in range(len(model.layers)-1, -1, -1):
        if isinstance(model.layers[i], Conv2D) or \
                isinstance(model.layers[i], Activation) or\
                isinstance(model.layers[i], SeparableConv2D) or\
                isinstance(model.layers[i], Concatenate):  #inception v3 Concatenate
            last_conv_layer = i
            break

    # get AMP layer weights
    last_layer_weights = model.layers[last_layer].get_weights()[0]

    # extract wanted output
    output_model = Model(inputs=model.input,
            outputs=(model.layers[last_conv_layer].output, model.layers[last_layer].output))

    return model, output_model, last_layer_weights


# this method is deployed to RPC service
# multi-labels input pred, multi-class don't need pred parameter(can caculate)
# input image preprocess, generate image tensor, and then call gen_cam
def server_cam(model_no, img_source, pred, cam_relu=True, preprocess=True,
               blend_original_image=True):

    image_size = dicts_models[model_no]['image_size']

    if preprocess:
        img_preprocess = my_preprocess.do_preprocess(img_source, crop_size=384)
        img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                                                                         image_shape=(image_size, image_size, 3))
    else:
        img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_source,
                                                                         image_shape=(image_size, image_size, 3))

    #region generate CAM
    model1 = dicts_models[model_no]['model_cam']
    all_amp_layer_weights = dicts_models[model_no]['all_amp_layer_weights']

    last_conv_output, pred_vec = model1.predict(img_input)

    # pred = np.argmax(pred_vec)  # get model's prediction class
    # Remove single-dimensional entries from the shape of an array.
    last_conv_output = np.squeeze(last_conv_output)

    # get AMP layer weights
    amp_layer_weights = all_amp_layer_weights[:, pred]  # dim: (2048,)

    # jijie add relu
    # 对于每一个类别C，每个特征图K的均值都有一个对应的w
    if cam_relu:
        amp_layer_weights = np.maximum(amp_layer_weights, 0)

    cam_small = np.dot(last_conv_output, amp_layer_weights)  # dim: 224 x 224
    cam_small = np.maximum(cam_small, 0)    # ReLU
    cam = cv2.resize(cam_small, (image_size, image_size))  # 14*14 224*224
    heatmap = cam / np.max(cam)  # heatmap:0-1
    #cam: 0-255
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    if blend_original_image:
        # Return to BGR [0..255] from the preprocessed image
        image_original = img_input[0, :]

        from LIBS.ImgPreprocess.my_image_norm import input_norm_reverse
        image_original = input_norm_reverse(image_original)
        image_original = image_original.astype(np.uint8)

        image_original -= np.min(image_original)
        image_original = np.minimum(image_original, 255)

        cam = np.float32(cam) + np.float32(image_original)
        cam = 255 * cam / np.max(cam)

    #endregion

    #region 将CAM保存到文件

    str_uuid = str(uuid.uuid1())
    filename_CAM = os.path.join(BASE_DIR_SAVE, str_uuid, 'CAM{}.jpg'.format(pred))

    if not os.path.exists(os.path.dirname(filename_CAM)):
        os.makedirs(os.path.dirname(filename_CAM))

    cv2.imwrite(filename_CAM, cam)

    # endregion

    return filename_CAM


# RPC Service, keras-vis, visualize_cam single thread too slow
def server_grad_cam(model_no, img_source, pred, preprocess=True,
                            blend_original_image=True):

    model = dicts_models[model_no]['model_original']

    image_size = dicts_models[model_no]['image_size']

    if preprocess:
        img_preprocess = my_preprocess.do_preprocess(img_source, crop_size=384)
        img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                                                                         image_shape=(image_size, image_size, 3))
    else:
        img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_source,
                                                                         image_shape=(image_size, image_size, 3))

    penultimate_layer = get_last_conv_layer_number(model)
    layer_idx = len(model.layers) - 1

    modifier = 'guided'  # [None, 'guided', 'relu']
    # too slow
    grads = visualize_cam(model, layer_idx, filter_indices=[pred],
                  seed_input=img_input, penultimate_layer_idx=penultimate_layer,
                  backprop_modifier=modifier)

    cam = cv2.applyColorMap(np.uint8(255 * grads), cv2.COLORMAP_JET)

    if blend_original_image:
        # Return to BGR [0..255] from the preprocessed image
        image_original = img_input[0, :]

        from LIBS.ImgPreprocess.my_image_norm import input_norm_reverse
        image_original = input_norm_reverse(image_original)
        image_original = image_original.astype(np.uint8)

        image_original -= np.min(image_original)
        image_original = np.minimum(image_original, 255)

        cam = np.float32(cam) + np.float32(image_original)
        cam = 255 * cam / np.max(cam)

    # 传过来的是web目录
    str_uuid = str(uuid.uuid1())
    filename_CAM = os.path.join(BASE_DIR_SAVE, str_uuid, 'Grad_CAM{}.jpg'.format(pred))

    if not os.path.exists(os.path.dirname(filename_CAM)):
        os.makedirs(os.path.dirname(filename_CAM))

    cv2.imwrite(filename_CAM, cam)


    return filename_CAM


def grad_cam_plus(model, img, pred=None, image_size=224):
    if pred is None:
        cls = np.argmax(model.predict(img))
    else:
        cls = pred

    layer_name = get_last_conv_layer_name(model) #very quick

    y_c = model.output[0, cls]

    conv_output = model.get_layer(layer_name).output

    grads = K.gradients(y_c, conv_output)[0]

    first = K.exp(y_c)*grads
    second = K.exp(y_c)*grads*grads
    third = K.exp(y_c)*grads*grads

    gradient_function = K.function([model.input], [y_c, first, second, third, conv_output, grads])
    y_c, conv_first_grad, conv_second_grad,conv_third_grad, conv_output, grads_val = gradient_function([img])
    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num/alpha_denom

    weights = np.maximum(conv_first_grad[0], 0.0)

    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0), axis=0)

    alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))

    deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
    #print deep_linearization_weights
    grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    # Passing through ReLU
    cam = np.maximum(grad_CAM_map, 0)
    cam = zoom(cam, image_size / cam.shape[0])
    cam = cam / np.max(cam) # scale 0 to 1.0

    return cam


def server_gradcam_plusplus(model_no, img_source, pred, preprocess=True,
                            blend_original_image=True):

    image_size = dicts_models[model_no]['image_size']

    if preprocess:
        img_preprocess = my_preprocess.do_preprocess(img_source, crop_size=384)
        img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                                                                         image_shape=(image_size, image_size, 3))
    else:
        img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_source,
                                                                         image_shape=(image_size, image_size, 3))

    gradcamplus = grad_cam_plus(dicts_models[model_no]['model_original'], img_input,
                                pred=pred, image_size=image_size)

    cam = cv2.applyColorMap(np.uint8(255 * gradcamplus), cv2.COLORMAP_JET)

    if blend_original_image:
        # Return to BGR [0..255] from the preprocessed image
        image_original = img_input[0, :]

        from LIBS.ImgPreprocess.my_image_norm import input_norm_reverse
        image_original = input_norm_reverse(image_original)
        image_original = image_original.astype(np.uint8)

        image_original -= np.min(image_original)
        image_original = np.minimum(image_original, 255)

        cam = np.float32(cam) + np.float32(image_original)
        cam = 255 * cam / np.max(cam)

    #region 将CAM保存到文件

    # 传过来的是web目录
    str_uuid = str(uuid.uuid1())
    filename_CAM = os.path.join(BASE_DIR_SAVE, str_uuid, 'GradCAM_PlusPlus{}.jpg'.format(pred))

    if not os.path.exists(os.path.dirname(filename_CAM)):
        os.makedirs(os.path.dirname(filename_CAM))

    cv2.imwrite(filename_CAM, cam)

    return filename_CAM

    # endregion


#region command parameters: class type no and port no


if len(sys.argv) != 3:  # sys.argv[0]  exe file itself
    reference_class = '0'  # bigclass multi_class
    # reference_class = '1'  # bigclass multi_label
    port = 23000
else:
    reference_class = str(sys.argv[1])
    port = int(sys.argv[2])

#endregion


#region load models

dicts_models = []

if reference_class == '0':
    dict_model1 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_19/split_pat_id/InceptionResNetV2-010-0.958.hdf5'),
                   'image_size': 299}
    dicts_models.append(dict_model1)

    dict_model2 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_19/split_pat_id/Xception-008-0.957.hdf5'),
                   'image_size': 299}
    dicts_models.append(dict_model2)

    dict_model3 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_19/split_pat_id/Inception_V3-006-0.955.hdf5'),
                   'image_size': 299}
    dicts_models.append(dict_model3)

if reference_class == '1':
    dict_model1 = {'model_file': os.path.join(DIR_MODELS, 'bigclasses_multilabels_new/bigclass_30_param0.11_2.4/InceptionResnetV2-traintop-001-0.919.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model1)

    dict_model2 = {'model_file': os.path.join(DIR_MODELS, 'bigclasses_multilabels_new/bigclass_30_param0.11_2.4/Xception-traintop-001-0.910.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model2)

    dict_model3 = {'model_file': os.path.join(DIR_MODELS, 'bigclasses_multilabels_new/bigclass_30_param0.11_2.4/InceptionV3-traintop-001-0.913.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model3)

for dict1 in dicts_models:
    print('prepare to load model:' + dict1['model_file'])
    original_model, output_model, all_amp_layer_weights1 = get_CNN_model(dict1['model_file'])

    if 'image_size' not in dict1:
        if original_model.input_shape[2] is not None:
            dict1['image_size'] = original_model.input_shape[2]
        else:
            dict1['image_size'] = 299

    dict1['model_original'] = original_model
    dict1['model_cam'] = output_model
    dict1['all_amp_layer_weights'] = all_amp_layer_weights1

    print('model load complete!')


#endregion

# region test code
if my_config.debug_mode:
    img_source = '/tmp1/img4.jpg'

    if os.path.exists(img_source):
        filename_CAM1 = server_cam(0, img_source, pred=1,
               cam_relu=True, preprocess=True, blend_original_image=True)
        # filename_CAM1 = server_cam(1, img_source, pred=1)
        # filename_CAM1 = server_grad_cam(0, img_source, pred=1)
        # filename_CAM1 = server_gradcam_plusplus(0, img_source, pred=1)

        print(filename_CAM1)

#endregion

#region start service

# server = SimpleXMLRPCServer(("localhost", port))
server = SimpleXMLRPCServer(("0.0.0.0", port))
print("Listening on port: ", str(port))
server.register_function(server_cam, "server_cam")
server.register_function(server_gradcam_plusplus, "server_gradcam_plusplus")
server.serve_forever()

#endregion