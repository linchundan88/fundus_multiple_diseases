'''
  RPC Service
  Support both Softmax and Multi-class multi-label(multiple binary output)
'''

import os
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from xmlrpc.server import SimpleXMLRPCServer
import keras
from keras.utils.generic_utils import CustomObjectScope
import numpy as np
import LIBS.ImgPreprocess.my_image_helper
from LIBS.ImgPreprocess import my_preprocess
import my_config

DIR_MODELS = my_config.dir_deploy_models
list_threthold = my_config.LIST_THRETHOLD  #thresholds

def predict_softmax(img1, preproess=False):
    if preproess:
        img1 = my_preprocess.do_preprocess(img1, 512)

    prob_np = []
    prob = []
    pred = []  # only one class, multi_models multi-class multi-label has multiple classes

    for dict1 in dicts_models:
        #real time image aug
        img_tensor = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img1,
                                                                          image_shape=(dict1['image_size'], dict1['image_size'], 3))

        prob1 = dict1['model'].predict_on_batch(img_tensor)
        prob1 = np.mean(prob1, axis=0)  # batch mean, test time img aug
        pred1 = prob1.argmax(axis=-1)

        prob_np.append(prob1)  #  numpy  weight avg prob_total

        prob.append(prob1.tolist())    #担心XMLRPC numpy
        pred.append(int(pred1))   # numpy int64, int  XMLRPC

    list_weights = []  # the prediction weights of models
    for dict1 in dicts_models:
        list_weights.append(dict1['model_weight'])

    prob_total = np.average(prob_np, axis=0, weights=list_weights)
    pred_total = prob_total.argmax(axis=-1)

    prob_total = prob_total.tolist()  #RPC Service can not pass numpy variable
    pred_total = int(pred_total)     # 'numpy.int64'  XMLRPC

    # correct_model_no is used for choosing which model to generate CAM
    # on extreme condition: average softmax prediction class is not in every model's prediction class
    correct_model_no = 0
    for i, pred1 in enumerate(pred):
        if pred1 == pred_total:
            correct_model_no = i    #start from 0
            break

    return prob, pred, prob_total, pred_total, correct_model_no

# Multi-label is different from Multi-class:  pred correct_model_no are lists
def predict_sigmoids(img1, preproess=False, neglect_class0=False):
    if preproess:
        img1 = my_preprocess.do_preprocess(img1, 512)

    prob_np = []
    prob = []
    pred = []

    for dict1 in dicts_models:
        # real time image aug
        img_tensor = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img1,
                                                                          image_shape=(dict1['image_size'], dict1['image_size'], 3))

        prob1 = dict1['model'].predict_on_batch(img_tensor)
        prob1 = np.mean(prob1, axis=0) # batch mean, test time img aug

        if neglect_class0:
            prob1[0] = 0 #do not use class0, if all classes are negative ...

        pred_classes1 = []
        for i in range(NUM_CLASSES):
            if prob1[i] > list_threthold[i]:
                pred_classes1.append(1)
            else:
                pred_classes1.append(0)

        prob_np.append(prob1)
        prob.append(prob1.tolist())    #担心XMLRPC numpy
        pred.append(pred_classes1)

    list_weights = []  #the prediction weights of models
    for dict1 in dicts_models:
        list_weights.append(dict1['model_weight'])

    prob_total = np.average(prob_np, axis=0, weights=list_weights)

    pred_total = []
    for j in range(NUM_CLASSES):
        if prob_total[j] > list_threthold[j]:
            pred_total.append(1)
        else:
            pred_total.append(0)

    prob_total = prob_total.tolist()  #RPC parameter do not use Numpy

    correct_model_no = [] #采用那个模型热力图
    for i in range(NUM_CLASSES):
        for j in range(len(dicts_models)):
            # model j, class i
            if pred[j][i] == pred_total[i]:
                correct_model_no.append(j)
                break

    return prob, pred, prob_total, pred_total, correct_model_no


#command parameters: predict class type no and port number
if len(sys.argv) == 3:  # sys.argv[0]  exe file itself
    reference_class = str(sys.argv[1])
    port = int(sys.argv[2])
else:
    reference_class = '0'  #bigclass
    port = 20000

    reference_class = '1'  # DR2-3
    port = 20010


#region define models
dicts_models = []

#img_position   macula center:0,optic_disc center:1, other
if reference_class == '-5':
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'img_position/MobileNetV2-017-train0.8141_val0.828.hdf5'),
             'image_size': 224, 'model_weight': 1}
    dicts_models.append(dict1)

    dict1 = {'model_file': os.path.join(DIR_MODELS, 'img_position/NasnetMobile-013-0.834.hdf5'),
             'image_size': 224, 'model_weight': 1}
    dicts_models.append(dict1)

#left right eye
if reference_class == '-4':
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'LeftRight/MobileNetV2-005-0.997.hdf5'),
              'image_size': 224, 'model_weight': 1}
    dicts_models.append(dict1)

    # dict1 = {'model_file': os.path.join(DIR_MODELS, 'LeftRight/NasnetMobile-007-0.991.hdf5'),
    #          'model_weight': 1, 'image_size': 224}
    # models.append(dict1)

#gradable
if reference_class == '-3':
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'Gradable/MobileNetV2-005-0.946.hdf5'),
              'image_size': 224, 'model_weight': 1}
    dicts_models.append(dict1)

    # dict2 = {'model_file': os.path.join(DIR_MODELS,DIR_MODELS + 'Gradable/NasnetMobile-006-0.945.hdf5'),
    #          'image_size': 224, 'model_weight': 1}
    # models.append(dict2)

#ocular surface
if reference_class == '-1':
    # dict1 = {'model_file': os.path.join(DIR_MODELS, 'ocular_surface/MobilenetV2-006-1.000.hdf5'),
    #           'image_size': 224, 'model_weight': 1}
    # dicts_models.append(dict1)
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'ocular_surface/MobilenetV2-006-1.000_add_others_borders.hdf5'),
              'image_size': 224, 'model_weight': 1}
    dicts_models.append(dict1)


#big_classes multi-class
if reference_class == '0':
    dict_model1 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_19/split_pat_id/InceptionResNetV2-010-0.958.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model1)

    dict_model2 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_19/split_pat_id/Xception-008-0.957.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model2)
    dict_model3 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_19/split_pat_id/Inception_V3-006-0.955.hdf5'),
                   'image_size': 299, 'model_weight': 0.8}
    dicts_models.append(dict_model3)

#big_classes multi-label
if reference_class == '0_10':
    dict_model1 = {'model_file': os.path.join(DIR_MODELS, 'bigclasses_multilabels_new/bigclass_30_param0.11_2.4/InceptionResnetV2-traintop-001-0.919.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model1)

    dict_model2 = {'model_file': os.path.join(DIR_MODELS, 'bigclasses_multilabels_new/bigclass_30_param0.11_2.4/Xception-traintop-001-0.910.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model2)
    dict_model3 = {'model_file': os.path.join(DIR_MODELS, 'bigclasses_multilabels_new/bigclass_30_param0.11_2.4/InceptionV3-traintop-001-0.913.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model3)


#subclass0 分成  Normal,Tigroid fundus
if reference_class == '0_1':
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'models_subclass_2019_4_26/DLP_SubClass0.1/InceptionResNetV2-023-0.958.hdf5'),
             'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict1)
    dict2 = {'model_file': os.path.join(DIR_MODELS, 'models_subclass_2019_4_26/DLP_SubClass0.1/Xception-007-0.955.hdf5'),
             'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict2)

# subclass0 分成  Normal, big optic cup
if reference_class == '0_2':
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'models_subclass_2019_4_26/DLP_SubClass0.2/InceptionResNetV2-013-0.870.hdf5'),
             'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict1)
    dict2 = {'model_file': os.path.join(DIR_MODELS, 'models_subclass_2019_4_26/DLP_SubClass0.2/Xception-019-0.873.hdf5'),
             'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict2)

# subclass0 分成  Normal, DR1
if reference_class == '0_3':
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'SubClass0_3/ResNet448-005-0.841.hdf5'),
             'image_size': 448, 'model_weight': 1, 'image_size': 448}
    dicts_models.append(dict1)


#subclass1  DR2+分成  DR2 and DR3
if reference_class == '1':
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'models_subclass_2019_4_26/DLP_SubClass1/InceptionResNetV2-017-0.928.hdf5'),
              'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict1)
    dict2 = {'model_file': os.path.join(DIR_MODELS, 'models_subclass_2019_4_26/DLP_SubClass1/Xception-017-0.930.hdf5'),
              'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict2)

#subclass2  RVO 分成  BRVO and CRVO
if reference_class == '2':
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'models_subclass_2019_4_26/DLP_SubClass2/InceptionResNetV2-026-0.997.hdf5'),
              'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict1)
    dict2 = {'model_file': os.path.join(DIR_MODELS, 'models_subclass_2019_4_26/DLP_SubClass2/Xception-036-0.995.hdf5'),
              'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict2)

#subclass5  Posterior exudative DR-CSCR and  Posterior exudative RD-VKH
if reference_class == '5':
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'models_subclass_2019_4_26/DLP_SubClass5/InceptionResNetV2-035-0.953.hdf5'),
              'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict1)
    dict2 = {'model_file': os.path.join(DIR_MODELS, 'models_subclass_2019_4_26/DLP_SubClass5/Xception-031-0.948.hdf5'),
              'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict2)

#subclass10 分成  Probable glaucoma	C/D > 0.7 and Optic atrophy	pale with normal cupping
if reference_class == '10':
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'SubClass10/Resnet112-024-0.953.hdf5'),
            'image_size': 112, 'model_weight': 1}
    dicts_models.append(dict1)
    dict2 = {'model_file': os.path.join(DIR_MODELS, 'SubClass10/Resnet112-036-0.948.hdf5'),
              'image_size': 112, 'model_weight': 1}
    dicts_models.append(dict2)

#subclass15 分成  RP and Bietti crystalline dystrophy
if reference_class == '15':
    # dict1 = {'model_file': os.path.join(DIR_MODELS, 'SubClass15/InceptionResNetV2-012-train0.9603_val0.9801.hdf5'),
    #          'image_size': 299, 'model_weight': 1}
    # dicts_models.append(dict1)
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'models_subclass_2019_4_26/DLP_SubClass15/InceptionResNetV2-041-0.940.hdf5'),
             'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict1)
    dict2 = {'model_file': os.path.join(DIR_MODELS, 'models_subclass_2019_4_26/DLP_SubClass15/Xception-060-0.936.hdf5'),
             'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict2)

#subclass29 分成  Blur fundus and Blur fundus - probably PDR
if reference_class == '29':
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'models_subclass_2019_4_26/DLP_SubClass29/InceptionResNetV2-016-0.980.hdf5'),
             'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict1)
    dict2 = {'model_file': os.path.join(DIR_MODELS, 'models_subclass_2019_4_26/DLP_SubClass29/Xception-020-0.974.hdf5'),
             'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict2)

#Neovascularization(Deprecated)
if reference_class == '60':
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'Neovascularization(Deprecated)/InceptionResNetV2-006-0.872.hdf5'),
             'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict1)
    dict2 = {'model_file': os.path.join(DIR_MODELS, 'Neovascularization(Deprecated)/Xception-007-0.871.hdf5'),
             'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict2)

#endregion

#load models
for dict1 in dicts_models:
    model_file = dict1['model_file']
    print('%s load start!' % (model_file))
    # ValueError: Unknown activation function:relu6  MobileNet V2
    with CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
        dict1['model'] = keras.models.load_model(model_file, compile=False)

    if 'image_size' not in dict1:
        if dict1['model'].input_shape[2] is not None:
            dict1['image_size'] = dict1['model'].input_shape[2]
        else:
            dict1['image_size'] = 299

    print('%s load complte!' % (model_file))

NUM_CLASSES = dicts_models[0]['model'].output_shape[1]


#region test mode
if my_config.debug_mode:
    img_source = '/tmp1/brvo1.jpg'

    if os.path.exists(img_source):
        # img1 = my_preprocess.my_preprocess(img_source, 512)
        img1 = my_preprocess.do_preprocess(img_source, 512)

        # prob_list, pred_list, prob_total, pred_total, correct_model_no = predict_softmax(img1)

        prob_list, pred_list, prob_total, pred_total, correct_model_no = predict_sigmoids(img1)
        print(prob_total)
    else:
        print('file:', img_source, ' does not exist.')
#endregion


server = SimpleXMLRPCServer(("localhost", port))
print("Listening on port: ", str(port))
server.register_function(predict_softmax, "predict_softmax")
server.register_function(predict_sigmoids, "predict_sigmoids")
server.serve_forever()

