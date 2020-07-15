'''
Support both Softmax and Multi-class multi-label(multiple binary output)
'''

import os

import LIBS.ImgPreprocess.my_image_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""  #force Keras to use CPU

import keras
from keras.utils.generic_utils import CustomObjectScope
import sys
sys.path.append("./")
sys.path.append("../")

from LIBS.ImgPreprocess import my_preprocess
from LIBS.DataPreprocess import my_images_generator
import my_config
dir_models = my_config.dir_deploy_models

#region Predict Softmax

# 调用一个模型进行inference，输出 概率和类别
def predict_one_model_softmax(dict1, image_tensor, argmax=True):
    probabilities = dict1['model'].predict_on_batch(image_tensor)

    if argmax:
        #输出预测的类  int类型
        y_pred1 = probabilities.argmax(axis=-1)
        return probabilities[0], int(y_pred1[0])
    else:
        #输出概率分布   numpy类型  无法RPC传递
        # return probabilities.tolist()  tolist() 之后 多个概率无法相加
        return probabilities[0]

# 该方法对外发布  每一个诊断类别使用多个模型 inference，输出 概率
def predict_softmax(img1, preproess=False):
    if preproess:
        img1 = my_preprocess.do_preprocess(img1, 512)

    prob_list_np = []
    prob_list = []
    pred_list = []  # only one class, multi-class multi-label has multiple classes

    # 调用每一个模型预测概率和所属类
    list_weights = []
    for dict1 in models:
        list_weights.append(dict1['model_weight'])

        x = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img1, image_shape=(dict1['image_size'], dict1['image_size']))

        prob1, pred1 = predict_one_model_softmax(dict1, x, argmax=True)

        prob_list_np.append(prob1)
        prob_list.append(prob1.tolist())    #担心XMLRPC numpy
        pred_list.append(pred1)

    # 计算多模型合并的该率
    for i, prob1 in enumerate(prob_list_np):
        if i == 0:
            prob_total = prob1 * list_weights[i]
        else:
            prob_total = prob_total + prob1 * list_weights[i]

    prob_total = prob_total / len(prob_list_np)

    # 多模型平均的预测类
    pred_total = prob_total.argmax(axis=-1)
    #  'numpy.int64' 转换为 标准类型，担心XMLRPC 传递
    pred_total = int(pred_total)

    correct_model_no = 0   # 每一个模型预测的类别，热力图 选择哪个模型需要
    for i, pred1 in enumerate(pred_list):
        if pred1 == pred_total:
            correct_model_no = i    #start from 0
            break

    prob_total = prob_total.tolist()

    return prob_list, pred_list, prob_total, pred_total, correct_model_no

#endregion

#region Predict Multi Labels

# 该方法对外发布 Multi-class Multi-label  每一个诊断类别使用多个模型 inference，输出 概率
NUM_CLASSES = 29
# 疾病大类二分类的阈值
list_threthold = [0.5 for i in range(NUM_CLASSES)]
def predict_one_model_multilabels(dict1, img1):
    probabilities = dict1['model'].predict_on_batch(img1)

    pred_classes = []
    for j in range(NUM_CLASSES):
        if probabilities[0][j] > list_threthold[j]:
            pred_classes.append(1)
        else:
            pred_classes.append(0)

    return probabilities[0], pred_classes

def predict_multi_labels(img1, preproess=False):
    if preproess:
        img1 = my_preprocess.do_preprocess(img1, 512)

    prob_list_np = []
    prob_list = []
    pred_list_classes = []

    # 调用每一个模型预测概率和所属类
    list_weights = []
    for dict1 in models:
        list_weights.append(dict1['model_weight'])

        x = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img1, image_shape=(dict1['image_size'], dict1['image_size']))

        prob1, pred_classes1 = predict_one_model_multilabels(dict1, x)

        prob_list_np.append(prob1)
        prob_list.append(prob1.tolist())    #担心XMLRPC numpy
        pred_list_classes.append(pred_classes1)

    # 计算多模型合并的该率
    for i, prob1 in enumerate(prob_list_np):
        if i == 0:
            prob_total = prob1 * list_weights[i]
        else:
            prob_total = prob_total + prob1 * list_weights[i]

    prob_total = prob_total / len(prob_list_np)

    pred_list_classes_total = []
    for j in range(NUM_CLASSES):
        if prob_total[j] > list_threthold[j]:
            pred_list_classes_total.append(1)
        else:
            pred_list_classes_total.append(0)

    prob_total = prob_total.tolist()

    correct_model_no = [] #采用那个模型热力图
    for i in range(NUM_CLASSES):
        for j in range(len(models)):
            # 第j个模型，第个类别
            if pred_list_classes[j][i] == pred_list_classes_total[i]:
                correct_model_no.append(j)
                break

    return prob_list, pred_list_classes, prob_total, pred_list_classes_total, correct_model_no

#endregion


#region 定义模型文件组
models = []

dir_models ='/home/ubuntu/dlp/deploy_models_new/'

dict1 = {'model_file': dir_models + 'bigclasses_multilabels/class_weights5_0.2_0.75/Multi_label_my_Xception-016-train0.9541_val0.933.hdf5',
         'image_size': 299, 'model_weight': 1}

models.append(dict1)

dict1 = {'model_file': dir_models + 'bigclasses_multilabels/class_weights5_0.2_0.75/Multi_label_InceptionResNetV2-015-train0.9629_val0.935.hdf5',
         'image_size': 299, 'model_weight': 1}

models.append(dict1)


for dict1 in models:
    model_file = dict1['model_file']
    print('%s load start!' % (model_file))
    # ValueError: Unknown activation function:relu6  MobileNet V2
    with CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
        dict1['model'] = keras.models.load_model(model_file, compile=False)
    print('%s load complte!' % (model_file))


#endregion

#加载模型文件组

img_source = '/tmp1/img1.jpg'
img_source = '/tmp1/img5.jpg'  #class23 Chorioretinal atrophy high prob
img_source = '/tmp1/img6.jpg'  #class23 very high prob

if os.path.exists(img_source):
    img1 = my_preprocess.do_preprocess(img_source, 512)

    # prob_list, pred_list, prob_total, pred_total, correct_model_no = predict_softmax(img1)

    prob_list, pred_list, prob_total, pred_total, correct_model_no = predict_multi_labels(img1)

    print(prob_total)
else:
    print('file:', img_source, ' does not exist.')


