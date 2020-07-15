import os
import keras
import numpy as np

import LIBS.ImgPreprocess.my_image_helper
from LIBS.DataPreprocess import my_data, my_images_generator
from LIBS.ImgPreprocess.my_preprocess import do_preprocess

def do_predict_single(model1, img1, img_size=299, preproess=False, cuda_visible_devices=""):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    if isinstance(model1, str):
        model1 = keras.models.load_model(model1, compile=False)

    if preproess:
        img1 = do_preprocess.my_preprocess(img1, 512)

    # /= 255. etc.
    img_tensor = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img1,
                                                                      image_shape=(img_size, img_size, 3))

    prob1 = model1.predict_on_batch(img_tensor)
    prob1 = np.mean(prob1, axis=0)  # batch mean
    # pred1 = prob1.argmax(axis=-1)

    return prob1

#used by computing confusion matrix, etc.
def do_predict_batch(dicts_models, all_files,
                     batch_size_test=64, argmax=False, cuda_visible_devices="", gpu_num=1):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    if isinstance(all_files, str):  #csv file
        all_files, all_labels = my_data.get_images_labels(filename_csv_or_pd=all_files)

    assert len(all_files) > 0, 'No Data'


    prob_lists = []  #each element contain all probabilities  multiple batch, np.vstack
    preds_list = []

    for dict_model in dicts_models:
        if ('model' not in dict_model) or (dict_model['model'] is None):
            print('prepare load model:', dict_model['model_file'])
            model1 = keras.models.load_model(dict_model['model_file'], compile=False)
            print('load model:', dict_model['model_file'], ' complete')

            if gpu_num > 1:
                print('convert Multi-GPU model:', dict_model['model_file'])
                model1 = keras.utils.multi_gpu_model(model1, gpus=gpu_num)
                print('convert Multi-GPU model:', dict_model['model_file'], ' complete')

            dict_model['model'] = model1
        else:
            model1 = dict_model['model']  # avoid loading models multiple times

        if 'image_size' in dict_model:
            image_size = dict_model['image_size']
        elif model1.input_shape[2] is not None:
            image_size = model1.input_shape[2]
        else:
            image_size = 299


        j = 0 # batch
        for x in my_images_generator.my_Generator_test(all_files,
                                                       image_shape=(image_size, image_size, 3), batch_size=batch_size_test):

            probabilities = model1.predict_on_batch(x)

            if j == 0:    #'probs' not in locals().keys():
                probs = probabilities
            else:
                probs = np.vstack((probs, probabilities))

            j += 1
            print('batch:', j)

        prob_lists.append(probs)

        if argmax:
            y_preds = probs.argmax(axis=-1)
            y_preds = y_preds.tolist()
            preds_list.append(y_preds)

    sum_models_weights = 0
    for i, prob1 in enumerate(prob_lists):
        if 'model_weight' not in dicts_models[i]:
            model_weight = 1
        else:
            model_weight = dicts_models[i]['model_weight']

        if i == 0:
            prob_total = prob1 * model_weight
        else:
            prob_total += prob1 * model_weight

        sum_models_weights += model_weight


    prob_total /= sum_models_weights

    if argmax:
        y_pred_total = prob_total.argmax(axis=-1)
        return prob_total, y_pred_total, prob_lists, preds_list
    else:
        return prob_total, prob_lists



if __name__ ==  '__main__':

    model_file1 = '/home/ubuntu/dlp/deploy_models_2019/ocular_surface/3class/InceptionV3-021-train0.9985_val0.9996.hdf5'

    img_file1 = '/tmp2/fundus2.JPG'
    prob1 = do_predict_single(model_file1, img_file1, preproess=False, img_size=299)

    print('OK')


