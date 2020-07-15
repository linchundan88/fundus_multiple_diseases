'''
multi_labels
根据CSV计算 （CSV有单个标注） compute_multi_model_multi_files
根据目录，目录没有标注  compute_multi_files_dir()

单个模型预测单个文件 compute_one_model_one_file()
'''

import os
import keras
import numpy as np
import pickle
import csv

import LIBS.ImgPreprocess.my_image_helper
from LIBS.DataPreprocess import my_data, my_images_generator


def do_predict_dir(dicts_models, all_files, num_class=30, gpu_num=1):
    if isinstance(all_files, str):  # csv file
        all_files, all_labels = my_data.get_images_labels(filename_csv_or_pd=all_files)

    batch_size_test = 64  # 虽然可以128，但是my_images_generator 将list转换np 耗费CPU

    # 不是写死几个模型，提高灵活性
    prob_list = []  ##每一个模型的概率

    for i, model in enumerate(dicts_models):
        if ('model' not in model1) (model['model'] is None):
            print('prepare load model:', model['model_file'])
            model1 = keras.models.load_model(model['model_file'], compile=False)
            print('load model:', model['model_file'], ' complete')

            if gpu_num > 1:
                print('convert Multi-GPU model:', model['model_file'])
                model1 = keras.utils.multi_gpu_model(model1, gpus=gpu_num)
                print('convert Multi-GPU model:', model['model_file'], ' complete')
        else:
            model1 = model['model']

        prob_list.append(np.empty((0, num_class)))

        j = 0  # 样本数可能很多，每计算100个，一个数出提示

        image_size = model['image_size']
        for x in my_images_generator.my_Generator_test(all_files,
                   image_shape=(image_size, image_size, 3),
                   batch_size=batch_size_test):
            probabilities = model1.predict_on_batch(x)
            if prob_list[i].size == 0:
                prob_list[i] = probabilities
            else:
                prob_list[i] = np.vstack((prob_list[i], probabilities))

            j += 1
            print('batch:', j)


    sum_models_weights = 0
    for i, prob1 in enumerate(prob_list):
        if 'model_weight' not in dicts_models[i]:
            model_weight = 1
        else:
            model_weight = dicts_models[i]['model_weight']

        sum_models_weights += model_weight

        if i == 0:
            prob_total = prob1 * model_weight
        else:
            prob_total += prob1 * model_weight

    prob_total /= sum_models_weights

    return prob_total

# 读取csv时候有labels
def compute_probs(models, file_csv,
                  batch_size=32, cuda_visible_devices='0'):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    all_files, all_labels = my_data.get_images_labels(file_csv)

    #region using every model to predict every image, generate list_probs
    list_probs = []
    for i, model in enumerate(models):
        if ('model' not in model) or (model['model'] is None):
            print('prepare load model:', model['model_file'])
            model1 = keras.models.load_model(model['model_file'], compile=False)
            print('load model:', model['model_file'], ' complete')
        else:
            model1 = model['model']

        batch_no = 0  # batch number
        tmp_probs = None

        image_size = model['image_size']

        for x in my_images_generator.my_Generator_test(all_files,  image_shape=(image_size, image_size, 3),
                               batch_size=batch_size):
            probabilities = model1.predict_on_batch(x)
            if tmp_probs is None:
                tmp_probs = probabilities
            else:
                tmp_probs = np.vstack((tmp_probs, probabilities))

            batch_no += 1
            print('batch:', batch_no)

        list_probs.append(tmp_probs)
    #endregion

    #region weighted average every model's result, generate list_probs_weighted
    list_probs_weighted = None
    model_weight_sum = 0

    for i, probs1 in enumerate(list_probs):
        model_weight = models[i]['model_weight']
        model_weight_sum += model_weight
        if i == 0:
            list_probs_weighted = probs1 * model_weight
        else:
            list_probs_weighted = list_probs_weighted + probs1 * model_weight

    list_probs_weighted = list_probs_weighted / model_weight_sum
    #endregion


    # import keras.backend as K
    # K.clear_session()  # release GPU memory

    return list_probs, list_probs_weighted

def save_multi_label_csv(file_csv, list_probs, csv_results, list_threshold):

    num_classes = len(list_threshold)
    all_files, all_labels = my_data.get_images_labels(file_csv)

    if csv_results != '':
        if os.path.exists(csv_results):
            os.remove(csv_results)

        with open(csv_results, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')

            list_title = ['images', 'labels', 'bigclasses']
            for i in range(num_classes):
                list_title.append('class' + str(i))

            csv_writer.writerow(list_title)

            for i in range(len(all_files)):
                file, label = all_files[i], all_labels[i]
                probs1 = list_probs[i]

                pred_classes = '_'
                for class_i in range(num_classes):
                    if probs1[class_i] > list_threshold[class_i]:
                        pred_classes = pred_classes + str(class_i) + str('_')

                list_row = [file, label, pred_classes]

                for k in range(num_classes):
                    list_row.append(round(probs1[k], 3))

                csv_writer.writerow(list_row)

        print('csv file ok')


# 读取csv时候没labels, 预处理之后的目录
def compute_multi_files_dir(models, dir_base, list_threshold=None, csv_results=None, dump_pkl='',
                            batch_size=32, cuda_visible_devices='0'):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


    #根据目录生成 all_files 一个个图像文件
    all_files = []
    for dir_path, subpaths, files in os.walk(dir_base, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)

            filename, file_extension = os.path.splitext(img_file_source)
            if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                continue

            all_files.append(img_file_source)


    #region 每一个模型对所有文件进行预测
    results_list = []
    for i, model in enumerate(models):
        if ('model' not in model) or (model['model'] is None):
            print('prepare load model:', model['model_file'])
            model1 = keras.models.load_model(model['model_file'], compile=False)
            print('load model:', model['model_file'], ' complete')
        else:
            model1 = model['model']

        if i == 0:
            if list_threshold is None:
                num_classes = model1.layers[-1].output.shape[-1]
                list_threshold = [0.5 for _ in range(30)]
            else:
                num_classes = len(list_threshold)

        j = 0  # 样本数可能很多，每计算100个，一个数出提示
        results = None
        model = keras.models.load_model(model1['model_file'], compile=False)
        image_size = model['image_size']

        for x in my_images_generator.my_Generator_test(all_files,
                       image_shape=(image_size, image_size, 3), batch_size=batch_size):
            probabilities = model.predict_on_batch(x)
            if results is None:
                results = probabilities
            else:
                results = np.vstack((results, probabilities))

            j += 1
            print('batch:', j)

        results_list.append(results)
    #endregion

    #region 合并多个模型的计算结果
    result_total = None
    for i, result1 in enumerate(results_list):
        if i == 0:
            result_total = result1
        else:
            result_total = result_total + result1
    result_total = result_total / (len(results_list))
    #endregion

    if csv_results != '':
        if os.path.exists(csv_results):
            os.remove(csv_results)

        with open(csv_results, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
            # csv_writer.writerow(['images', 'bigclasses', 'class0',
            #                      'class1', 'class2', 'class3', 'class4', 'class5',
            #                      'class6', 'class7', 'class8', 'class9', 'class10',
            #                      'class11', 'class12', 'class13', 'class14', 'class15',
            #                      'class16', 'class17', 'class18', 'class19', 'class20',
            #                      'class21', 'class22', 'class23', 'class24', 'class25',
            #                      'class26', 'class27', 'class28'
            #                      ])

            list_title = ['images', 'bigclasses']
            for i in range(num_classes):
                list_title.append('class' + str(i))

            csv_writer.writerow(list_title)

            for i in range(len(all_files)):
                file = all_files[i]

                prob1 = result_total[i]

                s_bigclass = '_'

                for j in range(num_classes):
                    if prob1[j] > list_threshold[j]:
                        s_bigclass = s_bigclass + str(j) + str('_')

                # csv_writer.writerow([file, s_bigclass, round(prob1[0], 3),
                #      round(prob1[1], 3), round(prob1[2], 3), round(prob1[3], 3), round(prob1[4], 3), round(prob1[5], 3),
                #      round(prob1[6], 3), round(prob1[7], 3), round(prob1[8], 3), round(prob1[9], 3), round(prob1[10], 3),
                #      round(prob1[11], 3), round(prob1[12], 3), round(prob1[13], 3), round(prob1[14], 3), round(prob1[15], 3),
                #      round(prob1[16], 3), round(prob1[17], 3), round(prob1[18], 3), round(prob1[19], 3), round(prob1[20], 3),
                #      round(prob1[21], 3), round(prob1[22], 3), round(prob1[23], 3), round(prob1[24], 3), round(prob1[25], 3),
                #      round(prob1[26], 3), round(prob1[27], 3), round(prob1[28], 3)
                # ])

                list_row = [file, s_bigclass]
                for i in range(num_classes):
                    list_row.append(round(prob1[i], 3))

                csv_writer.writerow(list_row)

        print('csv file ok')

    if dump_pkl != '':
        with open(dump_pkl, 'wb') as file:
            pickle.dump(result_total, file)
            print('dump complete.')

    return results_list, result_total


def compute_one_model_one_file(model1, img_source,
                               image_shape=(299, 299), list_threshold=None,
                               cuda_visible_devices='0'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    if isinstance(model1, str):
        # 参数传递过来的事模型文件
        model1 = keras.models.load_model(model1, compile=False)

    if list_threshold is None:
        num_classes = model1.layers[-1].output.shape[-1]
        list_threshold = [0.5 for _ in range(30)]
    else:
        num_classes = len(list_threshold)

    # 不同模型 输入分辨率不同
    x = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_source, image_shape=image_shape)

    probabilities = model1.predict_on_batch(x)

    list_class = [] #所属病种类别
    #  正常大类忽略， 没有其他疾病就是正常（其他病种概率都小）
    for j in range(1, num_classes):
        if probabilities[0][j] > list_threshold[j]:
            list_class.append(j)

    # # test_result[0][1:]   28 个 float32
    return probabilities[0], list_class

