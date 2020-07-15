'''
  generating a results csv
  computer confusion matrix for every class
  export FN, FP files
'''

import os, sys
from LIBS.DataValidation.my_multi_labels import compute_probs, save_multi_label_csv
from LIBS.DLP.my_multilabel_depedence import postprocess_exclusion, postprocess_all_negative
import pickle

CUDA_VISIBLE_DEVICES = '2'

GEN_RESULTS = True
COMPUTE_EXCLUSION_PROBS = True #mutual exclusive:4 vs 5, 4 vs 7, 5 vs 6, 6 vs 8, 10 vs 12 vs 13 vs 14
COMPUTE_ALLNEGATIVE = True #all negative ,prob of non-referable is low.

COMPUTE_CF = True
COMPUTE_EXCLUSION_NUM = True

EXPORT_ERROR_FILES = False
EXP_FP = True
EXP_FN = True

dir_preprocess = '/home/ubuntu/multi_labels_2919_1_15/preprocess384/'
dir_original = '/media/ubuntu/data1/multi_labels_2919_1_15/'

result_prefix ='2019_8_29_0.11_2.4_0.7_e_n_'
dir_export = '/tmp3/' + result_prefix

NUM_CLASSES = 30
LIST_THRESHOLD = [0.5 for _ in range(NUM_CLASSES)]

dicts_models = []

base_model_dir = '/home/ubuntu/dlp/deploy_models_2019/bigclasses_multilabels_new/bigclass_30_param0.11_2.4'
model1 = {'model_file': os.path.join(base_model_dir, 'InceptionResnetV2-traintop-001-0.919.hdf5'),
          'image_size': 299, 'model_weight': 1}
dicts_models.append(model1)
model2 = {'model_file': os.path.join(base_model_dir, 'Xception-traintop-001-0.910.hdf5'),
          'image_size': 299, 'model_weight': 1}
dicts_models.append(model2)
model3 = {'model_file': os.path.join(base_model_dir, 'InceptionV3-traintop-001-0.913.hdf5'),
          'image_size': 299, 'model_weight': 1}
dicts_models.append(model3)


if GEN_RESULTS:
    file_csv_train = os.path.abspath(os.path.join(sys.path[0], "..",
                    'datafiles',  'DLP_patient_based_split_train.csv'))
    result_csv_train = result_prefix+'_results_train.csv'
    list_probs, list_probs_weighted = compute_probs(dicts_models, file_csv=file_csv_train,
                                                    batch_size=64,
                                                    cuda_visible_devices=CUDA_VISIBLE_DEVICES)
    dump_pkl = result_prefix + 'multi_label_train.pkl'
    with open(dump_pkl, 'wb') as file:
        pickle.dump(list_probs_weighted, file)

    if COMPUTE_EXCLUSION_PROBS:
        list_probs_weighted = postprocess_exclusion(list_probs_weighted, list_threshold=LIST_THRESHOLD)
        dump_pkl = result_prefix + 'multi_label_train_with_exclusion.pkl'
        with open(dump_pkl, 'wb') as file:
            pickle.dump(list_probs_weighted, file)

    if COMPUTE_ALLNEGATIVE:
        list_probs_weighted = postprocess_all_negative(list_probs_weighted, list_threshold=LIST_THRESHOLD)

    save_multi_label_csv(file_csv=file_csv_train, list_probs=list_probs_weighted,
                         csv_results=result_csv_train, list_threshold=LIST_THRESHOLD)

    file_csv_valid = os.path.abspath(os.path.join(sys.path[0], "..",
                    'datafiles',  'DLP_patient_based_split_valid.csv'))
    result_csv_valid = result_prefix+'_results_valid.csv'
    list_probs, list_probs_weighted = compute_probs(dicts_models, file_csv=file_csv_valid,
                                                    batch_size=64, cuda_visible_devices=CUDA_VISIBLE_DEVICES)
    dump_pkl = result_prefix + 'multi_label_valid.pkl'
    with open(dump_pkl, 'wb') as file:
        pickle.dump(list_probs_weighted, file)

    if COMPUTE_EXCLUSION_PROBS:
        list_probs_weighted = postprocess_exclusion(list_probs_weighted, list_threshold=LIST_THRESHOLD)
        dump_pkl = result_prefix + 'multi_label_valid.pkl'
        with open(dump_pkl, 'wb') as file:
            pickle.dump(list_probs_weighted, file)

    if COMPUTE_ALLNEGATIVE:
        list_probs_weighted = postprocess_all_negative(list_probs_weighted, list_threshold=LIST_THRESHOLD)

    save_multi_label_csv(file_csv=file_csv_valid, list_probs=list_probs_weighted,
                         csv_results=result_csv_valid, list_threshold=LIST_THRESHOLD)

#region compute confusion matrix
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import pandas as pd

def convert_prob_to_label(list_prob, class_no, list_threshold):
    list1 = []
    for prob1 in list_prob:
        if prob1 > list_threshold[class_no]:
            list1.append(1)
        else:
            list1.append(0)

    return list1

def encoding_labels(list_labels, class_no):
    list_results = []
    for labels in list_labels:  #labels: '4_25_29_'
        list1 = str(labels).split('_')

        match_positive = False
        for label in list1:
            if label == '':
                continue
            if label == str(class_no):
                match_positive = True
                list_results.append(1)
                break
        if not match_positive:
            list_results.append(0)

    return list_results

if COMPUTE_CF:
    for filename_csv in [result_prefix+'_results_train.csv',
                         result_prefix+'_results_valid.csv']:
        print('computer confusion matrix:', filename_csv + '\n')

        df = pd.read_csv(filename_csv, delimiter=',')
        gt_label = df['labels'].tolist()

        for i in range(0, NUM_CLASSES):
            y_true = encoding_labels(gt_label, i)
            list_prob = df['class' + str(i)].tolist()
            y_pred = convert_prob_to_label(list_prob, i, list_threshold=LIST_THRESHOLD)
            cf1 = sk_confusion_matrix(y_true, y_pred)

            print(str(i) + ':')
            print(cf1)
            # sk_roc_auc_score(y_true, y_score)

#endregion

if COMPUTE_EXCLUSION_NUM:
    num_exclusion = 0

    for filename_csv in [result_prefix+'results_train.csv',
                         result_prefix+'results_valid.csv']:
        print('computer confusion matrix:', filename_csv + '\n')

        df = pd.read_csv(filename_csv, delimiter=',')
        for _, row in df.iterrows():
            image_file = row["images"]
            label1 = str(row["class1"])
            label2 = str(row["class2"])
            label4 = str(row["class4"])
            label5 = str(row["class5"])
            label6 = str(row["class6"])
            label7 = str(row["class7"])
            label8 = str(row["class8"])
            label10 = str(row["class10"])
            label12 = str(row["class12"])
            label13 = str(row["class13"])
            label14 = str(row["class14"])

            if label1 == '1' and label2 == '1':
                num_exclusion += 1
                continue
            if label4 == '1' and label5 == '1':
                num_exclusion += 1
                continue
            if label4 == '1' and label7 == '1':
                num_exclusion += 1
                continue
            if label5 == '1' and label6 == '1':
                num_exclusion += 1
                continue
            if label6 == '1' and label8 == '1':
                num_exclusion += 1
                continue

            if label10 == '1' and label12 == '1':
                num_exclusion += 1
                continue
            if label10 == '1' and label13 == '1':
                num_exclusion += 1
                continue
            if label10 == '1' and label14 == '1':
                num_exclusion += 1
                continue
            if label12 == '1' and label13 == '1':
                num_exclusion += 1
                continue
            if label12 == '1' and label14 == '1':
                num_exclusion += 1
                continue
            if label13 == '1' and label14 == '1':
                num_exclusion += 1
                continue

            # 4 vs 5， 4 vs 7， 5 vs 6， 6 vs 8， 10VS12VS13VS14

        print(num_exclusion)

if EXPORT_ERROR_FILES:
    import shutil
    from LIBS.DataPreprocess.my_compute_digest import CalcSha1

    for filename_csv in [result_prefix+'_results_train.csv',
                         result_prefix+'_results_valid.csv']:
        df = pd.read_csv(filename_csv, delimiter=',')

        if filename_csv == result_prefix+'results_train.csv':
            dest_dir_fp = os.path.join(dir_export, 'add_labels/train')
            dest_dir_fn = os.path.join(dir_export, 'del_labels/train')
        elif filename_csv == result_prefix+'results_valid.csv':
            dest_dir_fp = os.path.join(dir_export, 'add_labels/valid')
            dest_dir_fn = os.path.join(dir_export, 'del_labels/valid')
        else:
            raise Exception('filename_csv error')

        count = len(df.index)
        for i in range(count):
            img_file_preprocess = df.at[i, 'images']
            img_file_original = img_file_preprocess.replace(dir_preprocess, dir_original)

            labels_gt = df.at[i, 'labels']
            labels_gt = str(labels_gt)
            labels_predict = df.at[i, 'bigclasses']
            labels_predict = str(labels_predict)

            if EXP_FP:
                list_gt_labels = []
                for label_gt in labels_gt.split('_'):
                    if label_gt != '':
                        list_gt_labels.append(label_gt)

                for label_pred in labels_predict.split('_'):
                    if label_pred == '':
                        continue

                    if label_pred not in list_gt_labels:
                        predict_prob = df.at[i, 'class' + label_pred]
                        predict_prob = round(float(predict_prob), 2)

                        if predict_prob < 0.6:
                            continue

                        _, extname = os.path.splitext(img_file_original)
                        sha1 = CalcSha1(img_file_original)

                        img_file_dest = os.path.join(dest_dir_fp, label_pred,
                                                     'prob' + str(predict_prob) + '_gt' + labels_gt +
                                                     '__pred' + labels_predict + '_sha1' + sha1 + extname)

                        assert os.path.exists(img_file_original), \
                            img_file_original + 'not exists!'

                        if not os.path.exists(os.path.dirname(img_file_dest)):
                            os.makedirs(os.path.dirname(img_file_dest))

                        print(img_file_dest)
                        shutil.copy(img_file_original, img_file_dest)

            if EXP_FN:
                list_predict_labels = []
                for label_predict in labels_predict.split('_'):
                    if label_predict != '':
                        list_predict_labels.append(label_predict)

                for label_gt in labels_gt.split('_'):
                    if label_gt == '':
                        continue

                    if label_gt not in list_predict_labels:
                        predict_prob = df.at[i, 'class' + label_gt]
                        predict_prob = round(float(predict_prob), 2)

                        if predict_prob > 0.45:
                            continue

                        _, extname = os.path.splitext(img_file_original)
                        sha1 = CalcSha1(img_file_original)

                        img_file_dest = os.path.join(dest_dir_fn, label_gt,
                                                     'prob' + str(predict_prob) + '_gt' + labels_gt +
                                                     '__pred' + labels_predict + '_sha1' + sha1 + extname)

                        assert os.path.exists(img_file_original), \
                            img_file_original + 'not exists!'

                        if not os.path.exists(os.path.dirname(img_file_dest)):
                            os.makedirs(os.path.dirname(img_file_dest))

                        print(img_file_dest)
                        shutil.copy(img_file_original, img_file_dest)

print('OK')

'''
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)

roc_auc = auc(fpr, tpr)
# 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

'''



#region test one imagefile
# from LIBS.DataValidation.my_multi_labels import compute_one_model_one_file
# img_source = '/home/jsiec/pics/test/2013年03月11日15时08分IM002998.JPG'
# model_file = os.path.join(sys.path[0], 'weights', 'Multi_label_InceptionResNetV2-009-train0.9982_val0.9971.hdf5')
#
# probabilities, big_class = compute_one_model_one_file(model_file, img_source, image_shape=(299, 299))
#endregion

print('OK')

