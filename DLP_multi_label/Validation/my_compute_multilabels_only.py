'''
  generating a results csv
  write results to database
  export FN, FP files
'''

import os, sys
import pandas as pd
import shutil
from LIBS.DataValidation.my_multi_labels import compute_probs, save_multi_label_csv
from LIBS.DLP.my_multilabel_depedence import postprocess_exclusion
from LIBS.DB.db_helper_conn import get_db_conn
from LIBS.DataPreprocess.my_compute_digest import CalcSha1
from LIBS.DataPreprocess.my_operate_labels import get_multi_label_bigclasses

CUDA_VISIBLE_DEVICES = '2'

GEN_RESULTS_CSV = True
CONVERT_PROBS = False #mutual exclusive:4 vs 5, 4 vs 7, 5 vs 6, 6 vs 8, 10 vs 12 vs 13 vs 14

WRITE_PRED_LABELS_DB = False

EXPORT_FN = True
EXPORT_FP = True
WRITE_MATCH_DB = False
DEST_DIR_FN = '/tmp3/2019_8_23/FN'
DEST_DIR_FP = '/tmp3/2019_8_23/FP'

NUM_CLASSES = 30
LIST_THRESHOLD = [0.5 for _ in range(NUM_CLASSES)]

dicts_models = []

# base_model_dir = '/tmp3/bigclass_30_param0.11_3.1'
# model1 = {'model_file': os.path.join(base_model_dir, 'InceptionResnetV2-002-0.959.hdf5'),
#           'image_size': 299, 'model_weight': 1}
# dicts_models.append(model1)
# model2 = {'model_file': os.path.join(base_model_dir, 'Xception-002-0.948.hdf5'),
#           'image_size': 299, 'model_weight': 1}
# dicts_models.append(model2)
# model3 = {'model_file': os.path.join(base_model_dir, 'InceptionV3-003-0.962.hdf5'),
#           'image_size': 299, 'model_weight': 1}
# dicts_models.append(model3)

base_model_dir = '/tmp3/2019_8_21/bigclass_30_param0.12_3.3'
model1 = {'model_file': os.path.join(base_model_dir, 'InceptionResnetV2-traintop-001-0.922.hdf5'),
          'image_size': 299, 'model_weight': 1}
dicts_models.append(model1)
model2 = {'model_file': os.path.join(base_model_dir, 'Xception-traintop-001-0.908.hdf5'),
          'image_size': 299, 'model_weight': 0.85}
dicts_models.append(model2)
model3 = {'model_file': os.path.join(base_model_dir, 'InceptionV3-traintop-001-0.915.hdf5'),
          'image_size': 299, 'model_weight': 1}
dicts_models.append(model3)



if GEN_RESULTS_CSV:
    file_csv_train = os.path.abspath(os.path.join(sys.path[0], "..",
                    'datafiles',  'DLP_only_multilabels.csv'))
    result_csv_train = 'results_multilabels_only.csv'
    list_probs, list_probs_weighted = compute_probs(dicts_models, file_csv=file_csv_train,
                                                    dump_pkl='multi_label_onl.pkl', batch_size=64,
                                                    cuda_visible_devices=CUDA_VISIBLE_DEVICES)
    if CONVERT_PROBS:
        list_probs_weighted = postprocess_exclusion(list_probs_weighted, list_threshold=LIST_THRESHOLD)
    save_multi_label_csv(file_csv=file_csv_train, list_probs=list_probs_weighted,
                         csv_results=result_csv_train, list_threshold=LIST_THRESHOLD)

if WRITE_PRED_LABELS_DB:
    from LIBS.DataPreprocess.my_operate_labels import get_multi_label_bigclasses

    dir_preprocess = '/home/ubuntu/multi_labels_2919_1_15/preprocess384/'
    dir_original = '/media/ubuntu/data1/multi_labels_2919_1_15/'

    filename_csv ='results_multilabels_only.csv'
    df = pd.read_csv(filename_csv, delimiter=',')

    db_con = get_db_conn()
    cursor = db_con.cursor()

    count = len(df.index)
    for i in range(count):
        img_file_preprocess = df.at[i, 'images']
        img_file_original = img_file_preprocess.replace(dir_preprocess, dir_original)

        print(img_file_original)
        sha1 = CalcSha1(img_file_original)

        labels_gt = df.at[i, 'labels']
        labels_gt = str(labels_gt)
        labels_gt = get_multi_label_bigclasses(labels_gt, num_classes=30)

        labels_predict = df.at[i, 'bigclasses']
        labels_predict = str(labels_predict)
        labels_predict = get_multi_label_bigclasses(labels_predict, num_classes=30)

        sql = "update tb_multi_labels1 set multi_label_gt=%s,multi_label_predict=%s where sha1=%s"
        cursor.execute(sql, (labels_gt, labels_predict, sha1))
        db_con.commit()

if EXPORT_FN:
    dir_preprocess = '/home/ubuntu/multi_labels_2919_1_15/preprocess384/'
    dir_original = '/media/ubuntu/data1/multi_labels_2919_1_15/'

    filename_csv = 'results_multilabels_only.csv'
    df = pd.read_csv(filename_csv, delimiter=',')

    db_con = get_db_conn()
    cursor = db_con.cursor()

    match_num = 0

    count = len(df.index)
    for i in range(count):
        img_file_preprocess = df.at[i, 'images']
        img_file_original = img_file_preprocess.replace(dir_preprocess, dir_original)

        sha1 = CalcSha1(img_file_original)

        labels_gt = str(df.at[i, 'labels'])
        labels_gt = get_multi_label_bigclasses(labels_gt, num_classes=NUM_CLASSES)

        labels_predict = str(df.at[i, 'bigclasses'])
        labels_predict = get_multi_label_bigclasses(labels_predict, num_classes=NUM_CLASSES)

        set_gt = set()
        for label in labels_gt.split('_'):
            if label != '_' and label != '':
                set_gt.add(label)

        set_predict = set()
        for label in labels_predict.split('_'):
            if label != '_' and label != '':
                set_predict.add(label)

        match_ok = False

        if len(set_gt) - len(set_predict) == 1:
            match_local = True
            for label in set_predict:
                if label not in set_gt:
                    match_local = False

            if match_local:
                match_ok = True

            #find the missing label
            for label in set_gt:
                if label not in set_predict:
                    label_target = label

        if match_ok == True:
            match_num += 1

            predict_prob = df.at[i, 'class' + label_target]
            predict_prob = round(float(predict_prob), 2)

            _, extname = os.path.splitext(img_file_original)
            img_file_dest = os.path.join(DEST_DIR_FN, label_target,
                                         'prob' + str(predict_prob) + '_gt' + labels_gt +
                                         '__pred' + labels_predict + '_sha1' + sha1 + extname)

            assert os.path.exists(img_file_original), \
                img_file_original + 'not exists!'

            if not os.path.exists(os.path.dirname(img_file_dest)):
                os.makedirs(os.path.dirname(img_file_dest))

            print(img_file_dest)
            shutil.copy(img_file_original, img_file_dest)

            if WRITE_MATCH_DB:
                sql = "update tb_multi_labels1 set match_flag=2 where sha1=%s"
                cursor.execute(sql, (sha1,))
                db_con.commit()

    print(match_num)

if EXPORT_FP:
    dir_preprocess = '/home/ubuntu/multi_labels_2919_1_15/preprocess384/'
    dir_original = '/media/ubuntu/data1/multi_labels_2919_1_15/'

    filename_csv = 'results_multilabels_only.csv'
    df = pd.read_csv(filename_csv, delimiter=',')

    db_con = get_db_conn()
    cursor = db_con.cursor()

    match_num = 0

    count = len(df.index)
    for i in range(count):
        img_file_preprocess = df.at[i, 'images']
        img_file_original = img_file_preprocess.replace(dir_preprocess, dir_original)

        sha1 = CalcSha1(img_file_original)

        labels_gt = str(df.at[i, 'labels'])
        labels_gt = get_multi_label_bigclasses(labels_gt, num_classes=NUM_CLASSES)

        labels_predict = str(df.at[i, 'bigclasses'])
        labels_predict = get_multi_label_bigclasses(labels_predict, num_classes=NUM_CLASSES)

        set_gt = set()
        for label in labels_gt.split('_'):
            if label != '_' and label != '':
                set_gt.add(label)

        set_predict = set()
        for label in labels_predict.split('_'):
            if label != '_' and label != '':
                set_predict.add(label)

        match_ok = False

        if len(set_predict) - len(set_gt) == 1:
            match_local = True
            for label in set_gt:
                if label not in set_predict:
                    match_local = False

            if match_local:
                match_ok = True

            #find the missing label
            for label in set_predict:
                if label not in set_gt:
                    label_target = label

        if match_ok == True:
            match_num += 1

            predict_prob = df.at[i, 'class' + label_target]
            predict_prob = round(float(predict_prob), 2)

            _, extname = os.path.splitext(img_file_original)
            img_file_dest = os.path.join(DEST_DIR_FP, label_target,
                                         'prob' + str(predict_prob) + '_gt' + labels_gt +
                                         '__pred' + labels_predict + '_sha1' + sha1 + extname)

            assert os.path.exists(img_file_original), \
                img_file_original + 'not exists!'

            if not os.path.exists(os.path.dirname(img_file_dest)):
                os.makedirs(os.path.dirname(img_file_dest))

            print(img_file_dest)
            shutil.copy(img_file_original, img_file_dest)

            if WRITE_MATCH_DB:
                sql = "update tb_multi_labels1 set match_flag=3 where sha1=%s"
                cursor.execute(sql, (sha1,))
                db_con.commit()

    print(match_num)


print('OK')