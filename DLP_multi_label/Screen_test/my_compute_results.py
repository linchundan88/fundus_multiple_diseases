
import os
from LIBS.DataPreprocess.my_operate_labels import convert_multilabels
from LIBS.DataValidation.my_multi_labels import compute_probs, save_multi_label_csv
from LIBS.DLP.my_multilabel_depedence import postprocess_exclusion, postprocess_all_negative
import pickle
import pandas as pd
from LIBS.DataPreprocess.my_compute_digest import CalcSha1

from LIBS.DB.db_helper_conn import get_db_conn

CUDA_VISIBLE_DEVICES = '2'

dir_original = '/media/ubuntu/data2/医联体应用_已筛/original'
dir_preprocess = '/media/ubuntu/data2/医联体应用_已筛/preprocess384'
DIR_DEST_BASE = '/media/ubuntu/data2/医联体应用_已筛/results'

DO_PREPROCESS = True
if DO_PREPROCESS:
    from LIBS.ImgPreprocess.my_preprocess_dir import do_process_dir
    do_process_dir(dir_original, dir_preprocess, image_size=384,
                   add_black_pixel_ratio=0.02)

GENERATE_CSV = True

NUM_CLASSES = 30
LIST_THRESHOLD = [0.5 for _ in range(NUM_CLASSES)]

GEN_RESULTS = True
COMPUTE_EXCLUSION_PROBS = True
COMPUTE_ALLNEGATIVE = True #all negative ,prob of non-referable is low.

WRITE_TO_DB_IMG_FILES = False
WRITE_TO_DB_LABELS = False

models = []
base_model_dir = '/home/ubuntu/dlp/deploy_models_2019//bigclasses_multilabels_new/bigclass_30_param0.11_2.4/'
model1 = {'model_file': os.path.join(base_model_dir, 'InceptionResnetV2-traintop-001-0.919.hdf5'),
          'image_size': 299, 'model_weight': 1}
models.append(model1)
model2 = {'model_file': os.path.join(base_model_dir, 'Xception-traintop-001-0.910.hdf5'),
          'image_size': 299, 'model_weight': 1}
models.append(model2)
model3 = {'model_file': os.path.join(base_model_dir, 'InceptionV3-traintop-001-0.913.hdf5'),
          'image_size': 299, 'model_weight': 1}
models.append(model3)


PREFIX = 'screening_2020_6_16'

filename_csv = os.path.join(DIR_DEST_BASE, PREFIX + '.csv')
filename_csv_results = os.path.join(DIR_DEST_BASE + '_results.csv')
filename_csv_results_with_exclusion = os.path.join(DIR_DEST_BASE, PREFIX + '_results_exclusion.csv')
filename_csv_results_with_exclusion_negative = os.path.join(DIR_DEST_BASE, PREFIX + '_results_exclusion_negative.csv')

if GENERATE_CSV:
    from LIBS.DataPreprocess.my_data import write_csv_dir_nolabel
    write_csv_dir_nolabel(filename_csv, dir_preprocess)

if GEN_RESULTS:
    list_probs, list_probs_weighted = compute_probs(models, file_csv=filename_csv,
                                                    batch_size=64, cuda_visible_devices=CUDA_VISIBLE_DEVICES)
    dump_pkl = os.path.join(DIR_DEST_BASE, PREFIX + '_results.pkl')
    with open(dump_pkl, 'wb') as file:
        pickle.dump(list_probs_weighted, file)

    save_multi_label_csv(file_csv=filename_csv, list_probs=list_probs_weighted,
                         csv_results=filename_csv_results,
                         list_threshold=LIST_THRESHOLD)

    #postprocessing exclusion
    if COMPUTE_EXCLUSION_PROBS:
        list_probs_weighted = postprocess_exclusion(list_probs_weighted, list_threshold=LIST_THRESHOLD)
        dump_pkl = os.path.join(PREFIX + '_results_with_exclusion.pkl')
        with open(dump_pkl, 'wb') as file:
            pickle.dump(list_probs_weighted, file)

        save_multi_label_csv(file_csv=filename_csv, list_probs=list_probs_weighted,
                             csv_results=filename_csv_results_with_exclusion,
                             list_threshold=LIST_THRESHOLD)
    if COMPUTE_ALLNEGATIVE:
        list_probs_weighted = postprocess_all_negative(list_probs_weighted, list_threshold=LIST_THRESHOLD)
        dump_pkl = os.path.join(PREFIX + '_results_with_exclusion_with_negative.pkl')
        with open(dump_pkl, 'wb') as file:
            pickle.dump(list_probs_weighted, file)

        save_multi_label_csv(file_csv=filename_csv, list_probs=list_probs_weighted,
                             csv_results=filename_csv_results_with_exclusion_negative,
                             list_threshold=LIST_THRESHOLD)

if WRITE_TO_DB_IMG_FILES:
    db_con = get_db_conn()
    cursor = db_con.cursor()

    df = pd.read_csv(filename_csv_results, delimiter=',')
    for _, row in df.iterrows():
        image_file = row["images"]
        image_file_original = image_file.replace(dir_preprocess, dir_original)
        print(image_file_original)
        sha1 = CalcSha1(image_file_original)

        sql = "insert into tb_multi_label_test(pic_name, sha, type) values(%s, %s, %s)"
        cursor.execute(sql, (image_file_original, sha1, 'screen_test'))
        db_con.commit()

if WRITE_TO_DB_LABELS:
    db_con = get_db_conn()
    cursor = db_con.cursor()

    df = pd.read_csv(filename_csv_results, delimiter=',')
    for _, row in df.iterrows():
        image_file = row["images"]
        image_file_original = image_file.replace(dir_preprocess, dir_original)
        print(image_file_original)
        sha1 = CalcSha1(image_file_original)

        new_labels = convert_multilabels(str(row["bigclasses"]))
        sql = "update tb_multi_label_test set label3=%s where sha1=%s"
        cursor.execute(sql, (new_labels, sha1))
        db_con.commit()

    if COMPUTE_EXCLUSION_PROBS:
        df = pd.read_csv(filename_csv_results_with_exclusion, delimiter=',')
        for _, row in df.iterrows():
            image_file = row["images"]
            image_file_original = image_file.replace(dir_preprocess, dir_original)
            print(image_file_original)
            sha1 = CalcSha1(image_file_original)

            new_labels = convert_multilabels(str(row["bigclasses"]))
            sql = "update tb_multi_label_test set label2_with_e=%s where sha1=%s"
            cursor.execute(sql, (new_labels, sha1))
            db_con.commit()

    df = pd.read_csv(filename_csv_results_with_exclusion_negative, delimiter=',')
    for _, row in df.iterrows():
        image_file = row["images"]
        image_file_original = image_file.replace(dir_preprocess, dir_original)
        print(image_file_original)
        sha1 = CalcSha1(image_file_original)

        new_labels = convert_multilabels(str(row["bigclasses"]))
        sql = "update tb_multi_label_test set label1_with_e_n=%s where sha1=%s"
        cursor.execute(sql, (new_labels, sha1))
        db_con.commit()


print('OK')

