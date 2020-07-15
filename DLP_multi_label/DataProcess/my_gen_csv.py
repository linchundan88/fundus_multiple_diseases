'''
  convert_patient_id
     as for every person, avoid left eye in training dataset and right eye in valid dataset or vice verse

  convert_only_patient_null(only test)
     export patient_id is null, manually split

'''

import os, sys, csv
import pandas as pd
from sklearn.utils import shuffle
from LIBS.DataPreprocess import my_data

EXP_DB_TO_CSV = True
OP_PAT_ID_SPLIT = True

#region export csv from database

datafile_type = '2020_3_13'
filename_csv = os.path.abspath(os.path.join(sys.path[0], "..",
    'datafiles/', datafile_type, 'DLP_multi_label.csv'))

dir_original = '/media/ubuntu/data1/multi_labels_2020_2_29/original'
dir_preprocess = '/media/ubuntu/data1/multi_labels_2020_2_29/preprocess384/'

# sql = "SELECT pic_filename, multi_label2 as labels, patient_id  FROM tb_multi_labels " \
#         " where patient_id is not null and multi_label2 not like '%99%' " \
#       "union SELECT pic_filename, multi_label_gt_1 as labels, patient_id FROM tb_multi_labels1 " \
#         "where patient_id is not null and match_flag is not null and multi_label_gt_1 not like '%99%' "

# sql = "SELECT pic_filename, final_label as labels, patient_id  FROM tb_multi_labels " \
#         " where patient_id is not null and final_label not like '%99%' " \
#       "union SELECT pic_filename, final_label as labels, patient_id FROM tb_multi_labels1 " \
#         "where patient_id is not null and match_flag is not null and final_label not like '%99%' "

# sql = "SELECT pic_filename, final_label_20200223 as labels, patient_id  FROM tb_multi_labels " \
#         " where patient_id is not null and final_label_20200223 not like '%99%' "

sql = "SELECT pic_filename, final_label_20200223 as labels, patient_id  FROM tb_multi_labels " \
        " where train_valid in ('Public','Expert','train','valid') and patient_id is not null and final_label_20200223 not like '%99%' "


if EXP_DB_TO_CSV:
    from LIBS.DB.db_helper_multi_labels import multi_labels_export_to_csv
    multi_labels_export_to_csv(filename_csv, sql=sql,
                           source_dir=dir_original, dest_dir=dir_preprocess)

#endregion


#region patient_id, and split train validation

def convert_patient_id(filename_csv_source, filename_csv_result):
    df = pd.read_csv(filename_csv_source)
    df = shuffle(df)
    set_patient_id = set()

    for index, row in df.iterrows():
        patient_id = str(row["patient_id"]).strip()
        if patient_id == '':  # blank will process later, and add to csv first
            continue

        set_patient_id.add(patient_id)

    if os.path.exists(filename_csv_result):
        os.remove(filename_csv_result)

    with open(filename_csv_result, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'labels', 'patient_id'])

        for index, row in df.iterrows():
            patient_id = str(row["patient_id"]).strip()
            # 1736 no patient_id
            if patient_id == '':
                images = row["images"]
                labels = row["labels"]
                csv_writer.writerow([images, labels, ' '])

        for patient_id in set_patient_id:
            df_search = df[df['patient_id'].isin([patient_id])]

            for index, row in df_search.iterrows():
                images = row["images"]
                labels = row["labels"]
                patient_id = row["patient_id"]

                csv_writer.writerow([images, labels, patient_id])

def convert_only_patient_null(filename_csv_source, filename_csv_result):
    df = pd.read_csv(filename_csv_source)
    df = shuffle(df)

    if os.path.exists(filename_csv_result):
        os.remove(filename_csv_result)

    with open(filename_csv_result, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'labels', 'patient_id'])

        for index, row in df.iterrows():
            patient_id = row["patient_id"].strip()
            # 1736 no patient_id
            if patient_id == '':
                images = row["images"]
                labels = row["labels"]
                csv_writer.writerow([images, labels, ' '])


if OP_PAT_ID_SPLIT:
    filename_csv_result = os.path.abspath(os.path.join(sys.path[0], "..",
                    'datafiles/2020_2_29',  'DLP_patient_based_split.csv'))
    convert_patient_id(filename_csv, filename_csv_result)

    #input dataset order by patient_id, so do not shuffle
    train_files, train_labels, valid_files, valid_labels = my_data.split_dataset(
        filename_csv_result,  valid_ratio=0.15, shuffle=False, random_state=2223)

    filename_csv_train = os.path.join(os.path.abspath('..'),
                                      'datafiles', datafile_type, 'DLP_patient_based_split_train.csv')
    my_data.write_images_labels_csv(train_files, train_labels, filename_csv=filename_csv_train)
    filename_csv_valid = os.path.join(os.path.abspath('..'),
                                      'datafiles', datafile_type, 'DLP_patient_based_split_valid.csv')
    my_data.write_images_labels_csv(valid_files, valid_labels, filename_csv=filename_csv_valid)


    # filename_csv_only_patient_null = os.path.abspath(os.path.join(sys.path[0], "..",
    #                 'datafiles',  'DLP_only_patient_null.csv'))
    # convert_only_patient_null(filename_csv_source, filename_csv_only_patient_null)

#endregion


print('OK')



