import os
import sys
import pandas as pd
from LIBS.DB.db_helper_conn import get_db_conn
from LIBS.DataPreprocess.my_compute_digest import CalcSha1

def write_csv_to_db():
    dir_original = '/media/ubuntu/data1/multi_labels_2919_1_15'
    dir_preprocess = '/home/ubuntu/multi_labels_2919_1_15/preprocess384'

    db_con = get_db_conn()
    cursor = db_con.cursor()

    filename_csv = os.path.abspath(os.path.join(sys.path[0], "..",
                                'datafiles',  'DLP_patient_based_split.csv'))

    df = pd.read_csv(filename_csv)
    print(len(df))

    for i, row in df.iterrows():
        image_file = str(row["images"])
        labels = str(row["labels"])
        patient_id = str(row["patient_id"])

        image_file_orig = image_file.replace(dir_preprocess, dir_original)
        print(image_file)
        sha1 = CalcSha1(image_file_orig)

        sql = "insert into tb_multi_labels (pic_filename,multi_label1,patient_id,sha1) values(%s,%s,%s,%s)"
        cursor.execute(sql, (image_file, str(labels), patient_id, sha1))

        if i % 20 == 0:
            db_con.commit()

    db_con.commit()


def modify_label_based_on_dir():
    dir = '/media/ubuntu/data2/add_labels/'
    db_con = get_db_conn()
    cursor = db_con.cursor()

    for dir_path, subpaths, files in os.walk(dir, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)
            filename, file_extension = os.path.splitext(img_file_source)
            if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                print('file ext name:', f)
                continue

            sha1 = CalcSha1(img_file_source)
            label_add = img_file_source.replace(dir, '').split('/')[0]

            print(img_file_source)

            sql = "select multi_label1 from tb_multi_labels where sha1=%s"
            cursor.execute(sql, (sha1,))
            results = cursor.fetchall()

            assert len(results) == 1, 'error'

            label_old = results[0][0]
            print(label_old)

            if label_old != '0':  #add label
                sql = "update tb_multi_labels set multi_label2=concat(multi_label1,'_',%s) where sha1=%s"
                cursor.execute(sql, (str(label_add), sha1))
            else:   #normal replace label
                sql = "update tb_multi_labels set multi_label2=%s where sha1=%s"
                cursor.execute(sql, (str(label_add), sha1))

            db_con.commit()

    sql = "update tb_multi_labels set multi_label2=multi_label1 where multi_label2 is null"
    cursor.execute(sql,)

write_csv_to_db()
# modify_label_basedon_dir()

print('OK')