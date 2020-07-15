
import csv, os
from LIBS.DataPreprocess.my_compute_digest import CalcSha1
from LIBS.DB.db_helper_conn import get_db_conn

#read database write csv
def multi_labels_export_to_csv(file_csv, sql, source_dir='',  dest_dir='', contain_pat_id=True):

    if not source_dir.endswith('/'):
        source_dir += '/'
    if not dest_dir.endswith('/'):
        dest_dir += '/'

    db_con = get_db_conn()
    cursor = db_con.cursor()

    cursor.execute(sql)
    results = cursor.fetchall()

    if os.path.exists(file_csv):
        os.remove(file_csv)

    if not os.path.exists(os.path.dirname(file_csv)):
        os.makedirs(os.path.dirname(file_csv))

    with open(file_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        if contain_pat_id:
            csv_writer.writerow(['images', 'labels', 'patient_id'])
        else:
            csv_writer.writerow(['images', 'labels'])

        for rs in results:
            img_file = rs[0]
            if source_dir != '' and dest_dir != '':
                img_file = img_file.replace(source_dir, dest_dir)

            img_file = img_file.replace('\\', '/')

            class_labels = rs[1]
            if class_labels.endswith('_'):  #delete the last character '_'
                class_labels = class_labels[:-1]

            if contain_pat_id:
                patient_id = rs[2]
                if patient_id is None:
                    patient_id = ' '

                csv_writer.writerow([img_file, class_labels, patient_id])
            else:
                csv_writer.writerow([img_file, class_labels])

    db_con.close()

def multi_labels_update_labels(img_file, new_label, op_mode='append'):
    new_label = str(new_label)
    SHA1 = CalcSha1(img_file)

    db_con = get_db_conn()
    cursor = db_con.cursor()

    sql = "SELECT SHA1, BigClasses FROM tb_multi_labels_GT WHERE SHA1=(%s)"
    cursor.execute(sql, (SHA1,))
    results = cursor.fetchall()

    if len(results) > 0:
        if op_mode == 'update':
            sql = "update tb_multi_labels_GT set FilePath=%s BigClasses=%s where SHA1=%s"
            cursor.execute(sql, (img_file, new_label, SHA1))
            db_con.commit()
        elif op_mode == 'append':
            list_labels = results[0][1].split('_')
            if new_label in list_labels:
                new_labels = results[0][1]
            else:
                if results[0][1] == '':
                    new_labels = new_label
                else:
                    new_labels = results[0][1] + '_' + new_label

            sql = "update tb_multi_labels_GT set FilePath=%s, BigClasses=%s where SHA1=%s"
            cursor.execute(sql, (img_file, new_labels, SHA1))
            db_con.commit()
        else:
            assert op_mode in ['update', 'append'], "op_mode Error"
    else:
        sql = "insert into tb_multi_labels_GT(SHA1,FilePath,BigClasses) values(%s,%s,%s)"
        cursor.execute(sql, (SHA1, img_file, new_label))
        db_con.commit()





# multi_labels()