import pandas as pd
from LIBS.DB.db_helper_conn import get_db_conn

csv_file = 'DLP_patient_based_split.csv'
df = pd.read_csv(csv_file)

dir_preprocess = '/home/ubuntu/multi_labels_2919_1_15/preprocess384/'
dir_original = '/media/ubuntu/data1/multi_labels_2919_1_15/'

db_con = get_db_conn()
cursor = db_con.cursor()

i = 0
for _, row in df.iterrows():
    image_file = str(row["images"])
    image_file_original = image_file.replace(dir_preprocess, dir_original)
    labels = str(row["labels"])

    sql = "select * from tb_multi_labels1 where pic_filename=%s"
    cursor.execute(sql, (image_file_original,))
    results = cursor.fetchall()

    if len(results) > 0:
        print(i)
        print(image_file_original)

        sql = "update tb_multi_labels1 set match_flag=1 where pic_filename=%s"
        cursor.execute(sql, (image_file_original,))

        if i % 20 == 0:
            db_con.commit()

        i += 1

    db_con.commit()

print('OK!')