import os
import sys

dir_original = '/media/ubuntu/data1/multi_labels_2919_1_15/'
dir_preprocess = '/home/ubuntu/multi_labels_2919_1_15/preprocess384/'

filename_csv_source = os.path.abspath(os.path.join(sys.path[0], "..",
    'datafiles', 'DLP_only_multilabels.csv'))

# FILED_NAME = 'multi_label_new_9'
# sql1 = "SELECT concat(pic_dir,pic_name) as filename, {0} as multi_label, patient_id FROM tb_multi_labels1 ".format(FILED_NAME)
sql1 = "SELECT pic_filename as filename, multi_label_gt as multi_label, patient_id FROM tb_multi_labels1 where match_flag is null "

# where match_flag is null

from LIBS.DB.db_helper_multi_labels import multi_labels_export_to_csv
multi_labels_export_to_csv(filename_csv_source,
   sql=sql1,
   source_dir=dir_original, dest_dir=dir_preprocess)

print('OK')

# '/media/ubuntu/data1/multi_labels_2919_1_15/原训练集验证集未重标/0/18#1.0,0#0.0,1#0.0___27808_right.jpeg'

# '\\10.12.192.135\pacs\JSIEC\label\原训练集验证集可疑多标签\0\'