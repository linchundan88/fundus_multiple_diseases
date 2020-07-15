'''
'''

import sys, os

dir_original = '/media/ubuntu/data1/multi_labels_2919_1_15/'
dir_preprocess = '/home/ubuntu/multi_labels_2919_1_15/preprocess384/'


filename_csv_contain_subclass = os.path.abspath(os.path.join(sys.path[0], "..",
                            'datafiles', 'DLP_SubClass_0.2.csv'))

if os.path.exists(filename_csv_contain_subclass):
    os.remove(filename_csv_contain_subclass)

sql1 = "SELECT concat(pic_dir,pic_name) as filename, subclass_0_2,pat_id from  v_dlp_train_valid where  subclass_0_2 is not null "
sql = sql1

from LIBS.DB.db_helper_multi_labels import multi_labels_export_to_csv
multi_labels_export_to_csv(filename_csv_contain_subclass, sql=sql,
                           source_dir=r'\\10.12.192.135\pacs\JSIEC\label' + '\\',
                           dest_dir=dir_preprocess)


print('OK')