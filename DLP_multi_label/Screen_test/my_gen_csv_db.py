
import os, sys, csv
import pandas as pd
from sklearn.utils import shuffle
from LIBS.DataPreprocess import my_data


filename_csv = 'test_dataset.csv'

# dir_original = '/media/ubuntu/data1/multi_labels_2919_1_15/'
# dir_preprocess = '/home/ubuntu/multi_labels_2919_1_15/preprocess384/'

dir_original = '//10.12.192.135/label/测试集_已标注/'
dir_preprocess = '/media/ubuntu/data2/测试集_已标注/preprocess384/'

# sql ="select replace(concat( pic_dir,'\\',pic_name),'\\','/'),multi_label from  tb_multi_label_test where dataset = 'Test'"
sql ="select full_filename, multi_label, pat_id from tb_multi_label_test where dataset = 'Test' and multi_label!=''"

from LIBS.DB.db_helper_multi_labels import multi_labels_export_to_csv
multi_labels_export_to_csv(filename_csv, sql=sql,
                       source_dir=dir_original, dest_dir=dir_preprocess)


print('OK')