
import os, sys, csv
import pandas as pd
from sklearn.utils import shuffle
from LIBS.DataPreprocess import my_data

DO_PREPROCESS = True
GENERATE_CSV = True

from LIBS.ImgPreprocess.my_preprocess_dir import do_process_dir

dir_original = '/media/ubuntu/data2/其它数据集/筛查集//original'
dir_preprocess = '/media/ubuntu/data2/其它数据集/筛查集/preprocess384'

if DO_PREPROCESS:
    do_process_dir(dir_original, dir_preprocess, image_size=384,
               add_black_pixel_ratio=0.02)

filename_csv = 'screening.csv'



current_dir = os.path.abspath(os.path.dirname(__file__))
filename_csv = os.path.join(current_dir, filename_csv)


from LIBS.DataPreprocess.my_data import write_csv_dir_nolabel

if GENERATE_CSV:
    write_csv_dir_nolabel(filename_csv, dir_preprocess)

print('OK')