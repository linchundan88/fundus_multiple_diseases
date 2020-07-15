import os
import sys
import csv
import pandas as pd
from LIBS.DataPreprocess.my_compute_digest import CalcSha1
import shutil

#region copy csv(exported by database) to dir

dir_original = '/media/ubuntu/data1/multi_labels_2919_1_15/'
dir_preprocess = '/home/ubuntu/multi_labels_2919_1_15/preprocess384/'
dir_dest = '/tmp2/dr1_db'

filename_csv = os.path.abspath(os.path.join(sys.path[0], "..", 'datafiles', 'DLP_SubClass_0.3.csv'))
df = pd.read_csv(filename_csv)

for _, row in df.iterrows():
    labels = row["labels"]
    image_file = row["images"]
    original_image_file = image_file.replace(dir_preprocess, dir_original)

    _, filename = os.path.split(original_image_file)
    _,file_ext = os.path.splitext(filename)

    sha1 = CalcSha1(original_image_file)

    file_dest = os.path.join(dir_dest, str(labels), sha1+file_ext)
    if not os.path.exists(os.path.dirname(file_dest)):
        os.makedirs(os.path.dirname(file_dest))

    print(file_dest)
    shutil.copy(original_image_file, file_dest)

print('OK')
#endregion


filename_csv = 'Subclass_0.3.csv'

base_dir = '/media/ubuntu/data1/DR1_2019_5_5/preprocess384'

with open(filename_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    csv_writer.writerow(['images', 'labels'])

    for dir_path, subpaths, files in os.walk(base_dir, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)

            (filedir, tempfilename) = os.path.split(img_file_source)
            (filename, extension) = os.path.splitext(tempfilename)

            if extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                print('file ext name:', f)
                continue

            if ('/dr1_db/0' in dir_path) or ('/zgh/0' in dir_path):
                csv_writer.writerow([img_file_source, 0])

            if ('/dr1_db/1' in dir_path) or ('/zgh/1' in dir_path):
                csv_writer.writerow([img_file_source, 1])

from LIBS.DataPreprocess import my_data
train_files, train_labels, valid_files, valid_labels = my_data.split_dataset(
        filename_csv, valid_ratio=0.15, random_state=1111)
from LIBS.DataPreprocess.my_data import write_images_labels_csv
write_images_labels_csv(train_files, train_labels, filename_csv='Subclass_' + str(sub_class_no) + '_train.csv')
write_images_labels_csv(valid_files, valid_labels, filename_csv='Subclass_' + str(sub_class_no) + '_valid.csv')

print('OK')