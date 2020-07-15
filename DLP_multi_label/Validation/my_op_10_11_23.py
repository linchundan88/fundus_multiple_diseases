import pandas as pd
import shutil
import os
from LIBS.DataPreprocess.my_compute_digest import CalcSha1
dir_preprocess = '/home/ubuntu/multi_labels_2919_1_15/preprocess384/'
dir_original = '/media/ubuntu/data1/multi_labels_2919_1_15/'

BASE_DIR = '/media/ubuntu/data2/2019_8_19'

for filename_csv in ['results_train.csv', 'results_valid.csv']:
    print('computer confusion matrix:', filename_csv + '\n')

    df = pd.read_csv(filename_csv, delimiter=',')

    for _, row in df.iterrows():
        image_file = row["images"]
        img_file_original = image_file.replace(dir_preprocess, dir_original)
        sha1 = CalcSha1(image_file)
        _, extname = os.path.splitext(img_file_original)

        labels = row["labels"]

        for op_class in ['10', '11', '23']:
            predict_prob = row['class' + op_class]
            predict_prob = round(float(predict_prob), 2)

            if op_class in labels:
                img_file_dest = os.path.join(BASE_DIR, op_class, '1',
                                'prob'+ str(predict_prob)+'_' + sha1 + '_' + extname)
            else:
                img_file_dest = os.path.join(BASE_DIR, op_class, '0',
                                 'prob'+ str(predict_prob) + '_' + sha1 + '_' + extname)

            if not os.path.exists(os.path.dirname(img_file_dest)):
                os.makedirs(os.path.dirname(img_file_dest))

            print(img_file_dest)
            shutil.copy(img_file_original, img_file_dest)

print('OK')