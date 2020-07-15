import os, csv
import pandas as pd
import cv2

from LIBS.ImgPreprocess.my_preprocess_dir import do_resize_dir

def gen_csv():

    file_csv = 'Refuge.csv'

    if os.path.exists(file_csv):
        os.remove(file_csv)

    with open(file_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'masks'])

        dir_path = '/media/ubuntu/data1/公开数据集/视盘/Refuge/Training400/images'

        for dir_path, subpaths, files in os.walk(dir_path, False):
            for f in files:
                img_file_source = os.path.join(dir_path, f)

                filename, file_extension = os.path.splitext(img_file_source)

                if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                    print('file ext name:', f)
                    continue

                img_file_mask = img_file_source.replace('/images/', '/masks/')
                img_file_mask = img_file_mask.replace('.jpg', '.bmp')

                if os.path.exists(img_file_mask):
                    csv_writer.writerow([img_file_source, img_file_mask])

        dir_path = '/media/ubuntu/data1/公开数据集/视盘/Refuge/Validation400/images'

        for dir_path, subpaths, files in os.walk(dir_path, False):
            for f in files:
                img_file_source = os.path.join(dir_path, f)

                filename, file_extension = os.path.splitext(img_file_source)

                if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                    print('file ext name:', f)
                    continue

                img_file_mask = img_file_source.replace('/images/', '/masks/')
                img_file_mask = img_file_mask.replace('.jpg', '.bmp')

                if os.path.exists(img_file_mask):
                    csv_writer.writerow([img_file_source, img_file_mask])

def op_images():
    filename_csv = 'Refuge.csv'
    df = pd.read_csv(filename_csv)  # panda dataframe

    count = len(df.index)
    for i in range(count):
        img_file = df.at[i, 'images']
        img_mask_file = df.at[i, 'masks']

        img1 = cv2.imread(img_mask_file)

        threthold = 220
        ret, img_thresh1 = cv2.threshold(img1, threthold, 255, cv2.THRESH_BINARY)

        img_thresh1 = - img_thresh1
        img_thresh1 += 255

        print(img_mask_file)

        cv2.imwrite(img_mask_file, img_thresh1)


# gen_csv()

# op_images()


do_resize_dir('/media/ubuntu/data1/公开数据集/视盘/Refuge/preprocess/',
              '/media/ubuntu/data1/公开数据集/视盘/Refuge/preprocess384/', image_size=384)

print('OK')