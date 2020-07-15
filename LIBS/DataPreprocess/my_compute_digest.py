import hashlib
import os
import csv
import pandas as pd

def CalcSha1(filepath):
    with open(filepath, 'rb') as f:
        sha1obj = hashlib.sha1()
        sha1obj.update(f.read())
        hash = sha1obj.hexdigest()

        return hash


def CalcMD5(filepath):
    with open(filepath, 'rb') as f:
        md5obj = hashlib.md5()
        md5obj.update(f.read())
        hash = md5obj.hexdigest()

        return hash

def compute_digest_dir(base_dir, filename_csv, small_size=2048):
    if os.path.exists(filename_csv):
        os.remove(filename_csv)

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['digest', 'file'])

        for dir_path, subpaths, files in os.walk(base_dir, False):
            for f in files:
                image_file = os.path.join(dir_path, f)

                file_base, file_ext = os.path.splitext(image_file) #分离文件名与扩展名
                if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                    continue

                size = os.path.getsize(image_file)
                if size < small_size:
                    continue

                digestSha1 = CalcSha1(image_file)

                print('filename:', image_file)
                csv_writer.writerow([digestSha1, image_file])


def remove_match_or_not(base_dir, filename_csv, remove_match = False,
                        remove_not_match=False):
    df = pd.read_csv(filename_csv)  # panda dataframe

    for dir_path, subpaths, files in os.walk(base_dir, False):
        for f in files:
            file_base, file_ext = os.path.splitext(f)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue

            img_file = os.path.join(dir_path, f)
            digestSha1 = CalcSha1(img_file)

            # df_record = df.iloc[1]  #iloc  基于整数位置的索引  slice

            # df_record = df.loc[digestSha1]   #loc 按索引选取 通过标签选择
            # digest_this = df_record[0]
            # filename_this = df_record[1]

            df_search = df.loc[df['digest'].isin([digestSha1])]
            if len(df_search) == 0:
                if remove_not_match:    #匹配不到则删除
                    print('remove file:', img_file)
                    os.remove(img_file)
            else:
                if remove_match:    #匹配到则删除
                    print('remove file:', img_file)
                    os.remove(img_file)


def del_duplicate(dir_reference, dir_del, filename_csv='/tmp/del.csv'):
    compute_digest_dir(dir_reference, filename_csv, small_size=2048)

    df = pd.read_csv(filename_csv)

    for dir_path, subpaths, files in os.walk(dir_del, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)

            # filename, file_extension = os.path.splitext(img_file_source)

            # if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
            #     print('file ext name:', f)
            #     continue

            sha1 = CalcSha1(img_file_source)

            query_str = "digest=='{0}'".format(sha1)
            query_result = df.query(query_str)

            if len(query_result) > 0:
                print('delete file:', img_file_source)
                os.remove(img_file_source)



if __name__ == '__main__':

    basedir = '/media/ubuntu/data1/最大的三大数据集/70000/original'
    compute_digest_dir(basedir, '70000.csv')

    basedir = '/media/ubuntu/data1/最大的三大数据集/eye_pacs/original'
    compute_digest_dir(basedir, 'eyepacs.csv')

    basedir = '/media/ubuntu/data1/最大的三大数据集/jkkc/original'
    compute_digest_dir(basedir, 'jkkc.csv')


