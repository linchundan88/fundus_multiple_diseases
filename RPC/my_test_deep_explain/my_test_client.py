
import os, shutil
import xmlrpc.client

port = 24000  #CPU

with xmlrpc.client.ServerProxy("http://localhost:" + str(port) + '/') as proxy1:

    # img_source = '/media/ubuntu/data1/公开数据集/IDRID/B. Disease Grading/IDRID_results_2019_3_13/1/prob100#IDRiD_056.jpg'
    # import datetime
    # starttime = datetime.datetime.now()
    # result = proxy1.server_deep_explain(img_source, 1, True)
    # print(result)
    # endtime = datetime.datetime.now()
    # print((endtime - starttime))


    source_dir = '/media/ubuntu/data1/公开数据集/IDRID/B. Disease Grading/IDRID_results_2019_3_13/1'
    dest_dir = '/media/ubuntu/data1/公开数据集/IDRID/B. Disease Grading/IDRID_results_saliency/1'

    for dir_path, subpaths, files in os.walk(source_dir, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)

            filename, file_extension = os.path.splitext(img_file_source)

            if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                print('file ext name:', f)
                continue


            img_file_dest = img_file_source.replace(source_dir, dest_dir)
            if not os.path.exists(os.path.dirname(img_file_dest)):
                os.makedirs(os.path.dirname(img_file_dest))

            result = proxy1.server_deep_explain(img_file_source, 1, True)
            print(img_file_dest)
            shutil.copy(result, img_file_dest)

print('OK')