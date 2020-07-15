
import os, shutil
import xmlrpc.client

port = 25000

with xmlrpc.client.ServerProxy("http://localhost:" + str(port) + '/') as proxy1:

    # img_file_source = '/tmp1/brvo.jpg'
    img_file_source = '/tmp1/408ab0fbc5f676a1d674733e9f21b5e686fc7078.jpg'
    # server_shap_deep_explain(model_no, img_source, preprocess=True, ranked_outputs=1)
    list_classes, list_images = proxy1.server_shap_deep_explain(0, img_file_source, True, 2)

    list_classes, list_images = proxy1.server_shap_deep_explain(1, img_file_source, True, 2)


    print(list_images)

print('OK')