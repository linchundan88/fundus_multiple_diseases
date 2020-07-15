
import sys
sys.path.append('/home/jsiec/PycharmProjects/MyProject/')
sys.path.append('/home/jsiec/PycharmProjects/MyProject/ImgPreprocess/')
import xmlrpc.client

port = 30000  #CPU
port = 30006  #GPU Gerforce 1080

with xmlrpc.client.ServerProxy("http://localhost:" + str(port) + '/') as proxy1:
    img_source = '/home/jsiec/pics/test/2013年03月11日15时08分IM002998.JPG'

    import datetime

    starttime = datetime.datetime.now()

    predict_prob = proxy1.predict_all_classes(img_source)

    print(predict_prob)

    endtime = datetime.datetime.now()

    print((endtime - starttime))

    #使用定制编译的TensorFlow，速度快了一倍，0.91,0,45
    #GPU Gtx 1080  0.22