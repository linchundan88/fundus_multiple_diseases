from keras.layers import *
from keras.models import Model
import keras
import cv2
from LIBS.CNN_Models.Utils.my_utils import get_last_conv_layer_number

# Modify the model to output both original outout and the output of last_conv_layer
def get_CNN_model(model_file1, last_layer=-1):

    model = keras.models.load_model(model_file1, compile=False)
    # model.summary()

    # get the last layer before global average pooling
    last_conv_layer = get_last_conv_layer_number(model)

    # get AMP layer weights
    last_layer_weights = model.layers[last_layer].get_weights()[0]

    # extract wanted output
    output_model = Model(inputs=model.input,
            outputs=(model.layers[last_conv_layer].output, model.layers[last_layer].output))

    return output_model, last_layer_weights

# Method to genrate CAM
def gen_CAM(img_input, model, pred, all_amp_layer_weights,
            cam_relu, ImageSize):

    last_conv_output, pred_vec = model.predict(img_input)
    # pred = np.argmax(pred_vec)  # get model's prediction class
    # Remove single-dimensional entries from the shape of an array.
    last_conv_output = np.squeeze(last_conv_output)

    # get AMP layer weights
    # weights corresponding to the second-to-last layer to the predict class of last layer
    amp_layer_weights = all_amp_layer_weights[:, pred]  # dim: (2048,)
    # jijie add relu
    # 对于每一个类别C，每个特征图K的均值都有一个对应的w
    if cam_relu:
        amp_layer_weights = np.maximum(amp_layer_weights, 0)

    cam_small = np.dot(last_conv_output, amp_layer_weights)  # compute CAM

    # region generate CAM
    cam = cv2.resize(cam_small, (ImageSize, ImageSize))
    cam = np.maximum(cam, 0)   #ReLU
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image = img_input[0, :]
    # because images_generator do normalization
    # x_valid /= 255.
    # x_valid -= 0.5
    # x_valid *= 2.

    image /= 2
    image += 0.5
    image *= 255.
    image = image.astype(np.uint8)
    #

    image -= np.min(image)
    image = np.minimum(image, 255)

    # cv2.imwrite('test.jpg', image)  # 0-2

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    # endregion

    return cam

