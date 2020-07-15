# first saliency 30 second  First time convert model
#CPU second time saliency 6 seconds, cam 2 secomds
#GPU second time saliency 6 seconds, cam 2seconds
# most time spend on : grads = K.gradients(overall_loss, self.input_tensor)[0]

import matplotlib.pyplot as plt
import uuid, cv2
from vis.visualization import visualize_saliency, overlay
from keras import activations

from vis.utils import utils
import numpy as np
import matplotlib.cm as cm
from vis.visualization import visualize_cam
from keras.layers import *

#keras train history
def plot_history(history):
    plt.style.use('ggplot')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

# Generates an attention heatmap over the seed_input for
#  maximizing filter_indices output in the given layer_idx.
def get_guided_saliency(model, img1, layer_idx=None, predict_class=None):

    # https://raghakot.github.io/keras-vis/vis.visualization/#visualize_saliency
    #  If you are visualizing final keras.layers.Dense layer, consider switching 'softmax' activation for 'linear'
    # Swap softmax with linear
    # model.layers[layer_idx].activation = activations.linear
    # model = LIBS.apply_modifications(model)  # saliency maps no much difference

    if predict_class is None:
        prob = model.predict(np.expand_dims(img1, axis=0))
        predict_class = np.argmax(prob)

    if layer_idx is None:
        layer_idx = len(model.layers) - 1

    # Modifies backprop to only propagate positive gradients for positive activations.
    modifier = 'guided'  # ['guided', 'relu']
    if predict_class is not None:
        grads = visualize_saliency(model, layer_idx, filter_indices=[predict_class],
                                   seed_input=img1, backprop_modifier=modifier)
    else:
        grads = visualize_saliency(model, layer_idx,
                                   seed_input=img1, backprop_modifier=modifier)

    str_uuid = str(uuid.uuid1())
    filename = '/tmp/' + str_uuid + '.png'

    cam = cv2.applyColorMap(np.uint8(255 * grads), cv2.COLORMAP_JET)
    cv2.imwrite(filename, cam)

    # plt.title('Saliency Maps')
    # plt.imshow(grads, cmap='jet')
    # fig = plt.gcf()
    # fig.set_size_inches(3, 3)
    #
    #
    # fig.savefig(filename, dpi=100)  # fig.savefig('/tmp/test.png', dpi=100)
    # plt.close()


    return filename


'''
penultimate_layer_idx: The pre-layer to layer_idx whose feature maps should be used to compute gradients wrt filter output. If not provided, it is set to the nearest penultimate Conv or Pooling layer.
backprop_modifier [None, 'guided', 'relu']
'''

def get_guided_cam(model, img1, layer_idx=None, predict_class=None,
                   penultimate_layer_idx=None, backprop_modifier='guided'):

    if predict_class is None:
        prob = model.predict(np.expand_dims(img1, axis=0))
        predict_class = np.argmax(prob)

    if layer_idx is None:
        layer_idx = len(model.layers) - 1

    if penultimate_layer_idx is None:
        for i in range(len(model.layers)-1, -1, -1):
            if isinstance(model.layers[i], Conv2D) or \
                    isinstance(model.layers[i], MaxPool2D) or\
                    isinstance(model.layers[i], SeparableConv2D):
                penultimate_layer_idx=i
                break

    # penultimate_layer_idx = LIBS.find_layer_idx(model, 'mixed10')

    grads = visualize_cam(model, layer_idx, filter_indices=[predict_class],
                          seed_input=img1, penultimate_layer_idx=penultimate_layer_idx,
                          backprop_modifier=backprop_modifier)


    str_uuid = str(uuid.uuid1())
    filename = '/tmp/' + str_uuid + '.png'

    jet_heatmap = cv2.applyColorMap(np.uint8(255 * grads), cv2.COLORMAP_JET)
    cv2.imwrite('1111.png', jet_heatmap)

    img1 /= 2.0
    img1 += 0.5
    img1 *= 255.

    img1 = img1.astype(np.uint8)
    # x_train /= 255.
    # x_train -= 0.5
    # x_train *= 2.

    img_cam = overlay(jet_heatmap, img1)

    cv2.imwrite(filename, img_cam)

    # plt.title('Guided-CAM')
    # plt.imshow(overlay(jet_heatmap, img1))
    # fig = plt.gcf()
    # fig.set_size_inches(3, 3)

    # fig.savefig(filename, dpi=100)
    # plt.savefig('Guided_CAM1.png')
    # plt.close()

    return filename

if __name__=='__main__':
    from keras.applications.resnet50 import ResNet50
    from keras.preprocessing import image
    from keras.applications.resnet50 import preprocess_input, decode_predictions
    import numpy as np

    from keras.applications import ResNet50, InceptionV3, Xception
    from vis.utils import utils
    import numpy as np


    model = ResNet50(weights='imagenet')
    # model = InceptionV3(weights='imagenet', include_top=True)
    # model = Xception(weights='imagenet', include_top=True)
    img_path = 'ouzel1.jpg'
    # img = image.load_img(img_path, target_size=(299, 299))
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    preict_class = np.argmax(preds)
    # print('Predicted:', decode_predictions(preds, top=3)[0])

    from keras.applications import ResNet50, InceptionV3, Xception
    from vis.utils import utils
    import numpy as np

    # Build the ResNet50 network with ImageNet weights
    # model = ResNet50(weights='imagenet', include_top=True)
    model = InceptionV3(weights='imagenet', include_top=True)
    # model = Xception(weights='imagenet', include_top=True)

    img1 = utils.load_img('ouzel1.jpg', target_size=(224, 224))

    prob = model.predict(np.expand_dims(img1, axis=0))
    preict_class = np.argmax(prob)

    model.layers[-1].activation = activations.linear
    model = utils.apply_modifications(model)  # saliency maps no much difference


    # from CNN_Models.visualization.my_visualize import get_guided_cam

    filename = get_guided_cam(model, img1)

    print('OK')
