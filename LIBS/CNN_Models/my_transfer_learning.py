
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D, GlobalAveragePooling3D, AveragePooling3D, Flatten

'''
average_pooling2d_1
flatten_1 (Flatten) 
dense_1 (Dense)

GlobalAveragePooling2D
dense
'''

def add_top(model_input, num_output, dim='2D',activation_function='SoftMax'):

    x = model_input.output
    if dim == '3D':
        gap = GlobalAveragePooling3D()(x)
    else:
        gap = GlobalAveragePooling2D()(x)

    assert activation_function in ['SoftMax', 'Sigmoid', 'Regression'], \
        'activation_function type is error'

    if activation_function == 'SoftMax':
        predict = Dense(units=num_output, activation='softmax')(gap)
    elif activation_function == 'Sigmoid':
        predict = Dense(units=num_output, activation='sigmoid')(gap)
    else:
        predict = Dense(units=num_output)(gap)

    model1 = Model(inputs=model_input.input, outputs=predict)

    return model1


def convert_model_transfer(model_input, change_top=False, clsss_num=2,
                   activation_function='SoftMax', freeze_feature_extractor=True,
                   freeze_layes_num=None):

    if isinstance(model_input, str):
        model1 = keras.models.load_model(model_input, compile=False)
    else:
        model1 = model_input

    is_avgpool = False  # add top different, Flatten
    layer_num_GAP = 0

    for i in range(len(model1.layers)-1, -1, -1):
        if isinstance(model1.layers[i], GlobalAveragePooling2D) or\
                isinstance(model1.layers[i], GlobalAveragePooling3D):
            layer_num_GAP = i
            break
        if isinstance(model1.layers[i], AveragePooling2D) or \
                isinstance(model1.layers[i], AveragePooling3D):
            layer_num_GAP = i
            is_avgpool = True
            break

    if freeze_feature_extractor:
        if freeze_layes_num is None:
            freeze_border = layer_num_GAP
        else:
            freeze_border = freeze_layes_num
            # freeze_border = len(model1.layers) - freeze_layes_num
            # freeze_border = int(len(model1.layers) * freeze_layes_num)

        for layer in model1.layers[:freeze_border]:
            layer.trainable = False

    if change_top:
        x = model1.layers[layer_num_GAP].output

        if is_avgpool:
            x = Flatten()(x)

        assert activation_function in ['SoftMax', 'Sigmoid', 'Regression'], \
            'activation_function type is error'

        if activation_function == 'SoftMax':
            predictions = Dense(clsss_num, activation="softmax")(x)
        elif activation_function == 'Sigmoid':
            predictions = Dense(clsss_num, activation="sigmoid")(x)
        else:
            predictions = Dense(clsss_num)(x)

        model_changed = Model(inputs=model1.input, outputs=predictions)

        return model_changed
    else:
        return model1


def convert_trainable_all(model_input):
    if isinstance(model_input, str):
        model1 = keras.models.load_model(model_input, compile=False)
    else:
        model1 = model_input

    for layer in model1.layers:
        if not layer.trainable:
            layer.trainable = True

    return model1



def convert_to_regression(model_input, num_output=2):
    if isinstance(model_input, str):
        model1 = keras.models.load_model(model_input, compile=False)
    else:
        model1 = model_input

    for i in range(len(model1.layers) - 1, -1, -1):
        if isinstance(model1.layers[i], GlobalAveragePooling2D) or \
                isinstance(model1.layers[i], AveragePooling2D):
            layer_avg_pooling = i
            break

    layer_before_avg_pooling = model1.layers[layer_avg_pooling - 1]
    flatten1 = Flatten()(layer_before_avg_pooling.output)
    dense = Dense(units=num_output)(flatten1)

    model_regression = Model(inputs=model1.input, outputs=dense)

    return model_regression