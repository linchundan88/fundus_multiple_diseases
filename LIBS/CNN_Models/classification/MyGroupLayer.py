from keras import backend as K
from keras.engine.topology import Layer

class MyLayer(Layer):
    def __init__(self, c, grouped_channels, **kwargs):
        self.c = c
        self.grouped_channels = grouped_channels

        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x[:, :, :,
                   self.c * self.grouped_channels:(self.c + 1) * self.grouped_channels]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.grouped_channels)

    def get_config(self):
        config = {'c': self.c,
                  'grouped_channels': self.grouped_channels}
        return dict(list(config.items()))
