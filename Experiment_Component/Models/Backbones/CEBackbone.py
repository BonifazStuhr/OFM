import tensorflow as tf

from tensorflow.keras.layers import InputLayer, Conv2D, Activation
from tensorflow.keras.layers.experimental import SyncBatchNormalization

class CEBackbone(tf.keras.Model):
    """
    The CEBackbone creates encoder models based on the given parameters.
    """

    def __init__(self, input_shape, latent_dim=256, width_multiplier=1, stem="32"):
        """
        Constructor, initialize member variables.

        :param input_shape: (Array) The shape of the input. E.g. [32, 32, 3].
        :param latent_dim: (Integer) The size of the latent representation (last layer). 256 by default.
        :param width_multiplier: (Integer) The multiplier of the model with (filters). 1 by default.
        :param stem: (String) The stem of the backbone to control model and representation sizes. E.g. 32 is for cifar10. "32" by default.
        """
        super(CEBackbone, self).__init__()  

        self.input_s = input_shape

        last_stride = (1, 1)
        kernel_size_add = 0
        if stem == "64":
            kernel_size_add = 1
            last_stride = (2, 2)
        self.ce = tf.keras.Sequential(
        [
            InputLayer(input_shape=input_shape),
            Conv2D(filters=32*width_multiplier, kernel_size=3+kernel_size_add, strides=(2, 2)),
            SyncBatchNormalization(),
            Activation('relu'),
            Conv2D(filters=64*width_multiplier, kernel_size=3+kernel_size_add, strides=(2, 2)),
            SyncBatchNormalization(),
            Activation('relu'),
            Conv2D(filters=128*width_multiplier, kernel_size=3+kernel_size_add, strides=(2, 2)),
            SyncBatchNormalization(),
            Activation('relu'),
            Conv2D(filters=latent_dim, kernel_size=2+kernel_size_add, strides=last_stride),
            SyncBatchNormalization(),
            Activation('relu'),
        ])

    def __call__(self, data, is_training):
        """
        Call method of the model which processes the input data or features for training or validation

        :param data: (Dictionary) The input data to process.
        :param is_training: (Boolean) If true, the models its training graph (E.g. from batch norm).
        :return: representation: (Tensor) the representation of the input data from the model.
        """
        representation = self.ce(data, training=is_training)
        return representation

    def getRepresentationShape(self):
        """
        This method returns the output shape of the model.

        :return: shape: (Array) The output shape of the model
        """
        return self.ce.output_shape