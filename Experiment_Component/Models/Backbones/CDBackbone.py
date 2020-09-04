import tensorflow as tf

from tensorflow.keras.layers import Activation, Conv2DTranspose
from tensorflow.keras.layers.experimental import SyncBatchNormalization

class CDBackbone(tf.keras.Model):
    """
    The CDBackbone creates decoder models based on the given parameters.
    """

    def __init__(self, width_multiplier=1, stem="32"):
        """
        Constructor, initialize member variables.

        :param width_multiplier: (Integer) The multiplier of the model with (filters). 1 by default.
        :param stem: (String) The stem of the backbone to control model and representation sizes. E.g. 32 is for cifar10. "32" by default.
        """
        super(CDBackbone, self).__init__()  

        last_stride = (1, 1)
        last_kernel_size = 3
        layer3_kernel_size = 4
        if stem == "64":
            last_stride = (2, 2)
            last_kernel_size = 4
            layer3_kernel_size = 5

        self.cd = tf.keras.Sequential(
        [
            Conv2DTranspose(filters=128*width_multiplier, kernel_size=4, strides=(2, 2)),
            SyncBatchNormalization(),
            Activation('relu'),
            Conv2DTranspose(filters=64*width_multiplier, kernel_size=4, strides=(2, 2)),
            SyncBatchNormalization(),
            Activation('relu'),
            Conv2DTranspose(filters=32*width_multiplier, kernel_size=layer3_kernel_size, strides=(2, 2)),
            SyncBatchNormalization(),
            Activation('relu'),
            Conv2DTranspose(filters=3, kernel_size=last_kernel_size, strides=last_stride),
        ]
        )

    def __call__(self, data, is_training):
        """
        Call method of the model which processes the input data or features for training or validation

        :param data: (Dictionary) The input data to process.
        :param is_training: (Boolean) If true, the models its training graph (E.g. from batch norm).
        :return: reconstruction: (Tensor) the reconstruction of the input data from the model.
        """
        reconstruction = self.cd(data, training=is_training)
        return reconstruction

    def getReconstructionShape(self):
        """
        This method returns the output shape of the model.

        :return: shape: (Array) The output shape of the model
        """
        return self.cd.output_shape