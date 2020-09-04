import tensorflow as tf

from tensorflow.keras.layers import Activation, Conv2DTranspose, InputLayer, Reshape
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.initializers import VarianceScaling

class CDResNetBackbone(tf.keras.Model):
    """
    The CDBackbone creates decoder models based on the given parameters for the ResNet.
    """

    def __init__(self, width_multiplier=1, stem="32", weight_decay=1e-4):
        """
        Constructor, initialize member variables.

        :param width_multiplier: (Integer) The multiplier of the model with (filters). 1 by default.
        :param stem: (String) The stem of the backbone to control model and representation sizes. E.g. 32 is for cifar10. "32" by default.
        :param weight_decay: (Float) The weight decay for the l2 regularitation loss. 1e-4 by default.
        """
        super(CDResNetBackbone, self).__init__()

        self.weight_decay = weight_decay

        self.cd = tf.keras.Sequential(
            [
            Reshape(target_shape=(1, 1, -1)),
            Conv2DTranspose(filters=256*width_multiplier, kernel_size=3,strides=(2, 2), padding="SAME", kernel_initializer=VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), bias_regularizer=tf.keras.regularizers.l2(self.weight_decay)),
            SyncBatchNormalization(),
            Activation('relu'),
            Conv2DTranspose(filters=128*width_multiplier, kernel_size=3,strides=(2, 2), padding="SAME", kernel_initializer=VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), bias_regularizer=tf.keras.regularizers.l2(self.weight_decay)),
            SyncBatchNormalization(),
            Activation('relu'),
            Conv2DTranspose(filters=64*width_multiplier, kernel_size=3,strides=(2, 2), padding="SAME", kernel_initializer=VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), bias_regularizer=tf.keras.regularizers.l2(self.weight_decay)),
            SyncBatchNormalization(),
            Activation('relu'),
            Conv2DTranspose(filters=32*width_multiplier, kernel_size=3,strides=(2, 2), padding="SAME", kernel_initializer=VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), bias_regularizer=tf.keras.regularizers.l2(self.weight_decay)),
            SyncBatchNormalization(),
            Activation('relu'),
            Conv2DTranspose(filters=16*width_multiplier, kernel_size=3,strides=(2, 2), padding="SAME", kernel_initializer=VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), bias_regularizer=tf.keras.regularizers.l2(self.weight_decay)),
            SyncBatchNormalization(),
            Activation('relu'),
            Conv2DTranspose(filters=3*width_multiplier, kernel_size=3, strides=(1, 1), padding="SAME", kernel_initializer=VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), bias_regularizer=tf.keras.regularizers.l2(self.weight_decay)),
            ])

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