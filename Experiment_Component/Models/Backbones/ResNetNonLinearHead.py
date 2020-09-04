import tensorflow as tf

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.initializers import Ones

class ResNetNonLinearHead(tf.keras.Model):
    """
    The NonLinearHead creates a fc NonLinear Head for ResNets. E.g. for contrastive learning.
    """

    def __init__(self, input_shape, feature_dim=128, weight_decay=1e-4):
        """
        Constructor, initialize member variables.

        :param input_shape: (Array) The shape of the input. E.g. [32, 32, 3].
        :param feature_dim: (Integer) The size of the latent representation. 128 by default.
        :param weight_decay: (Float) The weight decay for the l2 regularitation loss. 1e-4 by default.
        """
        super(ResNetNonLinearHead, self).__init__()

        self.batch_norm_decay = 0.9
        self.batch_norm_epsilon = 1e-5
        self.weight_decay = weight_decay

        self.head = tf.keras.Sequential(
            [
            Dense(input_shape, use_bias=False, kernel_initializer=RandomNormal(stddev=.01),
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), bias_regularizer=tf.keras.regularizers.l2(self.weight_decay)),
            SyncBatchNormalization(center=True, scale=True, momentum=self.batch_norm_decay,
                                       epsilon=self.batch_norm_epsilon, gamma_initializer=Ones()),
            Activation("relu"),
            Dense(feature_dim, use_bias=False, kernel_initializer=RandomNormal(stddev=.01),
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay), bias_regularizer=tf.keras.regularizers.l2(self.weight_decay)),
            SyncBatchNormalization(center=False, scale=True, momentum=self.batch_norm_decay,
                                    epsilon=self.batch_norm_epsilon, gamma_initializer=Ones())
            ])

    def __call__(self, data, is_training):
        """
        Call method of the model which processes the input data or features for training or validation

        :param data: (Dictionary) The input data to process.
        :param is_training: (Boolean) If true, the models its training graph (E.g. from batch norm).
        :return: representation: (Tensor) the representation of the input data from the model.
        """
        representation = self.head(data, training=is_training)
        return representation

    def getRepresentationShape(self):
        """
        This method returns the output shape of the model.

        :return: shape: (Array) The output shape of the model
        """
        return self.head.output_shape