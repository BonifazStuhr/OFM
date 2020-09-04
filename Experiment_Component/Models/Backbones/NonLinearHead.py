import tensorflow as tf

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers.experimental import SyncBatchNormalization

class NonLinearHead(tf.keras.Model):
    """
    The NonLinearHead creates a fc NonLinear Head. E.g. for contrastive learning.
    """

    def __init__(self, input_shape, feature_dim=128):
        """
        Constructor, initialize member variables.

        :param input_shape: (Array) The shape of the input. E.g. [32, 32, 3].
        :param feature_dim: (Integer) The size of the latent representation. 128 by default.
        """

        super(NonLinearHead, self).__init__()

        self.head = tf.keras.Sequential(
        [
            Dense(input_shape),
            SyncBatchNormalization(),
            Activation("relu"),
            Dense(feature_dim),
            SyncBatchNormalization()
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