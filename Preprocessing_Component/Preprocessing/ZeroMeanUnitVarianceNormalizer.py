import tensorflow as tf

from Preprocessing_Component.AProcessor import AProcessor

class ZeroMeanUnitVarianceNormalizer(AProcessor):
    """
    The ZeroMeanUnitVariance normalizes the input for zero mean and unit variance via the given mean and standard
    deviation values of the dataset.

    Mean subtraction is the most common form of preprocessing.
    It involves subtracting the mean across every individual feature in the data, and has the geometric
    interpretation of centering the cloud of data around the origin along every dimension.
    In numpy, this operation would be implemented as: X -= np.mean(X, axis = 0). With images specifically,
    for convenience it can be common to subtract a single value from all pixels (e.g. X -= np.mean(X)),
    or to do so separately across the three color channels.

    Normalization refers to normalizing the data dimensions so that they are of approximately the same scale.
    There are two common ways of achieving this normalization. One is to divide each dimension by its standard
    deviation, once it has been zero-centered: (X /= np.std(X, axis = 0)). Another form of this preprocessing
    normalizes each dimension so that the min and max along the dimension is -1 and 1 respectively.
    It only makes sense to apply this preprocessing if you have a reason to believe that different
    input features have different scales (or units), but they should be of approximately equal importance to the
    learning algorithm. In case of images, the relative scales of pixels are already approximately equal
    (and in range from 0 to 255), so it is not strictly necessary to perform this additional preprocessing step.

    :Attributes:
        __dataset_mean: (Array) The mean values of the dataset for each channel.
        __dataset_std:  (Array) The standard deviation values of the dataset for each channel.
    """

    def __init__(self, dataset_mean=None, dataset_std=None, input_name="image", output_name="image"):
        """
        Constructor, initialize member variables.

        :param dataset_mean: (Array) The mean values of the dataset. Default None.
        :param dataset_variance: (Array) The standard deviation values of the dataset. Default None.
                :param input_name: (String) The name of the input to apply this operation. "image" by default.
        :param output_name: (String) The name of the output where this operation was applied. "image" by default.
        """
        super().__init__(input_name=input_name, output_names=[output_name])
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

    def process(self, input):
        """
        The function (or graph part) of the preprocess.
        This function normalizes the input via the mean and standard deviation for zero mean and unit variance.

        :return: result: (Tensor) The result of this operation.
        :return: output_names: (Array of Strings) The output names of this operation.
        """
        with tf.name_scope("ZeroMeanUnitVariance"):
            input = input[self.input_name]
            input = tf.cast(input, tf.float32)
            input = tf.subtract(input, self.dataset_mean)
            # This function forces Python 3 division operator semantics
            # where all integer arguments are cast to floating types first.
            result = tf.math.divide_no_nan(input, self.dataset_std)
            return [result], self.output_names

