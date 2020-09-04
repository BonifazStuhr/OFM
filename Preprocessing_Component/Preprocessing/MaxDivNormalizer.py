import tensorflow as tf

from Preprocessing_Component.AProcessor import AProcessor

class MaxDivNormalizer(AProcessor):
    """
    The MaxDivNormalizer normalizes by dividing the input trough the given value.

    :Attributes:
        max_value:    (Integer) The max_value of the dataset.
    """
    def __init__(self, max_value, input_name="image", output_name="image"):
        """
        Constructor, initialize member variables.

        :param max_value : The max_value of the dataset.
        :param input_name: (String) The name of the input to apply this operation. "image" by default.
        :param output_name: (String) The name of the output where this operation was applied. "image" by default.
        """
        super().__init__(input_name=input_name, output_names=[output_name])
        self.max_value = max_value

    def process(self, input):
        """
        The function (or graph part) of the preprocessing.
        This function normalizes the input via the max_value.

        :return: result: (Tensor) The result of this operation.
        :return: output_names: (Array of Strings) The output names of this operation.
        """
        with tf.name_scope("NormalizeTrough" + str(self.max_value)):
            input = input[self.input_name]
            result = tf.math.divide_no_nan(tf.cast(input, tf.float32), float(self.max_value))
            return [result], self.output_names
