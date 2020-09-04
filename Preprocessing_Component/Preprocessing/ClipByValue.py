import tensorflow as tf

from Preprocessing_Component.AProcessor import AProcessor

class ClipByValue(AProcessor):
    """
    The ClipByValue clips the input values to the given range.

    :Attributes:
        min_value:    (Integer) The min_value of the dataset.
        max_value:    (Integer) The max_value of the dataset.
    """
    def __init__(self, min_value=0.0, max_value=1.0, input_name="image", output_name="image"):
        """
        Constructor, initialize member variables.

        :param max_value : The allowed min_value.
        :param max_value : The allowed max_value.
        :param input_name: (String) The name of the input to apply this operation. "image" by default.
        :param output_name: (String) The name of the output where this operation was applied. "image" by default.
        """
        super().__init__(input_name=input_name, output_names=[output_name])
        self.min_value = min_value
        self.max_value = max_value

    def process(self, input):
        """
        The function (or graph part) of the preprocessing.
        This function clips  the input via the min_value and max_value.

        :return: result: (Tensor) The result of this operation.
        :return: output_names: (Array of Strings) The output names of this operation.
        """
        with tf.name_scope("Clip"):
            input = input[self.input_name]
            result = tf.clip_by_value(input, self.min_value, self.max_value)
            return [result], self.output_names
