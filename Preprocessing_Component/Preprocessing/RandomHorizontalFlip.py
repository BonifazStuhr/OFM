import tensorflow as tf

from Preprocessing_Component.AProcessor import AProcessor

class RandomHorizontalFlip(AProcessor):
    """
    The RandomHorizontalFlip randomly flips the input horizontal.
    """
    def __init__(self, input_name="image", output_name="image"):
        """
        Constructor, initialize member variables.

        :param input_name: (String) The name of the input to apply this operation. "image" by default.
        :param output_name: (String) The name of the output where this operation was applied. "image" by default.
        """
        super().__init__(input_name=input_name, output_names=[output_name])

    def process(self, input):
        """
        The function (or graph part) of the preprocess.
        This function randomly flips the input horizontal.

        :return: result: (Tensor) The result of this operation.
        :return: output_names: (Array of Strings) The output names of this operation.
        """
        with tf.name_scope("RandomHorizontalFlip"):
            input = input[self.input_name]
            result = tf.expand_dims(input, axis=0)
            result = tf.image.random_flip_left_right(result)
            result = tf.reshape(result, result.shape[1:])
            return [result], self.output_names
