import tensorflow as tf

from Preprocessing_Component.AProcessor import AProcessor

class OneHot(AProcessor):
    """
    The OneHot encodes class labels wie one hot encoding.

    :Attributes:
        num_classes:    (Integer) The number of classes of the dataset.
    """
    def __init__(self, num_classes, input_name="label", output_name="label"):
        """
        Constructor, initialize member variables.

        :param num_classes : The number of classes of the dataset.
        :param input_name: (String) The name of the input to apply this operation. "label" by default.
        :param output_name: (String) The name of the output where this operation was applied. "label" by default.
        """
        super().__init__(input_name=input_name, output_names=[output_name])
        self.num_classes = num_classes

    def process(self, input):
        """
        The function (or graph part) of the preprocessing.
        This function encodes class labels wie one hot encoding.

        :return: result: (Tensor) The result of this operation.
        :return: output_names: (Array of Strings) The output names of this operation.
        """
        with tf.name_scope("OneHot"):
            input = input[self.input_name]
            result = tf.one_hot(input, self.num_classes)
            return [result], self.output_names
