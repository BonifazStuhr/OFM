import tensorflow as tf

from Preprocessing_Component.AProcessor import AProcessor

class Random90xRotation(AProcessor):
    """
    The Random90xRotation rotates each image in the entire batch randomly by(0, 90, 180 or 270 degree)
    """
    def __init__(self, input_name="image", output_names=["image", "rotation_label"]):
        """
        Constructor, initialize member variables.

        :param input_name: (String) The name of the input to apply this operation. "image" by default.
        :param output_name: (String) The name of the output where this operation was applied. ["image", "rotation_label"] by default.
        """
        super().__init__(input_name=input_name, output_names=output_names)

    def process(self, input):
        """
        The function (or graph part) of the preprocessing.
        This function rotates each image randomly by(0,90,180 or 270 degree)

        :return: result: (Tensor) The result of this operation.
        :return: output_names: (Array of Strings) The output names of this operation.
        """
        with tf.name_scope("Rotate"):
            image = input[self.input_name]

            rotation_label = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            result = tf.image.rot90(image, k=rotation_label)

            rotation_label = tf.one_hot(rotation_label, 4)

            return [result, rotation_label], self.output_names
