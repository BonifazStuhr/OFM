import tensorflow as tf

from Preprocessing_Component.AProcessor import AProcessor

from utils.functions import random_apply

class RandomNormalNoise(AProcessor):
    """
    The RandomNormalNoise adds normal distributed noise to the input. 

    :Attributes:
        noise_mean:    (Float) The mean of the noise. 0.5 by default.
        noise_stddev:  (Float) The stddev of the output. 0.5 by default.
        min_range:     (Float) The min value of the output. 0 by default.
        max_range:     (Float) The max value of the output. 1 by default.
    """
    def __init__(self, noise_mean=0, noise_stddev=0.1, min_range=0, max_range=1, noise_prop=0.99, input_name="image", output_name="noisy_image"):
        """
        Constructor, initialize member variables.

        :param noise_mean: (Float) The mean of the noise. 0.5 by default.
        :param noise_stddev: (Float) The stddev of the output. 0.5 by default.
        :param min_range: (Float) The min value of the output. 0 by default.
        :param max_range: (Float) The max value of the output. 1 by default.
        :param noise_prop: (Float) The probability of applying the noise. 0.95 by default.
        :param input_name: (String) The name of the input to apply this operation. "image" by default.
        :param output_name: (String) The name of the output where this operation was applied. "noisy_image" by default.
        """
        super().__init__(input_name=input_name, output_names=[output_name])
        self.min_range = min_range
        self.max_range = max_range
        self.noise_prop = noise_prop
        self.noise_mean = noise_mean
        self.noise_stddev = noise_stddev

    def process(self, input):
        """
        The function (or graph part) of the preprocessing.
        This function adds noise to the input.

        :return: result: (Tensor) The result of this operation.
        :return: output_names: (Array of Strings) The output names of this operation.
        """
        with tf.name_scope("RandomNormalNoise" + str(self.noise_stddev)):
            input = input[self.input_name]

            def _transform(image):
                train_noise = tf.random.normal(image.shape, mean=self.noise_mean, stddev=self.noise_stddev)
                noisy_image = tf.add(image, train_noise)
                return noisy_image

            result = random_apply(_transform, p=self.noise_prop, x=input)
            result = tf.clip_by_value(result, self.min_range, self.max_range)

            return [result], self.output_names


