import functools
import tensorflow as tf

from Preprocessing_Component.AProcessor import AProcessor
from utils.functions import random_apply

class RandomColorJitter(AProcessor):
    """
    The RandomColorJitter applies augmentations of the color to the input: grayscale, brightness, contrast, saturation,
    hue.

    :Attributes:
        jitter_prop:              (Float) The probability og applying the color_jitter. 1.0 by default.
        color_jitter_strength:    (Float) The strength of the color augmentation. 0.5 by default.
    """
    def __init__(self, jitter_prop=1.0, color_jitter_strength=0.5, input_name="image", output_name="image"):
        """
        Constructor, initialize member variables.

        :param jitter_prop: (Float) The probability of applying the color_jitter. 1.0 by default.
        :param color_jitter_strength: (Float) The strength of the color augmentation. 0.5 by default.
        :param input_name: (String) The name of the input to apply this operation. "image" by default.
        :param output_name: (String) The name of the output where this operation was applied. "jitter_image" by default.
        """
        super().__init__(input_name=input_name, output_names=[output_name])
        self.jitter_prop = jitter_prop
        self.color_jitter_strength = color_jitter_strength

    def process(self, input):
        """
        The function (or graph part) of the preprocessing.
        This function applies augmentations of color to the given image.

        :return: result: (Tensor) The result of this operation.
        :return: output_names: (Array of Strings) The output names of this operation.
        """
        with tf.name_scope("RandomColorJitter"):
            image = input[self.input_name]

            def _transform(image):
                color_jitter_t = functools.partial(self.color_jitter, strength=self.color_jitter_strength)
                image = random_apply(color_jitter_t, p=0.8, x=image)
                return random_apply(self.to_grayscale, p=0.2, x=image)
            result = random_apply(_transform, p=self.jitter_prop, x=image)

            return [result], self.output_names
 
    def to_grayscale(self, image, keep_channels=True):
        """
        Converts the image to grayscale.

        :param image: (Tensor) The input image tensor.
        :param keep_channels: (Boolean) If true, the channels will stay the same.
        :return: grayscale_image: (Tensor) The distorted image tensor.
        """
        image = tf.image.rgb_to_grayscale(image)
        if keep_channels:
            image = tf.tile(image, [1, 1, 3])
        return image 

    def color_jitter(self, image, strength, random_order=True):
        """
        Distorts the color of the image.

        :param image: (Tensor) The input image tensor.
        :param strength: (Float) The strength of the color augmentation.
        :param random_order: (Boolean) Specifying whether to randomize the jittering order.
        :return: distorted_image: (Tensor) The distorted image tensor.
        """
        brightness = 0.8 * strength
        contrast = 0.8 * strength
        saturation = 0.8 * strength
        hue = 0.2 * strength
        if random_order:
            return self.color_jitter_rand(image, brightness, contrast, saturation, hue)
        else:
            return self.color_jitter_nonrand(image, brightness, contrast, saturation, hue)    

    def color_jitter_nonrand(self, image, brightness=0, contrast=0, saturation=0, hue=0):
        """Distorts the color of the image (jittering order is fixed).

        :param image: (Tensor) The input image tensor.
        :param brightness: (Float) Specifying the brightness for color jitter.
        :param contrast: (Float) Specifying the contrast for color jitter.
        :param saturation: (Float) Specifying the saturation for color jitter.
        :param hue: (Float) Specifying the hue for color jitter.
        :return: distorted_image: (Tensor) The distorted image tensor.
        """
        with tf.name_scope('distort_color'):
            def apply_transform(i, x, brightness, contrast, saturation, hue):
                """Apply the i-th transformation."""
                if brightness != 0 and i == 0:
                    x = tf.image.random_brightness(x, max_delta=brightness)
                elif contrast != 0 and i == 1:
                    x = tf.image.random_contrast(
                        x, lower=1-contrast, upper=1+contrast)
                elif saturation != 0 and i == 2:
                    x = tf.image.random_saturation(
                        x, lower=1-saturation, upper=1+saturation)
                elif hue != 0:
                    x = tf.image.random_hue(x, max_delta=hue)
                return x

            for i in range(4):
                image = apply_transform(i, image, brightness, contrast, saturation, hue)
                image = tf.clip_by_value(image, 0., 1.)
            return image


    def color_jitter_rand(self, image, brightness=0, contrast=0, saturation=0, hue=0):
        """
        Distorts the color of the image (jittering order is random).

        :param image: (Tensor) The input image tensor.
        :param brightness: (Float) Specifying the brightness for color jitter.
        :param contrast: (Float) Specifying the contrast for color jitter.
        :param saturation: (Float) Specifying the saturation for color jitter.
        :param hue: (Float) Specifying the hue for color jitter.
        :return: distorted_image: (Tensor) The distorted image tensor.
        """
        with tf.name_scope('distort_color'):
            def apply_transform(i, x):
                """Apply the i-th transformation."""
                def brightness_foo():
                    if brightness == 0:
                        return x
                    else:
                        return tf.image.random_brightness(x, max_delta=brightness)
                def contrast_foo():
                    if contrast == 0:
                        return x
                    else:
                        return tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
                def saturation_foo():
                    if saturation == 0:
                        return x
                    else:
                        return tf.image.random_saturation(x, lower=1-saturation, upper=1+saturation)
                def hue_foo():
                    if hue == 0:
                        return x
                    else: 
                        return tf.image.random_hue(x, max_delta=hue)
                x = tf.cond(tf.less(i, 2),
                            lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                            lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
                return x

            perm = tf.random.shuffle(tf.range(4))
            for i in range(4):
                image = apply_transform(perm[i], image)
                image = tf.clip_by_value(image, 0., 1.)
            return image