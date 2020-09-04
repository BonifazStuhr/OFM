import tensorflow as tf

from Preprocessing_Component.AProcessor import AProcessor

class CenterCropWithResize(AProcessor):
    """
    The CenterCropWithResize crops (and resizes) the given (image) input at the center.

    :Attributes:
        crop_proportion:   (Float) Proportion of image to retain along the less-cropped side. 0.875 by default.
        crop_shape:        (Array) The shape to crop. [1, 32, 32, 3] by default.
    """
    def __init__(self, crop_proportion=0.875, crop_shape=[1, 32, 32, 3], input_name="image", output_name="image"):
        """
        Constructor, initialize member variables.
        :param crop_proportion: (Float) Proportion of image to retain along the less-cropped side. 0.875 by default.
        :param crop_shape: (Array) The shape to crop. [1, 32, 32, 3] by default.
        :param input_name: (String) The name of the input to apply this operation. "image" by default.
        :param output_name: (String) The name of the output where this operation was applied. "image" by default.
        """
        super().__init__(input_name=input_name, output_names=[output_name])
        self.crop_proportion = crop_proportion
        self.crop_shape = crop_shape

    def process(self, input):
        """
        This function crops the center of image and rescales to desired size.

        :param input: (Tensor) The input to apply this operation on.
        :return: result: (Tensor) The result of this operation.
        :return: output_names: (Array of Strings) The output names of this operation.
        """
        with tf.name_scope("CenterCropWithResize"):
            input = input[self.input_name]

            shape = tf.shape(input)
            image_height = shape[0]
            image_width = shape[1]
            height = self.crop_shape[1]
            width = self.crop_shape[2]

            crop_height, crop_width = self.compute_crop_shape(image_height, image_width, height / width, self.crop_proportion)

            offset_height = ((image_height - crop_height) + 1) // 2
            offset_width = ((image_width - crop_width) + 1) // 2

            result = tf.image.crop_to_bounding_box(input, offset_height, offset_width, crop_height, crop_width)
            result = tf.compat.v1.image.resize_bicubic([result], [height, width])[0]

            return [result], self.output_names
 

    def compute_crop_shape(self, image_height, image_width, aspect_ratio, crop_proportion):
        """
        Compute aspect ratio-preserving shape for central crop.
        The resulting shape retains `crop_proportion` along one side and a proportion
        less than or equal to `crop_proportion` along the other side.

        :param image_height: (Integer) Height of image to be cropped.
        :param image_width: (Integer) Width of image to be cropped.
        :param aspect_ratio: (Float) Desired aspect ratio (width / height) of output.
        :param crop_proportion: (Float) Proportion of image to retain along the less-cropped side.

        :return: (Integer) crop_height: Height of image after cropping.
        :return: (Integer) crop_width: Width of image after cropping.
        """
        image_width_float = tf.cast(image_width, tf.float32)
        image_height_float = tf.cast(image_height, tf.float32)

        def _requested_aspect_ratio_wider_than_image():
            crop_height = tf.cast(tf.math.rint(
                crop_proportion / aspect_ratio * image_width_float), tf.int32)
            crop_width = tf.cast(tf.math.rint(
                crop_proportion * image_width_float), tf.int32)
            return crop_height, crop_width

        def _image_wider_than_requested_aspect_ratio():
            crop_height = tf.cast(
                tf.math.rint(crop_proportion * image_height_float), tf.int32)
            crop_width = tf.cast(tf.math.rint(
                crop_proportion * aspect_ratio *
                image_height_float), tf.int32)
            return crop_height, crop_width

        return tf.cond(
            aspect_ratio > image_width_float / image_height_float,
            _requested_aspect_ratio_wider_than_image,
            _image_wider_than_requested_aspect_ratio)