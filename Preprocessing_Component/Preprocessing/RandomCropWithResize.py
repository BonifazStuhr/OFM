import tensorflow as tf

from Preprocessing_Component.AProcessor import AProcessor
from utils.functions import random_apply

class RandomCropWithResize(AProcessor):
    """
    The RandomCropWithResize crops (and resizes) the given (image) input at a randomy position.

    :Attributes:
        crop_proportion:   (Float) Proportion of image to retain along the less-cropped side. 1.0 by default.
        crop_shape:        (Array) The shape to crop. [1, 32, 32, 3] by default.
    """
    def __init__(self, crop_prop=1.0, crop_shape=[1, 32, 32, 3], input_name="image", output_name="image"):
        """
        Constructor, initialize member variables.

        :param crop_proportion: (Float) Proportion of image to retain along the less-cropped side. 1.0 by default.
        :param crop_shape: (Array) The shape to crop. [1, 32, 32, 3] by default.
        :param input_name: (String) The name of the input to apply this operation. "image" by default.
        :param output_name: (String) The name of the output where this operation was applied. "image" by default.
        """
        super().__init__(input_name=input_name, output_names=[output_name])
        self.crop_prop = crop_prop
        self.crop_shape = crop_shape

    def process(self, input):              
        """
        This function crops the image randomly and rescales to desired size.

        :param input: (Tensor) The input to apply this operation on.
        :return: result: (Tensor) The result of this operation.
        :return: output_names: (Array of Strings) The output names of this operation.
        """
        with tf.name_scope("RandomCropWithResize"):
            input = input[self.input_name]
            height = self.crop_shape[1]
            width = self.crop_shape[2]

            def _transform(image):
                image = self.crop_and_resize(image, height, width)
                return image
            result = random_apply(_transform, p=self.crop_prop, x=input)
            result.set_shape([height, width, self.crop_shape[3]])

            return [result], self.output_names
 

    def crop_and_resize(self, image, height, width):
        """
        Make a random crop and resize it to height `height` and width `width`.

        :param height: (Integer) Desired image height.
        :param width: (Integer) Desired image width.
        :return: (Tensor) The cropped image.
        """
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        aspect_ratio = width / height
        image = self.distorted_bounding_box_crop(
            image,
            bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
            area_range=(0.08, 1.0),
            max_attempts=100)
        return tf.compat.v1.image.resize_bicubic([image], [height, width])[0]

    def distorted_bounding_box_crop(self, image, bbox, min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33),
                                    area_range=(0.05, 1.0), max_attempts=100):
        """
        Generates cropped_image using one of the bboxes randomly distorted.
        See `tf.image.sample_distorted_bounding_box` for more documentation.

        :param image: (Tensor) The image data.
        :param bbox: (Tensor) The bounding boxes arranged `[1, num_boxes, coords]`
                where each coordinate is [0, 1) and the coordinates are arranged
                as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
                image.
        :param min_object_covered: (Float) Optional. The cropped
                area of the image must contain at least this fraction of any bounding
                box supplied. 0.1 by default.
        :param aspect_ratio_range: (List of Float) Optional. The cropped area of the
                image must have an aspect ratio = width / height within this range. (0.75, 1.33) by default.
        :param area_range: (List of Float) Optional. The cropped area of the image
                must contain a fraction of the supplied image within in this range. (0.05, 1.0) by default.
        :param max_attempts: (Integer) Optional. Number of attempts at generating a cropped
                region of the image of the specified constraints. After `max_attempts`
                failures, return the entire image. 100 by default.
        :return: (Tensor) The cropped image.
        """
        with tf.name_scope('distorted_bounding_box_crop'):
            shape = tf.shape(image)
            sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                shape,
                bounding_boxes=bbox,
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
            bbox_begin, bbox_size, _ = sample_distorted_bounding_box

            # Crop the image to the specified bounding box.
            offset_y, offset_x, _ = tf.unstack(bbox_begin)
            target_height, target_width, _ = tf.unstack(bbox_size)
            image = tf.image.crop_to_bounding_box(
                image, offset_y, offset_x, target_height, target_width)

            return image