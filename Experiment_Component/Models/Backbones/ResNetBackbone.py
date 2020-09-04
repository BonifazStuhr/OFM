import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Activation, ZeroPadding2D, Dense, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.initializers import VarianceScaling, Ones, Zeros


class ResNetBackbone(tf.keras.Model):
    """
    The CDBackbone creates ResNet models based on the given parameters.
    """

    def __init__(self, model_config, input_shape, width_multiplier=1, stem="32", weight_decay=1e-4, latent_dim=None):
        """
        Constructor, initialize member variables.

        :param model_config: (Dictionary) The configuration of the model, containing layers specifications, learning rates, etc.
        :param input_shape: (Array) The shape of the input. E.g. [32, 32, 3].
        :param width_multiplier: (Integer) The multiplier of the model with (filters). 1 by default.
        :param stem: (String) The stem of the backbone to control model and representation sizes. E.g. 32 is for cifar10. "32" by default.
        :param weight_decay: (Float) The weight decay for the l2 regularitation loss. 1e-4 by default.
        :param latent_dim: (Integer) The size of the latent representation (last layer). None by default.
        """
        super(ResNetBackbone, self).__init__()  

        self.input_s = input_shape
        self.batch_norm_decay = 0.9
        self.batch_norm_epsilon = 1e-5
        self.weight_decay = weight_decay
        self.latent_dim = latent_dim

        self.stem = stem

        self.width_multiplier = width_multiplier

        self.classes = 0
        self.include_top = 0
        self.resNet_depth = model_config["resNetDepth"]

        layer_configuarations = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
            200: [3, 24, 36, 3]
        }

        if self.resNet_depth not in layer_configuarations:
            raise ValueError('Not a valid resNet_depth:', self.resNet_depth)

        if self.resNet_depth > 34:
            block = self.bottleneck_block
        else:
            block = self.residual_block     

        layers = layer_configuarations[self.resNet_depth]
        self.resNet = self.resnet_v1_generator(input_shape, block, layers, self.width_multiplier)

    def __call__(self, data, is_training):
        """
        Call method of the model which processes the input data or features for training or validation

        :param data: (Dictionary) The input data to process.
        :param is_training: (Boolean) If true, the models its training graph (E.g. from batch norm).
        :return: representation: (Tensor) the representation of the input data from the model.
        """
        representation = self.resNet(data, training=is_training)
        return representation

    def getRepresentationShape(self):
        """
        This method returns the output shape of the model.

        :return: shape: (Array) The output shape of the model
        """
        return self.resNet.output_shape

    def resnet_v1_generator(self, input_shape, block_fn, layers, width_multiplier, data_format='channels_last'):
        """
        Generates the ResNet model.

        :param input_shape: (Array) The shape of the input. E.g. [32, 32, 3].
        :param block_fn: (Function) The function to generate one ResNet block.
        :param layers: (Array) The number of layers per block.
        :param width_multiplier: (Integer) The multiplier of the model with (filters). 1 by default.
        :param data_format: (Integer) The latent dim of the last layer. 'channels_last' by default.
        :return: output (tf.keras.Model) The generated ResNet.
        """

        img_input = Input(shape=input_shape)

        if (self.stem == "32") or (self.stem == "64"):
            stride = 1
            if self.stem == "64":
                stride = 2

            inputs = self.conv2d_fixed_padding(
                inputs=img_input, filters=64 * width_multiplier, kernel_size=3,
                strides=stride, data_format=data_format)
            inputs = tf.identity(inputs, 'initial_conv')
            inputs = self.batch_norm_relu(inputs, data_format=data_format)
            inputs = tf.identity(inputs, 'initial_max_pool')
        else:
            inputs = self.conv2d_fixed_padding(
                inputs=img_input, filters=64 * width_multiplier, kernel_size=7,
                strides=2, data_format=data_format)
            inputs = tf.identity(inputs, 'initial_conv')
            inputs = self.batch_norm_relu(inputs, data_format=data_format)

            inputs = MaxPooling2D(pool_size=3, strides=2, padding="SAME", data_format=data_format)(inputs)     

            inputs = tf.identity(inputs, 'initial_max_pool')
        print(inputs)
        inputs = self.block_group(inputs=inputs, filters=64 * width_multiplier, block_fn=block_fn,
                                  blocks=layers[0], strides=1, name='block_group1', data_format=data_format)
        print(inputs)
        inputs = self.block_group(inputs=inputs, filters=128 * width_multiplier, block_fn=block_fn,
                                  blocks=layers[1], strides=2, name='block_group2', data_format=data_format)
        print(inputs)
        inputs = self.block_group(inputs=inputs, filters=256 * width_multiplier, block_fn=block_fn,
                                  blocks=layers[2], strides=2, name='block_group3', data_format=data_format)
        print(inputs)
        inputs = self.block_group(inputs=inputs, filters=512 * width_multiplier, block_fn=block_fn,
                                  blocks=layers[3], strides=2, name='block_group4', data_format=data_format)
        print(inputs)
                                  
        if self.latent_dim:
            inputs = self.conv2d_fixed_padding(inputs=inputs, filters=self.latent_dim, kernel_size=1, strides=1, data_format=data_format)
            inputs = self.batch_norm_relu(inputs, relu=True, data_format=data_format)
            print(inputs)

        inputs = GlobalAveragePooling2D(name='avg_pool')(inputs)
        if self.include_top:
            inputs = Dense(self.classes, activation='softmax', name='fcX')(inputs)
        print(inputs)
        resNet = tf.keras.Model(img_input, inputs, name='resNet')
        print(resNet)

        return resNet

    def block_group(self, inputs, filters, block_fn, blocks, strides, name, data_format='channels_last'):
        """
        Creates one group of blocks for the ResNet model.

        :param inputs: (Tensor) The input of size `[batch, channels, height, width]`.
        :param filters: (Integer) the number of filters for the first convolution of the layer.
        :param block_fn: (Function) The function for the block to use within the model
        :param blocks: (Integer) The number of blocks contained in the layer.
        :param strides: (Integer) The stride to use for the first convolution of the layer. If
                greater than 1, this layer will downsample the input.
        :param name: (String) The name for the Tensor output of the block layer.
        :param data_format: (String) either "channels_first" for `[batch, channels, height,
                width]` or "channels_last for `[batch, height, width, channels]`. 'channels_last'
        :param data_format: (Integer) The latent dim of the last layer. None by default.
        :return: output (Tensor) The output `Tensor` of the block layer.
        """
        # Only the first block per block_group uses projection shortcut and strides.
        inputs = block_fn(inputs, filters, strides, use_projection=True, data_format=data_format)

        for _ in range(1, blocks):
            inputs = block_fn(inputs, filters, 1, data_format=data_format)

        return tf.identity(inputs, name)

    def residual_block(self, inputs, filters, strides, use_projection=False, data_format='channels_last'):
        """
        Standard building block for residual networks with BN after convolutions.

        :param inputs: (Tensor) The input of size `[batch, channels, height, width]`.
        :param filters: (Integer) The number of filters for the first two convolutions. Note that
                the third and final convolution will use 4 times as many filters.
        :param strides: (Integer) The block stride. If greater than 1, this block will ultimately
                downsample the input.
        :param use_projection: (Boolean) Whether this block should use a projection
                shortcut (versus the default identity shortcut). This is usually `True`
                for the first block of a block group, which may change the number of
                filters and the resolution.
        :param  data_format: `str` either "channels_first" for `[batch, channels, height,
                width]` or "channels_last for `[batch, height, width, channels]`.
        :return: output: The output `Tensor` of the block.
        """
        shortcut = inputs
        if use_projection:
            # Projection shortcut in first layer to match filters and strides
            shortcut = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=strides, data_format=data_format)
            shortcut = self.batch_norm_relu(shortcut, relu=False, data_format=data_format)

        inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
        inputs = self.batch_norm_relu(inputs, data_format=data_format)

        inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1, data_format=data_format)
        inputs = self.batch_norm_relu(inputs, relu=False, init_zero=True, data_format=data_format)

        return Activation('relu')(inputs + shortcut)


    def bottleneck_block(self, inputs, filters, strides, use_projection=False, data_format='channels_last',):
        """
        Bottleneck block variant for residual networks with BN after convolutions.

        :param inputs: (Tensor) The input of size `[batch, channels, height, width]`.
        :param filters: (Integer) The number of filters for the first two convolutions. Note that
                the third and final convolution will use 4 times as many filters.
        :param strides: (Integer) The block stride. If greater than 1, this block will ultimately
                downsample the input.
        :param use_projection: (Boolean) Whether this block should use a projection
                shortcut (versus the default identity shortcut). This is usually `True`
                for the first block of a block group, which may change the number of
                filters and the resolution.
        :param data_format: (String) Either "channels_first" for `[batch, channels, height,
                width]` or "channels_last for `[batch, height, width, channels]`.
        :return: output: (Tensor) The output `Tensor` of the block.
        """
        shortcut = inputs
        if use_projection:
            # Projection shortcut only in first block within a group. Bottleneck blocks
            # end with 4 times the number of filters.
            filters_out = 4 * filters
            shortcut = self.conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=strides, data_format=data_format)
            shortcut = self.batch_norm_relu(shortcut, relu=False, data_format=data_format)

        inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=1, data_format=data_format)
        inputs = self.batch_norm_relu(inputs, data_format=data_format)

        inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
        inputs = self.batch_norm_relu(inputs, data_format=data_format)

        inputs = self.conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1, data_format=data_format)
        inputs = self.batch_norm_relu(inputs, relu=False, init_zero=True, data_format=data_format)

        return Activation('relu')(inputs + shortcut)

    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, data_format='channels_last'):
        """
        Strided 2-D convolution with explicit padding.
        The padding is consistent and is based only on `kernel_size`, not on the
        dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

        :param inputs: (Tensor) The input of size `[batch, channels, height_in, width_in]`.
        :param filters: (Integer) The number of filters in the convolution.
        :param kernel_size: (Integer) The size of the kernel to be used in the convolution.
        :param strides: (Integer) The strides of the convolution.
        :param data_format: (String) Either "channels_first" for `[batch, channels, height,
                width]` or "channels_last for `[batch, height, width, channels]`.
        :return: output: (tensor) A `Tensor` of shape `[batch, filters, height_out, width_out]`.
        """
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size, data_format=data_format)
        
        return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=VarianceScaling(),
            data_format=data_format, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                     bias_regularizer=tf.keras.regularizers.l2(self.weight_decay))(inputs)

    def fixed_padding(self, inputs, kernel_size, data_format='channels_last'):
        """
        Pads the input along the spatial dimensions independently of input size.

        :param inputs: (Tensor) The input of size `[batch, channels, height, width]` or
                `[batch, height, width, channels]` depending on `data_format`.
        :param kernel_size: (Integer) The kernel size to be used for `conv2d` or max_pool2d`
                operations. Should be a positive integer.
        :param data_format: (String) Either "channels_first" for `[batch, channels, height,
                width]` or "channels_last for `[batch, height, width, channels]`.
        :return: output (Tensor) A padded `Tensor` of the same `data_format` with size either intact
            (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
        """
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = ZeroPadding2D(padding=(pad_beg, pad_end), data_format=data_format)(inputs)
        return padded_inputs

    def batch_norm_relu(self, inputs, relu=True, init_zero=False,
                    center=True, scale=True, data_format='channels_last'):
        """
        Performs a batch normalization followed by a ReLU.

        :param inputs: (Tensor) The input of size `[batch, channels, ...]`.
        :param relu: (Boolean) If False, omits the ReLU operation.
        :param init_zero: (Boolean) If True, initializes scale parameter of batch
                normalization with 0 instead of 1 (default).
        :param center: (Boolean) Whether to add learnable bias factor.
        :param scale:  (Boolean) Whether to add learnable scaling factor.
        :param data_format: (String) Either "channels_first" for `[batch, channels, height,
                width]` or "channels_last for `[batch, height, width, channels]`.
        :return: output: A normalized `Tensor` with the same `data_format`.
        """
        if init_zero:
            gamma_initializer = Zeros()
        else:
            gamma_initializer = Ones()

        if data_format == 'channels_first':
            axis = 1
        else:
            axis = 3

        inputs = SyncBatchNormalization(axis=axis, center=center, scale=scale, momentum=self.batch_norm_decay,
                                   epsilon=self.batch_norm_epsilon, gamma_initializer=gamma_initializer)(inputs)
        if relu:
            inputs = Activation('relu')(inputs)

        return inputs