import tensorflow as tf

from tensorflow.keras.layers import Activation

from Experiment_Component.Models.Backbones.CEBackbone import CEBackbone
from Experiment_Component.Models.Backbones.CDBackbone import CDBackbone
from Experiment_Component.Models.Backbones.ResNetBackbone import ResNetBackbone
from Experiment_Component.Models.Backbones.CDResNetBackbone import CDResNetBackbone

class CAE(tf.keras.Model):
    """
    The CAE model is a typical autoencoder with a image encoder and decoder.
    """
    def __init__(self, model_config, input_shape):
        """
        Constructor, initialize member variables.

        :param model_config: (Dictionary) The configuration of the model, containing layers specifications, learning rates, etc.
        :param input_shape: (Array) The shape of the input. E.g. [32, 32, 3].
        """
        super(CAE, self).__init__()

        self.verbose = model_config["verbose"]
        self.input_name = model_config["inputName"]
        if "outputName" in model_config.keys():
            self.output_name = model_config["outputName"]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=model_config["learningRate"])
        self.loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        self.width_multiplier = model_config["widthMultiplier"]
        self.stem = str(input_shape[0])
        self.backbone = model_config["backbone"]

        if self.backbone == "resNet":
            self.encoder_last_feature_dim = None
            if "encoderLastFeatureDim" in model_config.keys():
                 self.encoder_last_feature_dim = model_config["encoderLastFeatureDim"]
            self.weight_decay = 1e-4
            self.encoder = ResNetBackbone(model_config["resNet"], input_shape=input_shape, width_multiplier=self.width_multiplier, stem=self.stem, weight_devay=self.weight_decay, latent_dim=self.encoder_last_feature_dim)
            self.decoder = CDResNetBackbone(width_multiplier=self.width_multiplier, stem=self.stem, weight_decay=self.weight_decay)

        elif self.backbone == "CAE":
            self.encoder_last_feature_dim = model_config["encoderLastFeatureDim"] 
            self.encoder = CEBackbone(input_shape=input_shape, latent_dim=self.encoder_last_feature_dim, width_multiplier=self.width_multiplier, stem=self.stem)
            self.decoder = CDBackbone(width_multiplier=self.width_multiplier, stem=self.stem)

        self.last_activation = Activation('sigmoid')    

    def __call__(self, data, is_training):
        """
        Call method of the model which processes the input data for training or validation

        :param data: (Dictionary) The input data to process.
        :param is_training: (Boolean) If true, the models its training graph (E.g. from batch norm).
        :return: model_specific: (Tensor) The reconstruction of the input (or the denoised or colorized input)
        :return: representation: (Tensor) the representation of the input data from the model.
        """
        with tf.name_scope("Autoencoder"):
            representation = self.encoder(data[self.input_name], is_training=is_training)
            print(representation)
            reconstruction = self.decoder(representation, is_training=is_training)
            reconstruction = self.last_activation(reconstruction)
            return reconstruction, representation
      
    def getLoss(self, data, predictions):
        """
        Computes the loss for this model.

        :param data: (Dictionary) The input data to process.
        :param predictions: (Tensor) The prediction of the model for the specific loss. E.g. rotations or reconstructions.
        :return: loss_obj: (Tensor) The objective loss of the model
        :return: regularization_loss: (Tensor) The regularization loss of the model.
        """
        regularization_loss = 0.0
        if self.backbone == "resNet":
            regularization_loss = tf.add_n(self.encoder.losses) + tf.add_n(self.decoder.losses)
        labels = data[self.output_name]
        loss_obj = self.loss_object(labels, predictions[0])
        loss_obj = tf.reduce_mean(loss_obj, axis=[1, 2])
        return loss_obj, regularization_loss

    def writeSummary(self, summary_values, step):
        """
        Writes the summary for the tensorboard.

        :param summary_values: (Array of Array of Tensors) The input data to process.
        :param step: (Tensor) The current training step.
        """
        outputs = summary_values[0]
        data = outputs[2]

        tf.summary.scalar('loss', outputs[0][0], step=step)
        tf.summary.scalar('obj_loss', outputs[0][1], step=step)
        tf.summary.scalar('reg_loss', outputs[0][2], step=step)

        if self.verbose:
            tf.summary.scalar('loss_debug', outputs[0][1]+outputs[0][2], step=step)
            tf.summary.scalar('learningRate', self.optimizer.lr, step=step)
            tf.summary.image('input_image', data[self.input_name][0], step=step, max_outputs=3)
            tf.summary.image('output_image', data[self.output_name][0], step=step, max_outputs=3)
            tf.summary.image('reconstructed_image', outputs[1][0][0], step=step, max_outputs=3)
            tf.summary.image('dov_image', abs(data[self.input_name][0]-outputs[1][0][0]), step=step, max_outputs=3)
        
    def getRepresentationShape(self):
        """
        Returns the shape of the representation used for the target model.

        :return: shape: (Array) The shape of the representation used for the target model.
        """
        return self.encoder.getRepresentationShape()

    def getMetrics(self):
        """
        Returns the values of metrics logged turing the training.

        :return: metrics_values: (Array) The values of metrics logged during the training. -1 by default.
        """
        return [tf.constant(-1.0)]

    def updateLearningRate(self, epoch):
        """
        Updates the learning rate for the current training.
        """
        pass

    def resetMetrics(self):
        """
        Resets the logging of the metrics.
        """
        pass

    def updateMetrics(self, data, predictions):
        """
        Updates the logging of the metrics.
        """
        pass