import tensorflow as tf

from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.layers import Dense, Flatten,  InputLayer, Activation
from tensorflow.keras import Model

class NonLinearClassifierFC(Model):
    """
    The NonLinearClassifierFC model learns on a representation of the given feature_extractor.
    Its stops the gradient before the feature_extractor, therefore only the LinearClassifierFC learns
    """

    def __init__(self, model_config, num_classes, input_shape, feature_extractor):
        """
        Constructor, initialize member variables.

        :param model_config: (Dictionary) The configuration of the model, containing layers specifications, learning rates, etc.
        :param num_classes: (Integer) The number of classes of the target objective.
        :param input_shape: (Array) The shape of the input. E.g. [32, 32, 3].
        :param feature_extractor: (tf.keras.Model) The feature_extractor to use to extract features from the input.
        """
        super(NonLinearClassifierFC, self).__init__()

        self.num_classes = num_classes
        self.verbose = model_config["verbose"]
        self.lr = model_config["learningRate"]
        self.gamma = model_config["learningRateDecayGamma"]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.accuracy_metrics = tf.keras.metrics.CategoricalAccuracy(name='accuracy')

        self.feature_extractor = feature_extractor

        self.backbone = [InputLayer(input_shape=input_shape), Flatten()]
        for units in model_config["layers"]:
            self.backbone.append(Dense(units, activation=None))
            self.backbone.append(SyncBatchNormalization())
            self.backbone.append(Activation("relu"))
        self.backbone.append(Dense(self.num_classes, activation="softmax"))

        self.linearClassifierFC = tf.keras.Sequential(self.backbone)
    

    def __call__(self, data, is_training):
        """
        Call method of the model which processes the input data for training or validation.

        :param data: (Dictionary) The input data to process.
        :param is_training: (Boolean) If true, the models its training graph (E.g. from batch norm).
        :return: x: (Tensor) The predicted class or regression.
        """
        _, x = self.feature_extractor(data, is_training=False)
        x = tf.stop_gradient(x)    
        x = self.linearClassifierFC(x, training=is_training)
        return [x]

    def getLoss(self, data, predictions):
        """
        Computes the loss for this model.

        :param data: (Dictionary) The input data to process.
        :param predictions: (Tensor) The prediction of the model for the specific loss. E.g. rotations or reconstructions.
        :return: loss_obj: (Tensor) The objective loss of the model
        :return: regularization_loss: (Tensor) The regularization loss of the model.
        """
        labels = data["label"]
        loss = self.loss_object(labels, predictions[0])
        regularization_loss = 0.0
        return loss, regularization_loss

    def getMetrics(self):
        """
        Returns the values of metrics logged turing the training.

        :return: metrics_values: (Array) The values of metrics logged during the training. Accuracy in this case.
        """
        acc = self.accuracy_metrics.result()
        return [acc]

    def resetMetrics(self):
        """
        Resets the logging of the metrics.
        """
        self.accuracy_metrics.reset_states()

    def updateMetrics(self, data, predictions):
        """
        Updates the logging of the metrics.
        """
        self.accuracy_metrics.update_state(data["label"], predictions)

    def writeSummary(self, summary_values, step):
        """
        Writes the summary for the tensorboard.

        :param summary_values: (Array of Array of Tensors) The input data to process.
        :param step: (Tensor) The current training step.
        """
        outputs = summary_values[0]
        metrics = summary_values[1]
        data = outputs[2]
        
        tf.summary.scalar('loss', outputs[0][0], step=step)
        tf.summary.scalar('obj_loss', outputs[0][1], step=step)
        tf.summary.scalar('reg_loss', outputs[0][2], step=step)
        tf.summary.scalar('acc', metrics[0], step=step)

        if self.verbose:
            tf.summary.scalar('loss_debug', outputs[0][1]+outputs[0][2], step=step)
            tf.summary.scalar('learningRate', self.lr, step=step)
            tf.summary.image('input_image', data["image"][0], step=step, max_outputs=3)

    def updateLearningRate(self, epoch):
        """
        Updates the learning rate for the current training.
        """
        pass
