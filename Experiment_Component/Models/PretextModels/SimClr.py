import tensorflow as tf

from tensorflow.keras.layers import Flatten
#from tensorflow.python.distribute import distribution_strategy_context as ds

from Experiment_Component.Models.Backbones.CEBackbone import CEBackbone
from Experiment_Component.Models.Backbones.ResNetBackbone import ResNetBackbone
from Experiment_Component.Models.Backbones.NonLinearHead import NonLinearHead
from Experiment_Component.Models.Backbones.ResNetNonLinearHead import ResNetNonLinearHead

LARGE_NUM = 1e9

class SimClr(tf.keras.Model):
    """
    The SimClr model encodes two inputs by a encoder and learns a representation layer by contrastive learning.
    A NonlinearHead transforms the representation layer for contrastive learning and the representation is not learned
    "directly".
    """
    def __init__(self, model_config, input_shape):
        """
        Constructor, initialize member variables.

        :param model_config: (Dictionary) The configuration of the model, containing layers specifications, learning rates, etc.
        :param input_shape: (Array) The shape of the input. E.g. [32, 32, 3].
        """
        super(SimClr, self).__init__()

        self.verbose = model_config["verbose"]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=model_config["learningRate"])
        self.loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.accuracy_metrics = tf.keras.metrics.CategoricalAccuracy(name='accuracy')

        self.width_multiplier = model_config["widthMultiplier"]
        self.stem = str(input_shape[0])
        self.backbone = model_config["backbone"]

        if self.backbone == "resNet":
            self.encoder_last_feature_dim = None
            if "encoderLastFeatureDim" in model_config.keys():
                 self.encoder_last_feature_dim = model_config["encoderLastFeatureDim"]
            self.weight_decay = 1e-4
            self.encoder = ResNetBackbone(model_config["resNet"], input_shape=input_shape, width_multiplier=self.width_multiplier, stem=self.stem, weight_decay=self.weight_decay, latent_dim=self.encoder_last_feature_dim)
            self.latent_dim = self.encoder.layers[-1].output_shape[-1]
            self.nonlinear_head = ResNetNonLinearHead(input_shape=self.latent_dim, weight_decay=self.weight_decay)

        elif self.backbone == "CAE":
            self.encoder_last_feature_dim = model_config["encoderLastFeatureDim"] 
            self.encoder = CEBackbone(input_shape=input_shape, latent_dim=self.encoder_last_feature_dim, width_multiplier=self.width_multiplier, stem=self.stem)
            self.latent_dim = self.encoder.layers[-1].output_shape[-1]
            self.nonlinear_head = NonLinearHead(input_shape=self.latent_dim)

        self.flatten = Flatten()

    def __call__(self, data, is_training):
        """
        Call method of the model which processes the input data for training or validation

        :param data: (Dictionary) The input data to process.
        :param is_training: (Boolean) If true, the models its training graph (E.g. from batch norm).
        :return: model_specific: (Array of Tensors) [proj_representation, entropy_con] The projected representation of the Nonlinear Head and the entropy of the loss.
        :return: representation: (Tensor) the representation of the input data from the model.
        """
        with tf.name_scope("SimClr"):
            if "image2" in data:
                images = tf.concat([data["image"], data["image2"]], 0)
            else:
                images = data["image"]
            representation = self.encoder(images, is_training=is_training)
            print(representation)
            representation_flatten = self.flatten(representation)

            proj_representation = self.nonlinear_head(representation_flatten, is_training=is_training)
            return proj_representation, representation
      
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
            regularization_loss = tf.add_n(self.encoder.losses) + tf.add_n(self.nonlinear_head.losses)
        loss, logits_con, labels_con = self.contrastiveLoss(predictions[0])
        self.accuracy_metrics.update_state(labels_con, logits_con)
        return loss, regularization_loss

    def contrastiveLoss(self, hidden, hidden_norm=True, temperature=0.5):

        if hidden_norm:
            hidden = tf.math.l2_normalize(hidden, -1)
        hidden1, hidden2 = tf.split(hidden, 2, 0)

        batch_size = tf.shape(hidden1)[0]
    
        def concat_fn(strategie, hidden1, hidden2):
            return tf.concat(strategie.experimental_local_results(hidden1), axis=0),tf.concat(strategie.experimental_local_results(hidden2), axis=0)

        """
        # Sync hiddens accros replicas to get a large batch size (more negatives).
        replica_ctx = tf.distribute.get_replica_context()
        if replica_ctx:
            hidden1_large, hidden2_large = replica_ctx.merge_call(concat_fn, (hidden1, hidden2))
            enlarged_batch_size = tf.shape(hidden1_large)[0]
            replica_id = replica_ctx.replica_id_in_sync_group
            labels_idx = tf.range(batch_size) + replica_id * batch_size
            labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
            masks = tf.one_hot(labels_idx, enlarged_batch_size)  
        else:
        """
        
        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)

        logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
        logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

        loss_a = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=tf.concat([logits_ab, logits_aa], 1))
        loss_b = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=tf.concat([logits_ba, logits_bb], 1))
        loss = loss_a + loss_b
        
        return loss, logits_ab, labels

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
        tf.summary.scalar('contrastive_acc', metrics[0], step=step)

        if self.verbose:
            tf.summary.scalar('loss_debug', outputs[0][1]+outputs[0][2], step=step)
            tf.summary.scalar('learningRate', self.optimizer.lr, step=step)
            tf.summary.image('input_image1', data["image"][0], step=step, max_outputs=3)
            tf.summary.image('input_image2', data["image2"][0], step=step, max_outputs=3)

    def getRepresentationShape(self):
        """
        Returns the shape of the representation used for the target model.

        :return: shape: (Array) The shape of the representation used for the target model.
        """
        return self.encoder.getRepresentationShape()

    def updateLearningRate(self, epoch):
        """
        Updates the learning rate for the current training.
        """

        """
        if self.use_learnig_rate_schedule:
            global_step = epoch * self.num_examples // self.train_batch_size + 1

            warmup_steps = int(round(self.warm_up_epochs * self.num_examples // self.train_batch_size))

            scaled_lr = self.base_learning_rate * self.train_batch_size / 256.
            learning_rate = (tf.to_float(global_step) / int(warmup_steps) * scaled_lr
                            if warmup_steps else scaled_lr)

            # Cosine decay learning rate schedule
            total_steps = self.train_epochs * self.num_examples // self.train_batch_size + 1 
            learning_rate = tf.where(
                global_step < warmup_steps, learning_rate,
                tf.train.cosine_decay(
                    scaled_lr,
                    global_step - warmup_steps,
                    total_steps - warmup_steps))

            tf.keras.backend.set_value(self.optimizer.lr, learning_rate)
        """
        pass

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
        pass