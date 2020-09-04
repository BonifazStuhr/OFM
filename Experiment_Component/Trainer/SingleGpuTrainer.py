import tensorflow as tf

from Experiment_Component.ITrainer import ITrainer

class SingleGpuTrainer(ITrainer):
    """
    The SingleGpuTrainer trains a model on one GPU.
    """
    @tf.function
    def trainingStep(self, model, data):
        """
        The function to execute one training step. Handles predictions, gradients, metrics and losses.

        :param model: (tf.keras.Model) The model to train.
        :param data: (Dictionary) The dataset to train the model.
        :return: results: (Array of Arrays) The results in form: [[mean_loss, mean_loss_obj, loss_reg], [predictions], data_out]
        """
        with tf.name_scope("SingleGPUTrainingStep"):
            with tf.GradientTape() as tape:
                predictions = model(data, is_training=True)

                per_sample_losses_obj, loss_reg = model.getLoss(data, predictions)
                mean_loss_obj = tf.reduce_mean(per_sample_losses_obj)
                mean_loss = mean_loss_obj + loss_reg

            gradients = tape.gradient(mean_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            model.updateMetrics(data, predictions)

        # To be compatible with multi gpu trainer.
        data_out = {}
        if model.verbose:
            for key, value in data.items():
                data_out[key] = [value]

        return [[mean_loss, mean_loss_obj, loss_reg], [predictions], data_out]

    @tf.function
    def validationStep(self, model, data):
        """
        The function to execute one validation step. Handles predictions, metrics and losses.

        :param model: (tf.keras.Model) The model to validate.
        :param data: (Dictionary) The dataset to validate the model.
        :return: results: (Array of Arrays) The results in form: [[per_sample_losses_obj, loss_reg], [predictions], data_out]
        """
        with tf.name_scope("SingleGPUValidationStep"):
            predictions = model(data, is_training=False)

            per_sample_losses_obj, loss_reg = model.getLoss(data, predictions)

            model.updateMetrics(data, predictions)

            # To be compatible with multi gpu trainer.
            data_out = {}
            if model.verbose:
                for key, value in data.items():
                    data_out[key] = [value]

            return [[per_sample_losses_obj, loss_reg], [predictions], data_out]
     
    def scope(self):
        """
        Returns the tensorflow scope of the trainer.

        :return: scope: (scope): The scope of the trainer.
        """
        return tf.name_scope("SingleGPUTraining")

    def distributeDataset(self, train_ds, eval_ds, test_ds):
        """
        This function distributes the dataset on the gpus (if necessary).

        :param train_ds: (tf.data.Dataset) The dataset + input pipeline for training.
        :param eval_ds: (tf.data.Dataset) The dataset + input pipeline for evaluation.
        :param test_ds: (tf.data.Dataset) The dataset + input pipeline for testing.

        :return: train_ds: (tf.data.Dataset (per replica)) The (distributed) dataset + input pipeline for training.
        :return: eval_ds: (tf.data.Dataset (per replica)) The (distributed) dataset + input pipeline for evaluation.
        :return: test_ds: (tf.data.Dataset (per replica)) The (distributed) dataset + input pipeline for testing.
        """
        return train_ds, eval_ds, test_ds


    def getMetrics(self, model):
        """
        Returns the values of metrics logged turing the training.

        :return: metrics_values: (Array) The values of metrics logged during the training.
        """
        return model.getMetrics()
