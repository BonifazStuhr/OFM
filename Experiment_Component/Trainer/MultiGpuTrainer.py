import tensorflow as tf

from Experiment_Component.ITrainer import ITrainer

class MultiGpuTrainer(ITrainer):
    """
    The MultiGpuTrainer trains a model on multiple GPUs.

     :Attributes:
        mirrored_strategy:  (tf.distribute.MirroredStrategy) The strategy used to train the model on multiple gpus.
        global_batch_size:  (Integer) The global batch size used to train the model.
    """
    def __init__(self, global_batch_size):
        """
        Constructor, initialize member variables.

        :param global_batch_size:  (Integer) The global batch size used to train the model.
        """
        self.mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
        self.global_batch_size = global_batch_size

    @tf.function
    def trainingFn(self, model, data, global_batch_size):
        """
        The function to execute one training step on a single gpu. Handles predictions, gradients, metrics and losses.

        :param model: (tf.keras.Model) The model to train.
        :param data: (Dictionary) The dataset to train the model.
        :param global_batch_size:  (Integer) The global batch size used to train the model.
        :return: results: (Arrays) The results in form: mean_loss, mean_loss_obj, scaled_regularization_loss, predictions
        """
        with tf.name_scope("SingleGPUTrainingStep"):

            with tf.GradientTape() as tape:
                predictions = model(data, is_training=True)
                per_sample_loss_obj, loss_reg = model.getLoss(data, predictions)
                mean_loss_obj = tf.nn.compute_average_loss(per_sample_loss_obj, global_batch_size=global_batch_size)
                scaled_regularization_loss = tf.nn.scale_regularization_loss(loss_reg) 
                mean_loss = mean_loss_obj + scaled_regularization_loss
                
            gradients = tape.gradient(mean_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            model.updateMetrics(data, predictions)
            
            return mean_loss, mean_loss_obj, scaled_regularization_loss, predictions

    @tf.function
    def trainingStep(self, model, data):
        """
        The function to execute one training step. Handles predictions, metrics and losses.
        Handles the distribution and reduction to different gpus as well. 

        :param model: (tf.keras.Model) The model to train.
        :param data: (Dictionary) The dataset to train the model.
        :return: results: (Array of Arrays) The results in form: [[mean_loss, mean_loss_obj, loss_reg], predictions, data_out]
        """
        with tf.name_scope("MultiGPUTrainingStep"):
            mean_losses, mean_losses_obj, scaled_regularization_losses, predictions = self.mirrored_strategy.run(self.trainingFn, args=(model, data, self.global_batch_size))
        
            mean_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, mean_losses, axis=None)
            mean_loss_obj = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, mean_losses_obj, axis=None)
            loss_reg = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, scaled_regularization_losses, axis=None)

            predictions_out = []
            data_out = {}
            if model.verbose:
                for pred in predictions:
                    predictions_out.append(self.mirrored_strategy.experimental_local_results(pred))
                for key, value in data.items():
                    data_out[key] = self.mirrored_strategy.experimental_local_results(value)

            return [[mean_loss, mean_loss_obj, loss_reg], predictions_out, data_out]
    
    @tf.function
    def validationFn(self, model, data, global_batch_size):
        """
        The function to execute one evaluation step on a single gpu. Handles predictions, metrics and losses.

        :param model: (tf.keras.Model) The model to train.
        :param data: (Dictionary) The dataset to train the model.
        :param global_batch_size:  (Integer) The global batch size used to evaluate the model.
        :return: results: (Arrays) The results in form: per_sample_losses_obj, scaled_regularization_loss, predictions
        """
        with tf.name_scope("SingleGPUValidationStep"):
            predictions = model(data, is_training=False)

            per_sample_losses_obj, loss_reg = model.getLoss(data, predictions)
            scaled_regularization_loss = tf.nn.scale_regularization_loss(loss_reg) 
           
            model.updateMetrics(data, predictions)

            return per_sample_losses_obj, scaled_regularization_loss, predictions

    @tf.function
    def validationStep(self, model, data):
        """
        The function to execute one validation step. Handles predictions, metrics and losses.
        Handles the distribution and reduction to different gpus as well. 

        :param model: (tf.keras.Model) The model to validate.
        :param data: (Dictionary) The dataset to validate the model.
        :return: results: (Array of Arrays) The results in form: [[per_sample_losses_obj, loss_reg], predictions, data_out]
        """
        with tf.name_scope("MultiGPUValidationStep"):
            per_sample_losses_obj, scaled_regularization_loss, predictions = self.mirrored_strategy.run(self.validationFn, args=(model, data, self.global_batch_size))
 
            per_sample_losses_obj = tf.concat(self.mirrored_strategy.experimental_local_results(per_sample_losses_obj), axis=0)
            loss_reg = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, scaled_regularization_loss, axis=None)

            predictions_out = []
            data_out = {}
            if model.verbose:
                for pred in predictions:
                    predictions_out.append(self.mirrored_strategy.experimental_local_results(pred))
                for key, value in data.items():
                    data_out[key] = self.mirrored_strategy.experimental_local_results(value)
    
            return [[per_sample_losses_obj, loss_reg], predictions_out, data_out]

    def scope(self):
        """
        Returns the tensorflow scope of the trainer.

        :return: scope: (scope): The scope of the trainer.
        """
        return self.mirrored_strategy.scope()

    def distributeDataset(self, train_ds, eval_ds, test_ds):
        """
        This function distributes the dataset on the gpus.

        :param train_ds: (tf.data.Dataset) The dataset + input pipeline for training.
        :param eval_ds: (tf.data.Dataset) The dataset + input pipeline for evaluation.
        :param test_ds: (tf.data.Dataset) The dataset + input pipeline for testing.

        :return: train_ds: (tf.data.Dataset (per replica)) The (distributed) dataset + input pipeline for training.
        :return: eval_ds: (tf.data.Dataset (per replica)) The (distributed) dataset + input pipeline for evaluation.
        :return: test_ds: (tf.data.Dataset (per replica)) The (distributed) dataset + input pipeline for testing.
        """
        train_ds = self.mirrored_strategy.experimental_distribute_dataset(train_ds)
        eval_ds = self.mirrored_strategy.experimental_distribute_dataset(eval_ds)
        test_ds = self.mirrored_strategy.experimental_distribute_dataset(test_ds)
        return train_ds, eval_ds, test_ds


    def getMetrics(self, data):
        """
        Returns the values of metrics logged turing the training.

        :return: metrics_values: (Array) The values of metrics logged during the training.
        """
        return self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, data, axis=None)

