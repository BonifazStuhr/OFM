from abc import ABCMeta, abstractmethod

class ITrainer(metaclass=ABCMeta):
    """
    The ITrainer provides the interface for trainer classes, such as a multi gpu trainer.
    """
    @abstractmethod
    def trainingStep(self, model, data):
        """
        Interface Method: Executes one training step of the model.
        The function to execute one training step. Handles predictions, metrics and losses.

        :param model: (tf.keras.Model) The model to train.
        :param data: (Dictionary) The dataset to train the model.
        :return: results: The results from the train.
        """
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def validationStep(self, model, data):
        """
        Interface Method: Executes one validation step of the model.
        The function to execute one training step. Handles predictions, metrics and losses.

        :param model: (tf.keras.Model) The model to validate.
        :param data: (Dictionary) The dataset to validate the model.
        :return: results: The results from the validation.
        """
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def scope(self):
        """
        Interface Method: Executes one validation step of the model.
        Returns the tensorflow scope of the optimizer.

        :return: scope: (scope): The scope of the trainer.
        """
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def distributeDataset(self, train_ds, eval_ds, test_ds):
        """
        Interface Method: Executes one validation step of the model.

        This function distributes the dataset on the gpus (if necessary).

        :param train_ds: (tf.data.Dataset) The dataset + input pipeline for training.
        :param eval_ds: (tf.data.Dataset) The dataset + input pipeline for evaluation.
        :param test_ds: (tf.data.Dataset) The dataset + input pipeline for testing.

        :return: train_ds: (tf.data.Dataset (per replica)) The (distributed) dataset + input pipeline for training.
        :return: eval_ds: (tf.data.Dataset (per replica)) The (distributed) dataset + input pipeline for evaluation.
        :return: test_ds: (tf.data.Dataset (per replica)) The (distributed) dataset + input pipeline for testing.
        """
        raise NotImplementedError('Not implemented')



