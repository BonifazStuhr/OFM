import tensorflow as tf

class TensorboardSummaryManager():
    """
    The TensorboardSummarymanager manages the train, eval and test summaries for the Tensorboard.

     :Attributes:
        __summary_train_writer: (tf.summary.FileWriter) The summary writer for the training steps.
        __summary_eval_writer: (tf.summary.FileWriter) The summary writer for the eval steps.
        __summary_test_writer: (tf.summary.FileWriter) The summary writer for the test steps.
    """
    def __init__(self, model_dir):
        """
        Constructor, initialize member variables.
        :param model_dir: (String) The path to the model directory, where the summary will be saved under /logs.
        """
        self.__summary_train_writer = tf.summary.create_file_writer(model_dir + "/logs/train/")
        self.__summary_eval_writer = tf.summary.create_file_writer(model_dir + "/logs/eval/")
        self.__summary_test_writer = tf.summary.create_file_writer(model_dir + "/logs/test/")

    def writer(self, mode):
        """
        Write the summary for the given mode.
        :param mode: (String) The mode for which the summary is saved.
        """
        if mode == "train":
            return self.__summary_train_writer
        elif mode == "eval":
            return self.__summary_eval_writer
        elif mode == "test":
            return self.__summary_test_writer
