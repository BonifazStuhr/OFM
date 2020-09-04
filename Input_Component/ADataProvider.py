import numpy as np
from abc import ABCMeta, abstractmethod

class ADataProvider(metaclass=ABCMeta):
    """
    The ADataProvider reads the dataset from tf records and provides the dataset in various forms.
    The ADataProvider is not responsible for augmenting the dataset!

    :Attributes:
        __dataset_path:                    (String) The path to the gz files of the mnist dataset.
        __dataset_name:                    (String) The name of the dataset.
        __dataset_size:                    (Integer) The size of the dataset (for mnist 70000).
        __train_size:                      (Integer) The size of the training set. 60000 by default.
        __eval_size:                       (Integer) The size of the eval set. 0 by default.
        __test_size:                       (Integer) The size of the test set. 10000 by default.
        __dataset_processable_at_once:     (Boolean) Is it possible to load and process the whole dataset in ram?
        __num_classes:                     (Integer) The number of classes of the dataset.
        __read_in_size:                    (Integer) The size of the dataset to read in.
        _read_in_shape:                    (Array) The shape of the input date to read in
        __read_in_images:                  (Array) If the dataset is read in in numpy and fits in memory the input data
                                            will be saved.
        __read_in_labels:                  (Array) If the dataset is read in in numpy and fits in memory the label data
                                            will be saved.
        __read_in_dataset:                 (Boolean) If false the dateset will be read from disk. If true the dataset is
                                            already in memory.
    """

    def __init__(self, dataset_path, dataset_name, dataset_size, train_size, eval_size, test_size,
                 dataset_processable_at_once, num_classes=None, read_in_size=None, read_in_shape=None):
        """
        Constructor, initialize member variables.
        :param dataset_path: (String) The path to the gz files of the mnist dataset.
        :param dataset_name: (String) The name of the dataset.
        :param dataset_size: (Integer) The size of the dataset (for mnist 70000).
        :param train_size: (Integer) The size of the training set.
        :param eval_size: (Integer) The size of the eval set.
        :param test_size: (Integer) The size of the test set.
        :param dataset_processable_at_once: (Boolean) Is the dataset processable at once?
        :param num_classes: (Integer) The number of classes of the dataset.
        :param read_in_size: (Integer) The size of the dataset to read in.
        :param read_in_shape: (Array) The shape of the input date to read in
        """
        self.__dataset_path = dataset_path
        self.__dataset_name = dataset_name
        self.__dataset_size = dataset_size

        # Set values for the dataset split.
        self.__train_size = train_size
        self.__eval_size = eval_size
        self.__test_size = test_size

        self.__dataset_processable_at_once = dataset_processable_at_once

        # Check if the split is possible.
        assert train_size + eval_size + test_size <= self.__dataset_size

        self.__num_classes = num_classes
        self.__read_in_size = read_in_size
        self._read_in_shape = read_in_shape

        # Variables to save the read in dataset in memory.
        self.__read_in_images = None
        self.__read_in_labels = None
        self.__read_in_dataset = False

    def getSplittedDatasetInNumpy(self, xFold_step, xFold_type, depth_first=False, onehot=False):
        """
        Reads and returns the data in numpy format with the set split (setDatasetSplit).
        :param xfold_step: (Integer) The step of the current xFold-cross-validation.
        :param xfold_type: (Integer) The type for the xFold operation.
        :param depth_first: (Boolean) If true the image dimensions are NHWC. False by default.
        :param onehot: (Boolean) If true the label is converted to onehot encoding. True by default.
        :return: dataset: (Dictionary) The dataset e.g. {"x_train":(train_size, 28, 28, 1), "y_train":(train_size,)
                          or if onehot (train_size, 10), x_eval....
        """
        if not self.__read_in_dataset:
            (x_train, y_train), (x_eval, y_eval), (x_test, y_test) = self.loadDataset()

            self.__read_in_images, self.__read_in_labels = self.convertDatasetToNumpy(x_train, y_train, x_eval, y_eval,
                                                            x_test, y_test, self._read_in_shape, self.__read_in_size,
                                                            self.__num_classes, depth_first, onehot)
            self.__read_in_dataset = True

        return self.prepare_dataset(xFold_step, xFold_type)

    def prepare_dataset(self, xFold_step, xFold_type):
        """
        Prepares the dataset for the current xFold_step and xFold_type.
        :param xfold_step: (Integer) The step of the current xFold-cross-validation.
        :param xfold_type: (Integer) The type for the xFold operation.
        :param depth_first: (Boolean) If true the image dimensions are NHWC. False by default.
        :param onehot: (Boolean) If true the label is converted to onehot encoding. True by default.
        :return: dataset: (Dictionary) The dataset e.g. {"x_train":(train_size, 28, 28, 1), "y_train":(train_size,)
                          or if onehot (train_size, 10), x_eval....
        """

        eval_samples_per_xfold = int(round((self.__train_size + self.__eval_size)/xFold_type))

        start_index = int(xFold_step*eval_samples_per_xfold)
        end_index = int(start_index + eval_samples_per_xfold)

        if end_index < len(self.__read_in_labels[-self.__test_size:]):
            end_index = len(self.__read_in_labels[-self.__test_size:])

        dataset = {
                "x_train": np.concatenate((self.__read_in_images[:start_index], self.__read_in_images[end_index:]), axis=0),
                "y_train": np.concatenate((self.__read_in_labels[:start_index], self.__read_in_labels[end_index:]), axis=0),

                "x_eval": self.__read_in_images[start_index:end_index],
                "y_eval": self.__read_in_labels[start_index:end_index],

                "x_test": self.__read_in_images[-self.__test_size:],
                "y_test": self.__read_in_labels[-self.__test_size:],
            }

        return dataset

    def convertDatasetToNumpy(self, x_train, y_train, x_eval, y_eval, x_test, y_test, shape, dataset_size, num_classes,
                              depth_first, onehot):
        """
        Reads, converts and returns the dataset_part in numpy format.
        :param x_train: (Array) The train data.
        :param y_train: (Array) The train label.
        :param x_eval: (Array) The eval data.
        :param y_eval: (Array) The eval label.
        :param x_test: (Array) The test data.
        :param y_test: (Array) The test label.
        :param shape: (Array) The shape of the in input data.
        :param dataset_size: (Integer) The size of the dataset.
        :param depth_first: (Boolean) If true the image dimensions are NCHW. False by default.
        :param onehot: (Boolean) If true the label is converted to onehot encoding. True by default.
        :return: images: (np.array) The images in (datasetpart_size, 28, 28, 1) shape.
        :return: labels: (np.array) The labels in (datasetpart_size,) shape.
        """
        if (x_eval is None) and (x_test is None):
            input = x_train
            labels = y_train
        elif x_eval is None:
            input = np.concatenate((x_train, x_test), axis=0)
            labels = np.concatenate((y_train, y_test), axis=0)
        else:
            input = np.concatenate((x_train, x_eval, x_test), axis=0)
            labels = np.concatenate((y_train, y_eval, y_test), axis=0)

        shape = np.insert(shape, 0, -1, axis=0)
        input = np.reshape(input, shape)
        shape[0] = dataset_size
        assert list(input.shape) == list(shape)
        assert np.min(input) >= 0
        assert np.max(input) <= 255

        if depth_first:
            input = np.transpose(input, [0, 3, 1, 2])

        labels = np.reshape(labels, [-1, ])
        labels = labels.astype(np.uint8)
        assert labels.shape == (dataset_size, ) 
        assert labels.dtype == np.uint8
        assert np.min(labels) == 0 
        assert np.max(labels) == (num_classes - 1)

        # If onehot encoding is needed, the labels will be encoded.
        if onehot:
            onehot_labels = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
            onehot_labels[np.arange(labels.size), labels] = 1.0
            labels = np.array(onehot_labels, dtype=np.uint8)
            assert labels.shape == (dataset_size, num_classes) and labels.dtype == np.uint8

        return input, labels

    @abstractmethod
    def loadDataset(self):
        """
        Interface Method: Reads and returns the dataset.
        :return: x_train: (Array) The train data.
        :return: y_train: (Array) The train label.
        :return: x_eval: (Array) The eval data.
        :return: y_eval: (Array) TThe eval label.
        :return: x_test: (Array) The test data.
        :return: y_test: (Array) The test label.
        """
        raise NotImplementedError('Not implemented')

    def _getDatasetPath(self):
        """
        Returns the path to the "raw" data of the dataset.
        :return: dataset_path: (String) The path to the dataset.
        """
        return self.__dataset_path




