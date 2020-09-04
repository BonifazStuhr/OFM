import os
import glob
import fnmatch

import numpy as np
from PIL import Image

from utils.functions import downloadDataset
from Input_Component.ADataProvider import ADataProvider

HOMEPAGE = "https://tiny-imagenet.herokuapp.com/"
DATA_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

class TinyImageNetProvider(ADataProvider):
    """
    The TinyImageNetProvider reads the Tiny ImageNet and provides the dataset in various forms.
    The TinyImageNetProvider is not responsible for augmenting the dataset!
    :Attributes:
        __classIdToLabel:   (Dictionary) Contains the mapping from class id to label.
    """

    def __init__(self):
        """
        Constructor, initialize member variables.
        """
        super().__init__(dataset_path=os.path.join('data', 'TinyImageNet'),
                         dataset_name='TinyImageNet',
                         dataset_size=110000,
                         train_size=100000,
                         eval_size=0,
                         test_size=10000,
                         dataset_processable_at_once=True,
                         num_classes=200,
                         read_in_size=110000,
                         read_in_shape=[64, 64, 3])
        self.__classIdToLabel = {}

    def loadDataset(self):
        """
        Reads and returns the dataset.
        :return: x_train: (Array) The train data.
        :return: y_train: (Array) The train label.
        :return: x_eval: (Array) The eval data.
        :return: y_eval: (Array) The eval label.
        :return: x_test: (Array) The test data.
        :return: y_test: (Array) The test label.
        """
        # Download Tiny ImageNet dataset if it not exists.
        if not os.path.exists(super()._getDatasetPath()):
            downloadDataset(DATA_URL, super()._getDatasetPath())

        # Reading both sources.
        image_train, labels_train = self.__readTinyImageNetToNumpy("train")
        images_test, labels_test = self.__readTinyImageNetToNumpy("test")

        return (image_train, labels_train), (None, None), (images_test, labels_test)

    def __readTinyImageNetToNumpy(self, dataset_part):
        """
        Reads, converts and returns the dataset_part of Tiny ImageNet in numpy format.
        :param dataset_part: (String) The string describing the dataset part.
        :return: images: (np.array) The images in (datasetpart_size, 64, 64, 3) shape.
        :return: labels: (np.array) The labels in (datasetpart_size,) shape.
        """
        images = []
        labels = []

        # To give each label an id
        label_id = 0

        # Read in labels
        if dataset_part is "train":
            image_files = []
            for root, dirnames, filenames in os.walk(super()._getDatasetPath() +'/tiny-imagenet-200/train'):
                for filename in fnmatch.filter(filenames, '*.JPEG'):
                    image_files.append(os.path.join(root, filename))
                    x = filename.split("_")[0]
                    if x not in self.__classIdToLabel.keys():
                        self.__classIdToLabel[x] = label_id
                        label_id += 1

        else:
            filenameToClassId = {}
            image_files = glob.glob(super()._getDatasetPath() + "/tiny-imagenet-200/val/images/*.JPEG")
            f = open(super()._getDatasetPath() +"/tiny-imagenet-200/val/val_annotations.txt", "r")
            for line in f.readlines():
                x = line.split("\t")
                filenameToClassId[x[0]] = x[1]

        for image_file_name in image_files:
            image = Image.open(image_file_name)

            if image.mode is "L":
                image = image.convert("RGB")

            image_np = np.array(image.getdata()).reshape((64, 64, 3)).astype(np.uint8)
            images.append(image_np)

            if dataset_part is "train":
                labels.append(self.__classIdToLabel[image_file_name.split("\\")[-1].split("_")[0]])
            else:
                labels.append(self.__classIdToLabel[filenameToClassId[image_file_name.split("\\")[-1]]])

        p = np.random.RandomState(seed=42).permutation(len(labels))
        images = np.array(images)[p]
        labels = np.array(labels)[p]

        return images, labels


