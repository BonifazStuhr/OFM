from Input_Component.DatasetAugmentationBuilder import DatasetAugmentationBuilder

from Preprocessing_Component.Preprocessing.RandomCropWithResize import RandomCropWithResize
from Preprocessing_Component.Preprocessing.CenterCropWithResize import CenterCropWithResize
from Preprocessing_Component.Preprocessing.RandomHorizontalFlip import RandomHorizontalFlip
from Preprocessing_Component.Preprocessing.MaxDivNormalizer import MaxDivNormalizer
from Preprocessing_Component.Preprocessing.RandomNormalNoise import RandomNormalNoise
from Preprocessing_Component.Preprocessing.ClipByValue import ClipByValue
from Preprocessing_Component.Preprocessing.RandomColorJitter import RandomColorJitter
from Preprocessing_Component.Preprocessing.Random90xRotation import Random90xRotation
from Preprocessing_Component.Preprocessing.OneHot import OneHot

def targetTaskAugmentations(dataset_config, given_shape):
    """
    Applies base augmentations for the training of target models.
    Train Augmentations: max_div_images --> RandomCropWithResize --> RandomHorizontalFlip --> RandomColorJitter --> ClipByValue
                         one_hot
    Val Augmentations: max_div_images --> (CenterCropWithResize) --> (ClipByValue)
                       one_hot

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    if "labelName" in dataset_config.keys():
        label_name = dataset_config["labelName"]
    else:
        label_name = "label"

    max_div_images = MaxDivNormalizer(255.0)
    one_hot = OneHot(dataset_config["numClasses"], input_name=label_name, output_name="label")

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[one_hot, max_div_images, RandomCropWithResize(crop_shape=crop_shape),
                                                                        RandomHorizontalFlip(), RandomColorJitter(), ClipByValue()])

    val_preprocessors = [one_hot, max_div_images]
    if list(given_shape) != list(data_shape):
        val_preprocessors.append(CenterCropWithResize(crop_shape=crop_shape))
        val_preprocessors.append(ClipByValue())

    dataset_augmentor_val = DatasetAugmentationBuilder(preprocessors=val_preprocessors)

    return dataset_augmentor_train, dataset_augmentor_val


def targetTaskAugmentationsNoJitter(dataset_config, given_shape):
    """
    Applies base augmentations for the training of target models.
    Train Augmentations: max_div_images --> RandomCropWithResize --> RandomHorizontalFlip --> ClipByValue
                         one_hot
    Val Augmentations: max_div_images --> (CenterCropWithResize) --> (ClipByValue)
                       one_hot

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    if "labelName" in dataset_config.keys():
        label_name = dataset_config["labelName"]
    else:
        label_name = "label"

    max_div_images = MaxDivNormalizer(255.0)
    one_hot = OneHot(dataset_config["numClasses"], input_name=label_name, output_name="label")

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[one_hot, max_div_images, RandomCropWithResize(crop_shape=crop_shape),
                                                                        RandomHorizontalFlip(), ClipByValue()])

    val_preprocessors = [one_hot, max_div_images]
    if list(given_shape) != list(data_shape):
        val_preprocessors.append(CenterCropWithResize(crop_shape=crop_shape))
        val_preprocessors.append(ClipByValue())

    dataset_augmentor_val = DatasetAugmentationBuilder(preprocessors=val_preprocessors)

    return dataset_augmentor_train, dataset_augmentor_val

def targetTaskAugmentationsNoJitterNoFlip(dataset_config, given_shape):
    """
    Applies base augmentations for the training of target models.
    Train Augmentations: max_div_images --> RandomCropWithResize --> ClipByValue
                         one_hot
    Val Augmentations: max_div_images --> (CenterCropWithResize) --> (ClipByValue)
                       one_hot

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    if "labelName" in dataset_config.keys():
        label_name = dataset_config["labelName"]
    else:
        label_name = "label"

    max_div_images = MaxDivNormalizer(255.0)
    one_hot = OneHot(dataset_config["numClasses"], input_name=label_name, output_name="label")

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[one_hot, max_div_images, RandomCropWithResize(crop_shape=crop_shape),
                                                                        ClipByValue()])

    val_preprocessors = [one_hot, max_div_images]
    if list(given_shape) != list(data_shape):
        val_preprocessors.append(CenterCropWithResize(crop_shape=crop_shape))
        val_preprocessors.append(ClipByValue())

    dataset_augmentor_val = DatasetAugmentationBuilder(preprocessors=val_preprocessors)

    return dataset_augmentor_train, dataset_augmentor_val

def targetTaskAugmentationsNoFlip(dataset_config, given_shape):
    """
    Applies base augmentations for the training of target models.
    Train Augmentations: max_div_images --> RandomCropWithResize --> RandomColorJitter --> ClipByValue
                         one_hot
    Val Augmentations: max_div_images --> (CenterCropWithResize) --> (ClipByValue)
                       one_hot

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    if "labelName" in dataset_config.keys():
        label_name = dataset_config["labelName"]
    else:
        label_name = "label"

    max_div_images = MaxDivNormalizer(255.0)
    one_hot = OneHot(dataset_config["numClasses"], input_name=label_name, output_name="label")

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[one_hot, max_div_images, RandomCropWithResize(crop_shape=crop_shape),
                                                                        RandomColorJitter(), ClipByValue()])

    val_preprocessors = [one_hot, max_div_images]
    if list(given_shape) != list(data_shape):
        val_preprocessors.append(CenterCropWithResize(crop_shape=crop_shape))
        val_preprocessors.append(ClipByValue())

    dataset_augmentor_val = DatasetAugmentationBuilder(preprocessors=val_preprocessors)

    return dataset_augmentor_train, dataset_augmentor_val

def caeAugmentations(dataset_config, given_shape):
    """
    Applies base augmentations for the training of plain autoencoders.
    Augmentations: max_div_images --> RandomCropWithResize --> RandomHorizontalFlip --> RandomColorJitter --> ClipByValue

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    max_div_images = MaxDivNormalizer(255.0)

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[max_div_images, RandomCropWithResize(crop_shape=crop_shape),
                                                                        RandomHorizontalFlip(), RandomColorJitter(), ClipByValue()])

    dataset_augmentor_val = dataset_augmentor_train

    return dataset_augmentor_train, dataset_augmentor_val

def caeAugmentationsNoJitter(dataset_config, given_shape):
    """
    Applies base augmentations for the training of plain autoencoders.
    Augmentations: max_div_images --> RandomCropWithResize --> RandomHorizontalFlip --> ClipByValue

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    max_div_images = MaxDivNormalizer(255.0)

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[max_div_images, RandomCropWithResize(crop_shape=crop_shape),
                                                                        RandomHorizontalFlip(), ClipByValue()])

    dataset_augmentor_val = dataset_augmentor_train

    return dataset_augmentor_train, dataset_augmentor_val

def caeAugmentationsNoJitterNoFlip(dataset_config, given_shape):
    """
    Applies base augmentations for the training of plain autoencoders.
    Augmentations: max_div_images --> RandomCropWithResize --> ClipByValue

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    max_div_images = MaxDivNormalizer(255.0)

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[max_div_images, RandomCropWithResize(crop_shape=crop_shape),
                                                                        ClipByValue()])

    dataset_augmentor_val = dataset_augmentor_train

    return dataset_augmentor_train, dataset_augmentor_val

def denoisingCaeAugmentations(dataset_config, given_shape):
    """
    Applies augmentations for the training of denoising autoencoders.
    Augmentations: max_div_images --> RandomCropWithResize --> RandomHorizontalFlip --> RandomColorJitter--> noise_images --> clip_by_value
                                                                                                                  |--> ClipByValue
    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    max_div_images = MaxDivNormalizer(255.0)
    noise_images = RandomNormalNoise(input_name="image", output_name="noisy_image")
    clip_by_value = ClipByValue(input_name="noisy_image", output_name="noisy_image")

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[max_div_images, RandomCropWithResize(crop_shape=crop_shape), RandomHorizontalFlip(), RandomColorJitter()],
                                                         generators=[noise_images],
                                                         output_preprocessors=[ClipByValue(), clip_by_value])

    dataset_augmentor_val = dataset_augmentor_train

    return dataset_augmentor_train, dataset_augmentor_val


def denoisingCaeAugmentationsNoJitter(dataset_config, given_shape):
    """
    Applies augmentations for the training of denoising autoencoders.
    Augmentations: max_div_images --> RandomCropWithResize --> RandomHorizontalFlip --> noise_images --> clip_by_value
                                                                                              |--> ClipByValue
    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    max_div_images = MaxDivNormalizer(255.0)
    noise_images = RandomNormalNoise(input_name="image", output_name="noisy_image")
    clip_by_value = ClipByValue(input_name="noisy_image", output_name="noisy_image")

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[max_div_images, RandomCropWithResize(crop_shape=crop_shape), RandomHorizontalFlip()],
                                                         generators=[noise_images],
                                                         output_preprocessors=[ClipByValue(), clip_by_value])

    dataset_augmentor_val = dataset_augmentor_train

    return dataset_augmentor_train, dataset_augmentor_val

def denoisingCaeAugmentationsNoJitterNoFlip(dataset_config, given_shape):
    """
    Applies augmentations for the training of denoising autoencoders.
    Augmentations: max_div_images --> RandomCropWithResize --> noise_images --> clip_by_value
                                                                    |--> ClipByValue
    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    max_div_images = MaxDivNormalizer(255.0)
    noise_images = RandomNormalNoise(input_name="image", output_name="noisy_image")
    clip_by_value = ClipByValue(input_name="noisy_image", output_name="noisy_image")

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[max_div_images, RandomCropWithResize(crop_shape=crop_shape)],
                                                         generators=[noise_images],
                                                         output_preprocessors=[ClipByValue(), clip_by_value])

    dataset_augmentor_val = dataset_augmentor_train

    return dataset_augmentor_train, dataset_augmentor_val

def colorCaeAugmentations(dataset_config, given_shape):
    """
    Applies augmentations for the training of color restoration autoencoders.
    Augmentations: max_div_images --> RandomCropWithResize --> RandomHorizontalFlip --> jitter_images --> clip_by_value
                                                                                              |--> ClipByValue

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    max_div_images = MaxDivNormalizer(255.0)
    jitter_images = RandomColorJitter(input_name="image", output_name="jitter_image")
    clip_by_value = ClipByValue(input_name="jitter_image", output_name="jitter_image")

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[max_div_images, RandomCropWithResize(crop_shape=crop_shape), RandomHorizontalFlip()],
                                                         generators=[jitter_images],
                                                         output_preprocessors=[ClipByValue(), clip_by_value])

    dataset_augmentor_val = dataset_augmentor_train

    return dataset_augmentor_train, dataset_augmentor_val

def colorCaeAugmentations(dataset_config, given_shape):
    """
    Applies augmentations for the training of color restoration autoencoders.
    Augmentations: max_div_images --> RandomCropWithResize --> RandomHorizontalFlip --> jitter_images --> clip_by_value
                                                                                              |--> ClipByValue

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    max_div_images = MaxDivNormalizer(255.0)
    jitter_images = RandomColorJitter(input_name="image", output_name="jitter_image")
    clip_by_value = ClipByValue(input_name="jitter_image", output_name="jitter_image")

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[max_div_images, RandomCropWithResize(crop_shape=crop_shape), RandomHorizontalFlip()],
                                                         generators=[jitter_images],
                                                         output_preprocessors=[ClipByValue(), clip_by_value])

    dataset_augmentor_val = dataset_augmentor_train

    return dataset_augmentor_train, dataset_augmentor_val

def colorCaeAugmentationsNoFlip(dataset_config, given_shape):
    """
    Applies augmentations for the training of color restoration autoencoders.
    Augmentations: max_div_images --> RandomCropWithResize --> RandomHorizontalFlip --> jitter_images --> clip_by_value
                                                                                              |--> ClipByValue

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    max_div_images = MaxDivNormalizer(255.0)
    jitter_images = RandomColorJitter(input_name="image", output_name="jitter_image")
    clip_by_value = ClipByValue(input_name="jitter_image", output_name="jitter_image")

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[max_div_images, RandomCropWithResize(crop_shape=crop_shape)],
                                                         generators=[jitter_images],
                                                         output_preprocessors=[ClipByValue(), clip_by_value])

    dataset_augmentor_val = dataset_augmentor_train

    return dataset_augmentor_train, dataset_augmentor_val


def rotationCaeAugmentations(dataset_config, given_shape):
    """
    Applies augmentations for the training of color restoration autoencoders.
    Augmentations: max_div_images --> RandomCropWithResize --> RandomHorizontalFlip --> RandomColorJitter --> Random90xRotation --> clip_by_value

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """
    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    max_div_images = MaxDivNormalizer(255.0)

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[max_div_images, RandomCropWithResize(crop_shape=crop_shape), RandomHorizontalFlip(), RandomColorJitter()],
                                                         generators=[Random90xRotation()],
                                                         output_preprocessors=[ClipByValue()])

    dataset_augmentor_val = dataset_augmentor_train

    return dataset_augmentor_train, dataset_augmentor_val

def rotationCaeAugmentationsNoJitter(dataset_config, given_shape):
    """
    Applies augmentations for the training of color restoration autoencoders.
    Augmentations: max_div_images --> RandomCropWithResize --> RandomHorizontalFlip --> Random90xRotation --> clip_by_value

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """
    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    max_div_images = MaxDivNormalizer(255.0)

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[max_div_images, RandomCropWithResize(crop_shape=crop_shape), RandomHorizontalFlip()],
                                                         generators=[Random90xRotation()],
                                                         output_preprocessors=[ClipByValue()])

    dataset_augmentor_val = dataset_augmentor_train

    return dataset_augmentor_train, dataset_augmentor_val

def rotationCaeAugmentationsNoJitterNoFlip(dataset_config, given_shape):
    """
    Applies augmentations for the training of color restoration autoencoders.
    Augmentations: max_div_images --> RandomCropWithResize --> Random90xRotation --> clip_by_value

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """
    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    max_div_images = MaxDivNormalizer(255.0)

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[max_div_images, RandomCropWithResize(crop_shape=crop_shape)],
                                                         generators=[Random90xRotation()],
                                                         output_preprocessors=[ClipByValue()])

    dataset_augmentor_val = dataset_augmentor_train

    return dataset_augmentor_train, dataset_augmentor_val

def simClrAugmentations(dataset_config, given_shape):
    """
    Applies augmentations for the training of contrastive learning.
    Augmentations: max_div_images --> random_crop_1 --> random_flip_2 --> jitter_images_2 --> clip_2
                                            |--> random_crop_2--> random_flip_1 --> jitter_images_1--> clip_1

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    max_div_images = MaxDivNormalizer(255.0)

    # Order matters here! First image to image2!
    random_crop_1 = RandomCropWithResize(input_name="image", output_name="image2", crop_shape=crop_shape)
    random_crop_2 = RandomCropWithResize(input_name="image", output_name="image", crop_shape=crop_shape)

    random_flip_1 = RandomHorizontalFlip(input_name="image", output_name="image")
    random_flip_2 = RandomHorizontalFlip(input_name="image2", output_name="image2")

    jitter_images_1 = RandomColorJitter(input_name="image", output_name="image")
    jitter_images_2 = RandomColorJitter(input_name="image2", output_name="image2")

    clip_1 = ClipByValue(input_name="image", output_name="image")
    clip_2 = ClipByValue(input_name="image2", output_name="image2")

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[max_div_images],
                                                         generators=[random_crop_1, random_crop_2, random_flip_1, random_flip_2, jitter_images_1, jitter_images_2],
                                                         output_preprocessors=[clip_1, clip_2])

    dataset_augmentor_val = dataset_augmentor_train

    return dataset_augmentor_train, dataset_augmentor_val

def simClrAugmentationsNoJitter(dataset_config, given_shape):
    """
    Applies augmentations for the training of contrastive learning.
    Augmentations: max_div_images --> random_crop_1 --> random_flip_2  --> clip_2
                                            |--> random_crop_2--> random_flip_1 --> clip_1

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    max_div_images = MaxDivNormalizer(255.0)

    # Order matters here! First image to image2!
    random_crop_1 = RandomCropWithResize(input_name="image", output_name="image2", crop_shape=crop_shape)
    random_crop_2 = RandomCropWithResize(input_name="image", output_name="image", crop_shape=crop_shape)

    random_flip_1 = RandomHorizontalFlip(input_name="image", output_name="image")
    random_flip_2 = RandomHorizontalFlip(input_name="image2", output_name="image2")

    clip_1 = ClipByValue(input_name="image", output_name="image")
    clip_2 = ClipByValue(input_name="image2", output_name="image2")

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[max_div_images],
                                                         generators=[random_crop_1, random_crop_2, random_flip_1, random_flip_2],
                                                         output_preprocessors=[clip_1, clip_2])

    dataset_augmentor_val = dataset_augmentor_train

    return dataset_augmentor_train, dataset_augmentor_val

def simClrAugmentationsNoJitterNoFlip(dataset_config, given_shape):
    """
    Applies augmentations for the training of contrastive learning.
    Augmentations: max_div_images --> random_crop_1 --> clip_2
                                            |--> random_crop_2 --> clip_1

    :param given_shape: (Array) Array containing the shape of the data from the input pipeline. E.g. [None,None,3]
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: dataset_augmentor_train: (DatasetAugmentationBuilder) Handles the augmentation of the training dataset.
    :return: dataset_augmentor_val: (DatasetAugmentationBuilder) Handles the augmentation of the validation datasets.
    """

    data_shape = dataset_config["dataShape"]
    crop_shape = [1, data_shape[0], data_shape[1], data_shape[2]]

    max_div_images = MaxDivNormalizer(255.0)

    # Order matters here! First image to image2!
    random_crop_1 = RandomCropWithResize(input_name="image", output_name="image2", crop_shape=crop_shape)
    random_crop_2 = RandomCropWithResize(input_name="image", output_name="image", crop_shape=crop_shape)

    clip_1 = ClipByValue(input_name="image", output_name="image")
    clip_2 = ClipByValue(input_name="image2", output_name="image2")

    dataset_augmentor_train = DatasetAugmentationBuilder(preprocessors=[max_div_images],
                                                         generators=[random_crop_1, random_crop_2],
                                                         output_preprocessors=[clip_1, clip_2])

    dataset_augmentor_val = dataset_augmentor_train

    return dataset_augmentor_train, dataset_augmentor_val


