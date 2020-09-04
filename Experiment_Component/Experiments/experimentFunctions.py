import importlib

import tensorflow as tf
import tensorflow_datasets as tfds

from Experiment_Component.Trainer.SingleGpuTrainer import SingleGpuTrainer
from Experiment_Component.Trainer.MultiGpuTrainer import MultiGpuTrainer

from Experiment_Component.AModelSuit import AModelSuit

def createPretextModel(model_config, dataset_config):
    """
    Function which creates the pretext model for the given model_config and dataset_config.
    :param model_config: (Dictionary) The configuration of the model containing all hyperparameters.
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: model: (tf.keras.Model) The pretext model.
    """
    input_module = importlib.import_module("Experiment_Component.Models.PretextModels." + model_config["modelClassName"])
    model = getattr(input_module, model_config["modelClassName"])(model_config, input_shape=dataset_config["dataShape"])
    return model

def createTargetModel(model_config, dataset_config, pretext_model_suit):
    """
    Function which creates the pretext model for the given model_config and dataset_config.
    :param model_config: (Dictionary) The configuration of the model containing all hyperparameters.
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :param pretext_model_suit: (Dictionary) The model_suit which handles the pretext model.
    :return: model: (tf.keras.Model) The target model.
    """
    pretext_model = pretext_model_suit.getModel()
    input_shape = pretext_model.getRepresentationShape()[1:]

    input_module = importlib.import_module("Experiment_Component.Models.TargetModels." + model_config["modelClassName"])
    model = getattr(input_module, model_config["modelClassName"])(model_config, dataset_config["numClasses"],
                                                                  input_shape, pretext_model)
    return model

def createAugmentations(augmentation_name):
    """
    Function which creates the augmentations for the given augmentation name.
    :param augmentation_name: (String) The name of the augmentations.
    :return: augmentations: (Function) Function to get the augmentations for the dataset.
    """
    input_module = importlib.import_module("Experiment_Component.Experiments.augmentations")
    augmentations = getattr(input_module, augmentation_name)
    return augmentations

def loadInputPipeline(dataset_config, xfold_step, xfold_type):
    """
    Function which loads the input pipeline of the dataset specified in the given dataset_config for the given
    xFold_type (e.g. 5 Fold) and the given xFold_step

    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :param xfold_step: (Integer) The step of the current xFold-cross-validation.
    :param xfold_type: (Integer) The type for the xFold operation.
    :return: train_ds: (tf.data.Dataset) The dataset + input pipeline for training.
    :return: eval_ds: (tf.data.Dataset) The dataset + input pipeline for evaluation.
    :return: test_ds: (tf.data.Dataset) The dataset + input pipeline for testing.
    :return: data_shape: (Array) The shape of the input of the dataset.
    """
    name = dataset_config["nameOfDataset"]

    xfold_step_percent = 100 // xfold_type
    xfold_current_percent = xfold_step_percent * xfold_step

    if name == "patch_camelyon":
        # x-fold cross-validation:
        eval_ds, eval_ds_info = tfds.load(name,split=f'validation[{xfold_current_percent}%:{xfold_current_percent + xfold_step_percent}%]',
                                          with_info=True)
        train_ds, train_ds_info = tfds.load(name, split=f'validation[:{xfold_current_percent}%]+validation[{xfold_current_percent + xfold_step_percent}%:]',
                                            with_info=True)
        test_ds, test_ds_info = tfds.load(name, split='test', with_info=True)
    elif (name == "shapes3d") or (name == "eurosat/rgb"):
        # x-fold cross-validation:
        eval_ds, eval_ds_info = tfds.load(name, split=f'train[{xfold_current_percent}%:{xfold_current_percent + xfold_step_percent}%]', with_info=True)
        train_ds, train_ds_info = tfds.load(name, split=f'train[:{xfold_current_percent}%]+train[{xfold_current_percent + xfold_step_percent}%:]', with_info=True)
        test_ds, test_ds_info = eval_ds, eval_ds_info # shapes3d has no official test set. Since we report the mismatch on 5 fold cross-valiadation on the eval data thats no problem.
    else:
        # x-fold cross-validation:
        train_ds, train_ds_info = tfds.load(name, split=f'train[:{xfold_current_percent}%]+train[{xfold_current_percent + xfold_step_percent}%:]', with_info=True)
        eval_ds, eval_ds_info = tfds.load(name, split=f'train[{xfold_current_percent}%:{xfold_current_percent + xfold_step_percent}%]', with_info=True)
        test_ds, test_ds_info = tfds.load(name, split='test', with_info=True)
       
    data_shape = train_ds_info.features["image"].shape

    if dataset_config["cache"]:
        train_ds = train_ds.cache()
        eval_ds = eval_ds.cache()
        test_ds = test_ds.cache()

    return train_ds, eval_ds, test_ds, data_shape

def prepareTfInputPipeline(dataset_config, xfold_step, xfold_type, batch_size, augmentation_name):
    """
    Function which loads prepares the input pipeline of the dataset specified in the given dataset_config for the given
    xFold_type (e.g. 5 Fold) and the given xFold_step. Augmentationswill be added and the dataset is batched.

    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :param xfold_step: (Integer) The step of the current xFold-cross-validation.
    :param xfold_type: (Integer) The type for the xFold operation.
    :param batch_size: (Integer) The batch size to use for the input pipeline.
    :return: augmentation_name: (String) The name of the augmentations to use.
    :return: train_ds: (tf.data.Dataset) The dataset + input pipeline for training.
    :return: eval_ds: (tf.data.Dataset) The dataset + input pipeline for evaluation.
    :return: test_ds: (tf.data.Dataset) The dataset + input pipeline for testing.
    """

    train_ds, eval_ds, test_ds, given_shape = loadInputPipeline(dataset_config, xfold_step, xfold_type)

    len_train = tf.data.experimental.cardinality(train_ds).numpy()
    len_eval = tf.data.experimental.cardinality(eval_ds).numpy()
    len_test = tf.data.experimental.cardinality(test_ds).numpy()

    train_ds = train_ds.shuffle(len_train, reshuffle_each_iteration=True)
    # Shuffel eval and test set to prevent spikes in error due to batch size
    eval_ds = eval_ds.shuffle(len_eval, reshuffle_each_iteration=False, seed=42)
    test_ds = test_ds.shuffle(len_test, reshuffle_each_iteration=False, seed=42)

    if augmentation_name:
        augmentations = createAugmentations(augmentation_name)
        dataset_augmentor_train, dataset_augmentor_val = augmentations(dataset_config, given_shape)

        train_ds = train_ds.map(dataset_augmentor_train.generate_sample_output, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        eval_ds = eval_ds.map(dataset_augmentor_val.generate_sample_output, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.map(dataset_augmentor_val.generate_sample_output, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #fig = tfds.show_examples(train_ds, train_ds_info)
    #fig = tfds.show_examples(eval_ds, eval_ds_info)
    #fig = tfds.show_examples(test_ds, test_ds_info)

    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    eval_ds = eval_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return train_ds, eval_ds, test_ds

def trainAndValPretextModel(pretext_model_config, pretext_model_dir, dataset_config, xfold_step, xfold_type, num_gpus, logger):
    """
    Trains and additionally evaluates the model defined in the model_config with the given dataset on num_gpus gpus
    and saves the model to the path model_dir.
    :param pretext_model_config: (Dictionary) The configuration of the model containing all hyperparameters.
    :param pretext_model_dir: (String) The path to the directory in which the model is saved.
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :param xfold_step: (Integer) The step of the current xFold-cross-validation.
    :param xfold_type: (Integer) The type for the xFold operation.
    :param num_gpus: (Integer) The number of gpus to train and val with.
    :param logger: (Logger) The logger for the experiment.
    """
    batch_size = pretext_model_config["batchSizes"][dataset_config["nameOfDataset"]]

    # Define the model trainer
    if (num_gpus <= 1) or (batch_size < num_gpus):
        trainer = SingleGpuTrainer()
    else:
        trainer = MultiGpuTrainer(global_batch_size=batch_size)  

    with trainer.scope():
        model = createPretextModel(pretext_model_config, dataset_config)
        train_ds, eval_ds, test_ds = prepareTfInputPipeline(dataset_config, xfold_step, xfold_type, batch_size, pretext_model_config["augmentationsName"])
        batches_in_epoch = tf.data.experimental.cardinality(train_ds).numpy()

        logger.info("Batches in epoch for: " + pretext_model_dir + ": " + str(batches_in_epoch), "OfmExperiment:execute")
        logger.tfDatasetInfo("train", train_ds, ":execute")
        logger.tfDatasetInfo("eval", eval_ds, ":execute")
        logger.tfDatasetInfo("test", test_ds, ":execute")

        train_ds, eval_ds, test_ds = trainer.distributeDataset(train_ds, eval_ds, test_ds)
        dataset = {"train": train_ds, "eval": eval_ds, "test": test_ds}
            
        model_suit = AModelSuit(model, trainer, dataset, batch_size, batches_in_epoch, model_dir=pretext_model_dir,
                                    save_checkpoint_steps=pretext_model_config["saveCheckpointSteps"],
                                    save_checkpoint_epochs=pretext_model_config["saveCheckpointEpochs"],
                                    log_steps=pretext_model_config["logSteps"],
                                    log_epochs=pretext_model_config["logEpochs"],
                                    save_summary_steps=pretext_model_config["saveSummarySteps"],
                                    save_summary_epochs=pretext_model_config["saveSummaryEpochs"])

        model_suit.doTraining(pretext_model_config["trainingSteps"], pretext_model_config["evalSteps"],
                              pretext_model_config["trainingEpochs"], pretext_model_config["evalEpochs"])

        if pretext_model_config["doDatasetValidation"]:
            model_suit.doDatesetValidation()
        
    tf.keras.backend.clear_session()

  
def trainAndValTargetModel(target_model_config, target_model_dir, dataset_config, xfold_step, xfold_type, num_gpus, logger,
                                     pretext_model_dir, pretext_model_checkpoint, pretext_model_config):
    """
    Trains and additionally evaluates the model defined in the model_config with the given dataset on num_gpus gpus
    and saves the model to the path model_dir.
    :param target_model_config: (Dictionary) The configuration of the model containing all hyperparameters.
    :param target_model_dir: (String) The path to the directory in which the model is saved.
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :param xfold_step: (Integer) The step of the current xFold-cross-validation.
    :param xfold_type: (Integer) The type for the xFold operation.
    :param num_gpus: (Integer) The number of gpus to train and val with.
    :param logger: (Logger) The logger for the experiment.
    :param pretext_model_dir: (String) The path to the directory in which the model is saved.
    :param pretext_model_checkpoint: (Integer) The checkpoint of the representation model to load.
    :param pretext_model_config: (Dictionary) The configuration of the representation model containing all hyperparameters.
    """
    batch_size = pretext_model_config["batchSizes"][dataset_config["nameOfDataset"]]
    # Define the model trainer
    if (num_gpus <= 1) or (batch_size < num_gpus):
        trainer = SingleGpuTrainer()
    else:
        trainer = MultiGpuTrainer(global_batch_size=batch_size)  

    with trainer.scope():
        train_ds, eval_ds, test_ds = prepareTfInputPipeline(dataset_config, xfold_step, xfold_type, batch_size, pretext_model_config["augmentationsName"])

        batches_in_epoch = tf.data.experimental.cardinality(train_ds).numpy()

        logger.info("Batches in epoch for: " + target_model_dir + ": " + str(batches_in_epoch), "OfmExperiment:execute")
        logger.tfDatasetInfo("train", train_ds, ":execute")
        logger.tfDatasetInfo("eval", eval_ds, ":execute")
        logger.tfDatasetInfo("test", test_ds, ":execute")

        train_ds, eval_ds, test_ds = trainer.distributeDataset(train_ds, eval_ds, test_ds)
        dataset = {"train": train_ds, "eval": eval_ds, "test": test_ds}

        pretext_model = createPretextModel(pretext_model_config, dataset_config)
        pretext_model_suit = AModelSuit(pretext_model, None, dataset, batch_size, batches_in_epoch,
                                        model_dir=pretext_model_dir, load_checkpoint=pretext_model_checkpoint)

        target_model = createTargetModel(target_model_config, dataset_config, pretext_model_suit)
        target_model_suit = AModelSuit(target_model, trainer, dataset, batch_size, batches_in_epoch, model_dir=target_model_dir,
                                    save_checkpoint_steps=target_model_config["saveCheckpointSteps"],
                                    save_checkpoint_epochs=target_model_config["saveCheckpointEpochs"],
                                    log_steps=target_model_config["logSteps"],
                                    log_epochs=target_model_config["logEpochs"],
                                    save_summary_steps=target_model_config["saveSummarySteps"],
                                    save_summary_epochs=target_model_config["saveSummaryEpochs"])

        target_model_suit.doTraining(target_model_config["trainingSteps"], target_model_config["evalSteps"],
                              target_model_config["trainingEpochs"], target_model_config["evalEpochs"],
                              only_save_best_checkpoints=True)

        if target_model_config["doDatasetValidation"]:
            target_model_suit.doDatesetValidation()

    tf.keras.backend.clear_session()