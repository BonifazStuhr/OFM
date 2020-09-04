import traceback

from LoggerNames import LoggerNames
from Logger_Component.SLoggerHandler import SLoggerHandler
from Experiment_Component.IExperiment import IExperiment
from ConfigInput_Component.ConfigProvider import ConfigProvider

from Experiment_Component.Experiments.experimentFunctions import trainAndValTargetModel

class TrainTargetModelsExperiment(IExperiment):
    """
    The experiment trains each target model for each given dataset using the representation of the given
    (unsupervised) pretext models and saves logs and checkpoints.
    It trains one target model for every given checkpoint.
    If xFoldCrossValidationsToLoad is given this will be repeated for all configured cross-validations.

    :Attributes:
        __config:    (Dictionary) The config of the experiment, containing all model parameters. Refer to the config
                      trainTargetModelsExperiment.json for an example.
        __logger:    (Logger) The logger for the experiment.
        __num_gpus:  (Integer) The number of GPUs to use.

    """
    def __init__(self, config):
        """
        Constructor, initialize member variables.
        :param config: (Dictionary) The config of the experiment, containing all model parameters. Refer to the config
                        trainModelsExperiment.json as an example.
        """
        self.__config = config
        self.__logger = SLoggerHandler().getLogger(LoggerNames.EXPERIMENT_C)
        self.__num_gpus = ConfigProvider().get_config("controllerConfig.json")["hardware"]["numGPUs"]

    def execute(self):
        """
        Executes the experiment with the given config.

        The experiment trains each model for each given dataset for the defined training steps on the representation of the given 
        representation model for the desired checkpoints.
        If xFoldCrossValidation is set this will be repeated x times.
        """
        # For each model to train and val
        for target_model_config in self.__config["targetModelConfigs"]:
            target_model_name = target_model_config["modelName"]
            try:
                for dataset_config in self.__config["datasetConfigs"]:
                    for pretext_model_config in self.__config["pretextModelConfigs"]:
                        for pretext_model_xFoldToLoad in pretext_model_config["xFoldCrossValidationsToLoad"]:
                            for pretext_model_checkpoint in pretext_model_config["loadCheckpoints"]:
                                
                                # Only train the model if a batch size for the representation model is given
                                if not dataset_config["nameOfDataset"] in pretext_model_config["batchSizes"].keys():
                                    continue

                                if pretext_model_config["loadCheckpointEpochMultipliers"][dataset_config["nameOfDataset"]]:
                                    pretext_model_checkpoint = pretext_model_checkpoint * pretext_model_config["loadCheckpointEpochMultipliers"][dataset_config["nameOfDataset"]]

                                # If the dataset contains multible labels we want to test, we save them in seperat directories        
                                dataset_dir_name = dataset_config["nameOfDataset"]             
                                if "labelName" in dataset_config.keys():
                                    dataset_dir_name = dataset_dir_name + "_" + dataset_config["labelName"]

                                # If the dataset contains different sizes we want to test, we save them in seperat directories 
                                pretext_dataset_dir_name = dataset_config["nameOfDataset"]       
                                if "trainDatasetSize" in dataset_config.keys():
                                    dataset_dir_name = dataset_dir_name + "_" + str(dataset_config["trainDatasetSize"])
                                    pretext_dataset_dir_name = pretext_dataset_dir_name + "_" + str(dataset_config["trainDatasetSize"])
                                    
                                # Construct the model Name
                                target_model_dir = "/" + target_model_name + "/" + dataset_dir_name + "/" + \
                                            pretext_model_config["modelName"] + "/loadedxFoldCrossVal_" +\
                                            str(pretext_model_xFoldToLoad) + "/checkpoint_" + str(pretext_model_checkpoint)

                                # Train the model
                                self.__logger.info("Starting to train: " + target_model_dir, "TrainTargetModelsExperiment:execute")

                                # The pretext model is trained unsupervised in our case, 
                                # therefore we can train in once on the dataset at load it for differtent target tasks on this dataset.
                                # This applies for shapes3D in our case.
                                pretext_model_dir = "/" + pretext_model_config["modelName"] + "/" + pretext_dataset_dir_name + \
                                                           "/xFoldCrossVal_" + str(pretext_model_xFoldToLoad)

                                trainAndValTargetModel(target_model_config, target_model_dir, dataset_config,
                                                       pretext_model_xFoldToLoad,  pretext_model_config["xFoldType"],
                                                       self.__num_gpus, self.__logger, pretext_model_dir,
                                                       pretext_model_checkpoint, pretext_model_config)

                                self.__logger.info("Finished to train: " + target_model_dir, "TrainTargetModelsExperiment:execute")
                                
            except:
                print(traceback.format_exc())




