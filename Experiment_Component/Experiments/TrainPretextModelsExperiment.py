import traceback

from LoggerNames import LoggerNames
from Logger_Component.SLoggerHandler import SLoggerHandler
from Experiment_Component.IExperiment import IExperiment
from ConfigInput_Component.ConfigProvider import ConfigProvider

from Experiment_Component.Experiments.experimentFunctions import trainAndValPretextModel

class TrainPretextModelsExperiment(IExperiment):
    """
    The experiment trains each pretext model for each given dataset and saves logs and checkpoints.
    If xFoldCrossValidation is given this will be repeated for all given cross-validations.

    :Attributes:
        __config:    (Dictionary) The config of the experiment, containing all pretext models parameters. Refer to the config
                      trainPretextModelsExperiment.json for an example.
        __logger:    (Logger) The logger for the experiment.
        __num_gpus:  (Integer) The number of GPUs to use.

    """
    def __init__(self, config):
        """
        Constructor, initialize member variables.
        :param config: (Dictionary) The config of the experiment, containing all pretext models parameters. Refer to the config
                        trainPretextModelsExperiment.json for an example.
        """
        self.__config = config
        self.__logger = SLoggerHandler().getLogger(LoggerNames.EXPERIMENT_C)
        self.__num_gpus = ConfigProvider().get_config("controllerConfig.json")["hardware"]["numGPUs"]

    def execute(self):
        """
        Executes the experiment with the given config.

        The experiment trains each model for each given dataset for the defined training steps.
        If xFoldCrossValidation is set this will be repeated x times.
        """
        for pretext_model_config in self.__config["pretextModelConfigs"]:
            model_name = pretext_model_config["modelName"]
            try:
                for dataset_config in self.__config["datasetConfigs"]:

                    # Only train the model if a batch size for the dataset is given
                    if not dataset_config["nameOfDataset"] in pretext_model_config["batchSizes"].keys():
                        continue

                    # and for each xFold iteration
                    for xFold_step in pretext_model_config["xFoldCrossValidation"]:

                        # If the dataset contains different sizes we want to test, we save them in seperat directories 
                        dataset_dir_name = dataset_config["nameOfDataset"]       
                        if "trainDatasetSize" in dataset_config.keys():
                            dataset_dir_name = dataset_dir_name + "_" + str(dataset_config["trainDatasetSize"])

                        # Construct the model Name
                        model_dir = "/" + model_name + "/" + dataset_dir_name + "/xFoldCrossVal_" + str(xFold_step)
                        self.__logger.info("Starting to train: " + model_dir, "TrainPretextModelsExperiment:execute")

                        # Train the model
                        self.__logger.info("Starting to train: " + model_dir, "TrainPretextModelsExperiment:execute")
                        trainAndValPretextModel(pretext_model_config, model_dir, dataset_config, xFold_step,
                                                pretext_model_config["xFoldType"], self.__num_gpus, self.__logger)
                        self.__logger.info("Finished to train: " + model_dir, "TrainPretextModelsExperiment:execute")
                           
            except:
                print(traceback.format_exc())




