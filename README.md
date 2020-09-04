# Original Implementation of the Paper “Don't miss the Mismatch: Investigating the Objective Function Mismatch for Unsupervised Representation Learning"
This is the original implementation of the paper 
[“Don't miss the Mismatch: Investigating the Objective Function Mismatch for Unsupervised Representation Learning”](TODO: Paper Link) in TensorFlow 2.2.0.

## Usage
### 1. Configuration
There are two .json configuration files: ```controllerConfig.json``` and ```experimetSchedule.json``` to configure a run. 
The default settings execute the training of a pretext autoencoder and afterwards the training of target models on-top of the representations of the autoencoder for different pretext training epochs on 4 GPUs and 28 CPU cores.
Datasets will be automaticly downloaded.
#### 1.1. Setup Configuration: ```controllerConfig.json```
 Here you can configure the number of GPUs and CPU cores to use for execution:
```json
{ 
    "hardware": {
        "numGPUs": 4,
        "numCPUCores": 28
    }
}
```
#### 1.2. Choose the Experiments to run:
Here you can choose the experiments to execute (by inserting the name of the IExperiment class) and set the .json configuration file(s) for these experiments:
```json
{
  "mode": "sequential",
  "experimentsToRun": ["TrainTargetModelsExperiment", "TrainPretextModelsExperiment"],
  "experimentConfigsToRun": {"TrainPretextModelsExperiment": ["trainPretextModelsExperiment.json"],
                             "TrainTargetModelsExperiment": ["trainTargetModelsExperiment.json"]}
}
```
Experiments can be found in ```Experiment_Component/Experiments``` and their configurations can be found in ```Experiment_Component/ExperimentConfigs```
### 2. Run the Experiments
#### 2.1. Directly:
Simply run the ```main.py``` file. 
#### 2.2. Within a Docker Container:
1. Navigate to directory containing the Docker Image: ```cd Docker```
2. Built the Docker Image: ```docker image build -t my_tensorflow_docker .```
3. Navigate back to the root project directory: ```cd ..```
4. Run the Docker Image: 
   ```
    docker run -it --rm --gpus all --cpus 28 \
   -v /home/$USER/tensorflow_datasets:/root/tensorflow_datasets \
   -v $PWD:/tmp -w /tmp my_tensorflow_docker python ./main.py
   ```
## Experiment Configuration
For each experiment .json configuration files must be defined to configure the pretext and target models and their 
training parameters. Furthermore, the number of runs (e.g., for cross-validation), 
and the datasets can be defined here (depending on the experiment). 
You can use the original configuration files of the experiments from the paper as template to 
configure your own experiments. These original configuration files will be used by default, 
but you can register your own configuration files in the ```experimetSchedule.json``` file.

### Correspondence between the Paper and the Experiments
The methods to calculate M3, SM3 and the OFM and the corresponding code for the plots can be 
found in the jupyter notebooks of the directory ```/jupyterNotebooks```
1. The configuration files in the directory ```/representationSize``` corresponds to the evidence 
we give in the Tables and Figures of the paper which considers the dependents of the mismatches to the pretext model representation size.
2. The configuration files in the directory ```/augmentations``` corresponds to the evidence 
we give in the Tables and Figures of the paper which consider the dependents of the mismatches to the augmentations used in pretext and target task training.
3. The configuration files in the directory ```/targetModelComplexity``` corresponds to the evidence
we give in the Tables and Figures of the paper which consider the dependents of the mismatches to the target model complexity (e.g. a linear dense layer or a MLP).
4. The configuration files in the directory ```/targetTaskType``` corresponds to the evidence 
we give in the Tables and Figures of the paper which consider the dependents of the mismatches to the target task type (e.g. classifying object shape, hue or orientation).
5. The configuration files in the directory ```/others``` corresponds to the evidence we give in the paper considering stability and mismatches on ResNets.

## Abstract of the Paper
Finding general evaluation metrics for unsupervised representation learning techniques is a challenging open research question, which recently has become more and more necessary due to the increasing interest in unsupervised methods. Even though these methods promise beneficial representation characteristics, most approaches currently suffer from the objective function mismatch. This mismatch states that the performance on a desired target task can decrease when the unsupervised pretext task is learned too long - especially when both tasks are ill-posed. In this work, we build upon the widely used linear evaluation protocol and define new general evaluation metrics to quantitatively capture the objective function mismatch and the more generic metrics mismatch. We discuss the usability and stability of our protocols on a variety of pretext and target tasks and study mismatches in a wide range of experiments. Thereby we disclose dependencies of the objective function mismatch across several pretext and target tasks with respect to the pretext model's representation size, target model complexity, pretext and target augmentations as well as pretext and target task types.​

Please read our paper for more details.

## License
[MIT](https://choosealicense.com/licenses/mit/)
