import os
import sys
import time

import numpy as np
import tensorflow as tf

from abc import ABCMeta

from LoggerNames import LoggerNames
from Logger_Component.SLoggerHandler import SLoggerHandler
from Output_Component.Summary.TensorboardSummaryManager import TensorboardSummaryManager
from Output_Component.Summary.TxtSummaryWriter import TxtSummaryWriter
from Output_Component.Summary.TxtFunctionTimeStopper import TxtFunctionTimeStopper

class AModelSuit(metaclass=ABCMeta):
    """
    A AModelSuit handles the train/eval/test/inference of a model. Therefore it brings the input, the model and
    the trainer, together in one place. In each AModelSuit functions for the training and validation must be defined.

    The AModelSuit provides basic functionality like model saving and defines interface methods for ModelSuits.

    :Attributes:
        _model:                         ("Model") The model to handle with the ModelSuit.
        _dataset:                       (Dictionary) The dataset to train/eval/test the model.
        _trainer:                       (ITrainer) The trainer to train the model.
        _batch_size:                    (Integer) The batch size for the model.
        _batches_in_epoch:              (Integer) The number of batches in one training epoch.
        _logger:                        (Logger) The logger for the ModelSuit.
        _model_dir:                     (String) The directory of the model (e.g. to save it).
        _save_checkpoint_steps:         (Integer) Every save_checkpoint_steps steps the ModelSuit saves model
                                            (training) checkpoints. 500 by default. (Optional) set to -1 if not needed.
        _save_checkpoint_epochs:        (Integer) Every save_checkpoint_epochs epochs the ModelSuit saves model
                                            (training) checkpoints. 1 by default. (Optional) set to -1 if not needed. 
                                            List of epochs supported (e.g. [1,5] saves only a checkpoint in the first and fifth epoch)
        _log_steps:                     (Integer) Every log_steps steps the ModelSuit writes logs. 100 by default. (Optional) set to -1 if not needed.
        _log_epochs:                    (Integer) Every log_epoch epochs the ModelSuit writes logs. 1 by default. (Optional) set to -1 if not needed.
        _save_summary_steps:            (Integer) Every save_summary_steps steps the ModelSuit saves Tensorboard summaries. 250 by default. 
                                            (Optional) set to -1 if not needed.
        _save_summary_steps:            (Integer) Every save_summary_epoch epochs the ModelSuit saves Tensorboard summaries. 1 by default. 
                                            (Optional) set to -1 if not needed.
        _ckpt:                          (tf.train.Checkpoint) Variable for the current checkpoint.
        _ckpt_manager:                  (tf.train.CheckpointManager) To manage the checkpoint.
        _summary_manager:               (TensorboardSummaryManager) The writer/manager for the Tensorboard summaries.
        _summary_txt_writer:            (TxtSummaryWriter) The writer for the text summaries.
        _txt_function_time_stopper:     (TxtFunctionTimeStopper) The writer and stopper of function times.
        __first_round:                  (Dictionary of three Booleans) Is it the first round of training/evaluation/test?
    """

    def __init__(self, model, trainer, dataset, batch_size, batches_in_epoch, model_dir="/model", save_checkpoint_steps=500, save_checkpoint_epochs=1,
                 log_steps=100, log_epochs=1, save_summary_steps=250, save_summary_epochs=1, load_checkpoint="latest"):
        """
        Constructor, initialize member variables.
        :param model: ("Model") The model to handle with the ModelSuit
        :param trainer: (ITrainer) The trainer to train the model.
        :param dataset: (Dictionary) The dataset to train/eval/test the model.
        :param batch_size: (Integer) The batch size for the model.
        :param batches_in_epoch: (Integer) The number of batches in one training epoch.
        :param model_dir: (String) The directory of the model (e.g. to save it). "/model" by default.
        :param save_checkpoint_steps: (Integer) Every save_checkpoint_steps steps the ModelSuit saves model
                                        (training) checkpoints. 500 by default. (Optional) set to -1 if not needed.
        :param save_checkpoint_epochs: (Integer) Every save_checkpoint_epochs epochs the ModelSuit saves model
                                        (training) checkpoints. 1 by default. (Optional) set to -1 if not needed. 
                                        List of epochs supported (e.g. [1,5] saves only a checkpoint in the first and fifth epoch)
        :param log_steps: (Integer) Every log_steps steps the ModelSuit writes logs. 100 by default. (Optional) set to -1 if not needed.
        :param log_epochs: (Integer) Every log_epoch epochs the ModelSuit writes logs. 1 by default. (Optional) set to -1 if not needed.
        :param save_summary_steps: (Integer) Every save_summary_steps steps the ModelSuit saves Tensorboard summaries. 250 by default. 
                                    (Optional) set to -1 if not needed.
        :param save_summary_steps: (Integer) Every save_summary_epoch epochs the ModelSuit saves Tensorboard summaries. 1 by default. 
                                    (Optional) set to -1 if not needed.
        :param load_checkpoint: (Integer) Loads the given model checkpoint. "latest" by default. 
        """
        # Set model, optimizer, dataset, trainer, batch_size.
        self._model = model
        self._dataset = dataset
        self._trainer = trainer
        self._batch_size = batch_size
        self._batches_in_epoch = batches_in_epoch

        # Setting up the Loggers
        self._logger = SLoggerHandler().getLogger(LoggerNames.EXPERIMENT_C)

        # Dir to save and reload model.
        self._model_dir = os.path.dirname(sys.modules['__main__'].__file__) + "/experimentResults" + model_dir

        # Log every log_interval_steps and/or _epochs
        self._log_steps = log_steps
        self._log_epochs = log_epochs

        # Save summary every save_summary_steps and/or _epochs
        self._save_summary_steps = save_summary_steps
        self._save_summary_epochs = save_summary_epochs

        # Save checkpoints every save_checkpoints_steps and/or _epochs
        self._save_checkpoint_steps = save_checkpoint_steps
        self._save_checkpoint_epochs = save_checkpoint_epochs

        # Checkpoint variable
        self._ckpt = tf.train.Checkpoint(optimizer=self._model.optimizer, net=self._model)

        # Create a manager for the checkpoint and restore the latest  (if there is one)
        self._ckpt_manager = tf.train.CheckpointManager(self._ckpt, self._model_dir+'/tf_ckpts', max_to_keep=None)

        # Load specified checkpoint if needed, else continue training if model exists
        if load_checkpoint is not None:
            if load_checkpoint == "latest":
                restore_checkpoint = self._ckpt_manager.latest_checkpoint
            else: 
                restore_checkpoint = self._model_dir+'/tf_ckpts/ckpt-' + str(load_checkpoint)

        if restore_checkpoint:   
            self._ckpt.restore(restore_checkpoint).assert_existing_objects_matched()
            self._logger.info("Restored model from {}".format(restore_checkpoint), "AModelSuit:__init__")
        else:
            self._logger.info("No checkpoint found. Initializing model from scratch", "AModelSuit:__init__")

        # To save summary.
        self._summary_manager = TensorboardSummaryManager(self._model_dir)
        self._summary_txt_writer = TxtSummaryWriter(self._model_dir)
        self._txt_function_time_stopper = TxtFunctionTimeStopper(self._model_dir)

        # Is it the first round of training testing or evaluation?
        self.__first_round = {"train": True, "eval": True, "test": True}
        
    def doTraining(self, train_steps, eval_steps, train_epochs=-1, eval_epochs=-1, only_save_best_checkpoints=False):
        """
        Trains the model with the trainer and the input of the ModelSuit.
        :param train_steps: (Integer) The steps to train the model. (Optional) set to -1 if not needed.
        :param eval_steps: (Integer) Every eval_steps steps the Model will be evaluated. (Optional) set to -1 if not needed.
        :param train_epochs: (Integer) The epochs to train the model. (Optional) set to -1 if not needed. -1 by default.
        :param eval_epochs: (Integer) Every eval_epochs epochs the Model will be evaluated. (Optional) set to -1 if not needed. -1 by default.
        :param only_save_best_checkpoints: (Boolean) If true only the best Model checkpoints on the evaluation set will
                                            be saved. Not used.
        """
        self._logger.train("Started training for " + str(train_steps) + " steps or "+str(train_epochs) +
                           " epochs. Evaluation every " + str(eval_steps) + " steps and/or " +str(eval_epochs) +
                           " epochs.",  "AModelSuit:doTraining")
        self._logger.train("Eager Execution: " + str(tf.executing_eagerly()), "AModelSuit:doTraining")
        self._logger.train("Eager Keras Model: " + str(self._model.run_eagerly), "AModelSuit:doTraining")

        # Stop times.
        start_training_time = time.time()
        start_log_loss_steps_time = time.time()
        start_log_loss_epochs_time = time.time()

        # Training variables.
        best_loss = 999999999

        current_step = self._model.optimizer.iterations.numpy()
        current_epoch = current_step//self._batches_in_epoch

        # Check if the model is already trained for the given steps or epochs
        if train_steps != -1:
            if current_step >= train_steps:
                return
        elif train_epochs != -1:
            if current_epoch >= train_epochs:
                return

        # Save first checkpoint with random weights
        if current_step == 0:
            save_path = self._ckpt_manager.save(checkpoint_number=0)
            self._logger.train("Saved checkpoint for step {}: {}".format(current_step, save_path), "AModelSuit:doTraining")
            # If evaluation is wished do validation.
            if (eval_steps > 0) or (eval_epochs > 0):
                eval_losses, eval_acc = self.doValidation("eval")
                eval_loss = eval_losses[0]
                # And if only_save_best_checkpoints is set save initil best losses and 
                if only_save_best_checkpoints:
                    best_loss = eval_loss
                    best_acc = eval_acc
                    best_losses = eval_losses
                    best_current_step = current_step
                    best_current_epoch = current_epoch
        self._model.resetMetrics()

        # If the model is not trained start training
        training_not_finished = True
        while training_not_finished:
            start_epoch_time = time.time()
            for data in self._dataset["train"]:

                # If its the first round of training trace the graph.
                #if self.__first_round["train"]:
                    #tf.summary.trace_on(graph=True, profiler=True)

                # Perform a training step.
                outputs = self._trainer.trainingStep(self._model, data)

                # If its the first round of training, add the graph trace to the summary.
                #if self.__first_round["train"]:
                    #with self._summary_manager.writer("train").as_default():
                    #    tf.summary.trace_export(name="train_initial_trace", step=0, profiler_outdir=self._model_dir)   
                #    self.__first_round["train"] = False

                # Get training values and metrics.
                losses = [outputs[0][i].numpy() for i in range(0, len(outputs[0]))]
                metrics = self._model.getMetrics()
                acc_value = metrics[0].numpy()

                # Increment the global step.
                current_step += 1

                # If log_steps should be saved and log_steps steps past, print the logs.
                if (self._log_steps != -1) and (current_step % self._log_steps == 0):
                    end_log_loss_steps_time = time.time()
                    self._logger.train("Step " + str(current_step) + ": " + str(self._log_steps) +
                                        " steps past in " + str(end_log_loss_steps_time - start_log_loss_steps_time)
                                            + "s. Acc: " + str(acc_value * 100) + "%. Losses: " + str(losses),
                                        "AModelSuit:doTraining")
                    start_log_loss_steps_time = time.time()

                # If a summary should be saved and save_summary_steps steps past, save the summary.
                if (self._save_summary_steps != -1) and (current_step % self._save_summary_steps == 0):
                    with self._summary_manager.writer("train").as_default():
                        self._model.writeSummary([outputs, metrics], current_step)
                        self._summary_manager.writer("train").flush()

                # If log_checkpoint_steps should be saved, save checkpoint every save_checkpoint_steps iterations 
                # if only_save_best_checkpoints is not set.
                if (self._save_checkpoint_steps != -1) and (not only_save_best_checkpoints) and (current_step % self._save_checkpoint_steps == 0):
                    save_path = self._ckpt_manager.save(checkpoint_number=current_step)
                    self._logger.train("Saved checkpoint for step {}: {}".format(current_step, save_path),
                                        "AModelSuit:doTraining")
                    self._logger.train("Losses: " + str(losses), "AModelSuit:doTraining")

                # If evaluation of steps is wished and if eval_steps steps past, do validation.
                if (eval_steps > 0) and (current_step % eval_steps == 0):
                    eval_losses, eval_acc = self.doValidation("eval")
                    eval_loss = eval_losses[0]
                    # And if only_save_best_checkpoints is set and the eval_acc is higher then the best save model.
                    if only_save_best_checkpoints and (best_loss > eval_loss):
                        save_path = self._ckpt_manager.save(checkpoint_number=current_step)
                        self._logger.train("Saved checkpoint for step {}: {}".format(current_step, save_path),
                                            "AModelSuit:doTraining")
                        self._logger.train("Eval Losses: " + str(eval_losses), "AModelSuit:doTraining")
                        best_loss = eval_loss
                        best_acc = eval_acc
                        best_losses = eval_losses
                        best_current_step = current_step
                        best_current_epoch = current_epoch

                self._model.resetMetrics()

                # Check if we at the end of the training.
                if train_steps != -1:
                    if current_step >= train_steps:
                        training_not_finished = False
                        break

            # One epoch passed
            current_epoch += 1   
                                
            # Now we repeat the same for epochs...  
            # If log_epochs should be saved and log_epochs epochs past, print the logs.
            if (self._log_epochs != -1) and (current_epoch % self._log_epochs == 0):
                end_log_loss_epochs_time = time.time()
                self._logger.train("Epoch " + str(current_epoch) + ": " + str(self._log_epochs) +
                                    " epochs past in " + str(end_log_loss_epochs_time - start_log_loss_epochs_time)
                                        + "s. Acc: " + str(acc_value * 100) + "%. Losses: " + str(losses),
                                    "AModelSuit:doTraining")
                start_log_loss_epochs_time = time.time()
                
            # If a summary should be saved and save_summary_epochs epochs past, save the summary.
            if (self._save_summary_epochs != -1) and (current_epoch % self._save_summary_epochs == 0):
                    with self._summary_manager.writer("train").as_default():
                        self._model.writeSummary([outputs, metrics], current_step) #Summary needs the current step not epoch!
                        self._summary_manager.writer("train").flush()

            # If log_checkpoint_epochs should be saved, save checkpoint every save_checkpoint_epochs iterations 
            # if only_save_best_checkpoints is not set.
            if (self._save_checkpoint_epochs != -1) and (not only_save_best_checkpoints) and (current_epoch % self._save_checkpoint_epochs == 0):
                save_path = self._ckpt_manager.save(checkpoint_number=current_step) 
                self._logger.train("Saved checkpoint for epoch {}: {}".format(current_epoch, save_path),
                                    "AModelSuit:doTraining")
                self._logger.train("Losses: " + str(losses), "AModelSuit:doTraining")

            # If evaluation of epochs is wished and if eval_epochs epochs past, do validation.
            if (eval_epochs > 0) and (current_epoch % eval_epochs == 0):
                eval_losses, eval_acc = self.doValidation("eval")
                eval_loss = eval_losses[0]
                # And if only_save_best_checkpoints is set and the eval_acc is higher then the best save model.
                if only_save_best_checkpoints and (best_loss > eval_loss):
                    save_path = self._ckpt_manager.save(checkpoint_number=current_step)
                    self._logger.train("Saved checkpoint for epoch {}: {}".format(current_epoch, save_path),
                                        "AModelSuit:doTraining")
                    self._logger.train("Losses: " + str(losses), "AModelSuit:doTraining")
                    best_loss = eval_loss
                    best_acc = eval_acc
                    best_losses = eval_losses
                    best_current_step = current_step
                    best_current_epoch = current_epoch

            # Update the learning rate based ot the current epoch
            if hasattr(self._model, 'updateLearningRate'):
                self._model.updateLearningRate(current_epoch)

            # Check if we at the end of the training.
            if train_epochs != -1:
                if current_epoch >= train_epochs:
                    break

        # Save checkpoints and summary at the end of the training                         
        with self._summary_manager.writer("train").as_default():
            self._model.writeSummary([outputs, metrics], current_step)
            self._summary_manager.writer("train").flush()

        # Do a validation at the end
        eval_losses, eval_acc = self.doValidation("eval")
        eval_loss = eval_losses[0]

        # Save the model at the end. if only_save_best_checkpoints is not set.
        if not only_save_best_checkpoints:
            save_path = self._ckpt_manager.save(checkpoint_number=current_step)  

            self._logger.train("Saved checkpoint for step {}: {}".format(current_step, save_path),
                                "AModelSuit:doTraining")
            self._logger.train("Losses: " + str(losses), "AModelSuit:doTraining")
        elif only_save_best_checkpoints:
            # And if only_save_best_checkpoints is set and the eval_acc is higher then the best save model.
            if best_loss > eval_loss:
                save_path = self._ckpt_manager.save(checkpoint_number=current_step)
                self._logger.train("Saved checkpoint for step {}: {}".format(current_step, save_path),
                                   "AModelSuit:doTraining")
                self._logger.train("Losses: " + str(losses), "AModelSuit:doTraining")
                best_loss = eval_loss
                best_acc = eval_acc
                best_losses = eval_losses
                best_current_step = current_step
                best_current_epoch = current_epoch
            self._summary_txt_writer.writeSummary("Best Loss Epoch: " + str(best_current_epoch), "eval")
            self._summary_txt_writer.writeSummary("Best Loss Step: " + str(best_current_step), "eval")
            self._summary_txt_writer.writeSummary("Best Losses: " + str(best_losses), "eval")
            self._summary_txt_writer.writeSummary("Best Acc: " + str(best_acc), "eval")

        # Stop training time.
        end_training_time = time.time()

        self._logger.train("Finished training for " +  str(current_epoch) +  " epochs or "+ str(train_steps) +
                            " steps. Evaluation was every " + str(eval_steps) + " steps and/or " +str(eval_epochs)+  " epochs. Training duration was: " +
                            str(end_training_time - start_training_time) + "s. Final Acc: " + str(
                acc_value * 100) +"%. Final losses: " + str(losses), "AModelSuit:doTraining")


    def doValidation(self, mode):
        """
        Validates the model on the subdataset subset defined by the mode.
        :param mode: (String) The subset of the dataset ("train", "eval" or "test).
        """
        self._logger.val("Started validation for " + str(mode) + " dataset.", "AModelSuit:doValidation")
        self._logger.val("Eager Execution: " + str(tf.executing_eagerly()), "AModelSuit:doValidation")
        self._logger.val("Eager Keras Model: " + str(self._model.run_eagerly), "AModelSuit:doValidation")

        # Stop times.
        start_validation_time = time.time()
        start_log_loss_steps_time = time.time()
          
        # Evaluation variables.
        loss_values_obj = []
        loss_values_reg = []

        acc_values = []
        outputs = 0
        val_step = 0

        # Train the model on the sub dataset (one of train/eval/test).
        for data in self._dataset[mode]:
            # If its the first round of training trace the graph.
            #if self.__first_round[mode]:
            #    tf.summary.trace_on(graph=True, profiler=True)
            
            # Perform a validation step.
            outputs = self._trainer.validationStep(self._model, data)
             
            # If its the first round of training, add the graph trace to the summary.
            #if self.__first_round[mode]:
            #    with self._summary_manager.writer(mode).as_default():
            #        tf.summary.trace_export(name=str(mode)+"_initial_trace", step=0, profiler_outdir=self._model_dir)
            #    self.__first_round[mode] = False

            # Get evaluation values and metrics.
            loss_vals_obj = outputs[0][0]
            loss_vals_reg = outputs[0][1]

            metrics = self._model.getMetrics()
            acc_value = metrics[0].numpy()
            loss_values_obj.append(loss_vals_obj)
            loss_values_reg.append(loss_vals_reg)

            acc_values.append(acc_value)

             # If log_steps should be saved and log_steps steps past, print the logs.
            if (self._log_steps != -1) and (val_step % self._log_steps == 0):
                end_log_loss_steps_time = time.time()
                o_loss = np.mean(loss_vals_obj)
                r_loss = np.mean(loss_vals_reg)
                self._logger.val("Step " + str(val_step) + ": " + str(self._log_steps) +
                                 " steps past in " + str(end_log_loss_steps_time - start_log_loss_steps_time)
                                 + "s. Accuracy till now: " + str(acc_value * 100) + "%. Loss value for step: " + str(loss_vals_obj+loss_vals_reg) + 
                                 " Obj Loss value for step: " + str(loss_vals_obj) +
                                 " Reg Loss value for step: " + str(loss_vals_reg), "AModelSuit:doValidation")
                start_log_loss_steps_time = time.time()
            val_step += 1

        # Get evaluation values and metrics for epoch.
        loss_values_obj = np.concatenate(loss_values_obj)  
        outputs[0][0] = np.mean(loss_values_obj)
        outputs[0][1] = np.mean(loss_values_reg)
        outputs[0].insert(0, outputs[0][0]+outputs[0][1]) 

        metrics = self._model.getMetrics()
        self._model.resetMetrics()

        # Save checkpoints and summary at the end of the validation/epoch  
        current_step = self._model.optimizer.iterations.numpy()
        with self._summary_manager.writer(mode).as_default():
            self._model.writeSummary([outputs, metrics], current_step)
            self._summary_manager.writer(mode).flush()

        # Stop evaluation time.
        end_validation_time = time.time()

        self._logger.val("Finished validation for " + str(mode) + " dataset. Validation duration was: " +
                            str(end_validation_time - start_validation_time) + "s. Final accuracy: " + str(
                metrics[0].numpy() * 100) +"%. Final losses: " + str(outputs[0]), "AModelSuit:doValidation")

        # Write Acc and Loss in textfile.
        self._summary_txt_writer.writeSummary("Acc for epoch: " + str(metrics[0].numpy() * 100), mode)
        self._summary_txt_writer.writeSummary("Losses for epoch: " + str(outputs[0]), mode)
        return outputs[0], metrics[0].numpy() * 100

    def doDatesetValidation(self):
        """
        Validates the model on the entire dataset.
        """
        self.doValidation("train")
        self.doValidation("eval")
        self.doValidation("test")

    def saveModel(self):
        """
        Saves the model.
        """
        tf.saved_model.save(self._model, self._model_dir)

    def getModel(self):
        """
        Returns the model.
        :return: model: ("Model") The model to handle with the ModelSuit
        """
        return self._model
