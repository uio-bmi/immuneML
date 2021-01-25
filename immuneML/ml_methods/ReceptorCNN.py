import copy
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_methods.pytorch_implementations.PyTorchReceptorCNN import PyTorchReceptorCNN as RCNN
from immuneML.ml_methods.util.Util import Util
from immuneML.util.PathBuilder import PathBuilder


class ReceptorCNN(MLMethod):
    """
    A CNN which separately detects motifs using CNN kernels in each chain of paired receptor data, combines the kernel activations into a unique
    representation of the receptor and uses this representation to predict the antigen binding.

    .. figure:: _static/images/receptor_cnn_immuneML.png
        :width: 70%

        The architecture of the CNN for paired-chain receptor data

    Requires one-hot encoded data as input (as produced by :ref:`OneHot` encoder).

    Note: ReceptorCNN can only be used with ReceptorDatasets, it does not work with SequenceDatasets


    Arguments:

        kernel_count (count): number of kernels that will look for motifs for one chain

        kernel_size (list): sizes of the kernels = how many amino acids to consider at the same time in the chain sequence, can be a tuple of
        values; e.g. for value [3, 4] of kernel_size, kernel_count*len(kernel_size) kernels will be created, with kernel_count kernels of size 3
        and kernel_count kernels of size 4 per chain

        positional_channels (int): how many positional channels where included in one-hot encoding of the receptor sequences (default is 3 in one-hot encoder)

        sequence_type (SequenceType): type of the sequence

        device: which device to use for the model (cpu or gpu) - for more details see PyTorch documentation on device parameter

        number_of_threads (int): how many threads to use

        random_seed (int): number used as a seed for random initialization

        learning_rate (float): learning rate scaling the step size for optimization algorithm

        iteration_count (int): for how many iterations to train the model

        l1_weight_decay (float): weight decay l1 value for the CNN; encourages sparser representations

        l2_weight_decay (float): weight decay l2 value for the CNN; shrinks weight coefficients towards zero

        batch_size (int): how many receptors to process at once

        training_percentage (float): what percentage of data to use for training (the rest will be used for validation); values between 0 and 1

        evaluate_at (int): when to evaluate the model, e.g. every 100 iterations

        background_probabilities: used for rescaling the kernel values to produce information gain matrix; represents the background probability of
        each amino acid (without positional information); if not specified, uniform background is assumed

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_receptor_cnn:
            ReceptorCNN:
                kernel_count: 5
                kernel_size: [3]
                positional_channels: 3
                sequence_type: amino_acid
                device: cpu
                number_of_threads: 16
                random_seed: 100
                learning_rate: 0.01
                iteration_count: 10000
                l1_weight_decay: 0
                l2_weight_decay: 0
                batch_size: 5000

    """

    def __init__(self, kernel_count: int = None, kernel_size=None, positional_channels: int = None, sequence_type: str = None, device=None,
                 number_of_threads: int = None, random_seed: int = None, learning_rate: float = None, iteration_count: int = None,
                 l1_weight_decay: float = None, l2_weight_decay: float = None, batch_size: int = None, training_percentage: float = None,
                 evaluate_at: int = None, background_probabilities=None, result_path:Path=None):

        super().__init__()
        self.kernel_count = kernel_count
        self.kernel_size = kernel_size
        self.positional_channels = positional_channels
        self.number_of_threads = number_of_threads
        self.random_seed = random_seed
        self.device = device
        self.l1_weight_decay = l1_weight_decay
        self.l2_weight_decay = l2_weight_decay
        self.learning_rate = learning_rate
        self.iteration_count = iteration_count
        self.batch_size = batch_size
        self.evaluate_at = evaluate_at
        self.training_percentage = training_percentage
        self.sequence_type = SequenceType[sequence_type.upper()]
        self.background_probabilities = background_probabilities if background_probabilities is not None \
            else np.array([1. / len(EnvironmentSettings.get_sequence_alphabet(self.sequence_type))
                           for i in range(len(EnvironmentSettings.get_sequence_alphabet(self.sequence_type)))])
        self.CNN = None
        self.label_name = None
        self.class_mapping = None
        self.result_path = result_path
        self.chain_names = None
        self.feature_names = None

    def predict(self, encoded_data: EncodedData, label_name: str):
        predictions_proba = self.predict_proba(encoded_data, label_name)
        return {self.label_name: [self.class_mapping[val] for val in (predictions_proba[self.label_name][:, 1] > 0.5).tolist()]}

    def predict_proba(self, encoded_data: EncodedData, label_name: str):
        # set the model to evaluation mode for inference
        self.CNN.eval()

        # convert encoded data from numpy arrays to tensors
        encoded_data_pt = self._make_encoded_data(encoded_data, np.arange(len(encoded_data.example_ids)))

        # make predictions
        with torch.no_grad():
            predictions = []
            for examples, labels, example_ids in self._get_data_batch(encoded_data_pt, label_name):
                logit_outputs = self.CNN(examples)
                prediction = torch.sigmoid(logit_outputs)
                predictions.extend(prediction.numpy())

        return {self.label_name: np.vstack([1 - np.array(predictions), predictions]).T}

    def fit(self, encoded_data: EncodedData, label_name: str, cores_for_training: int = 2):

        self.feature_names = encoded_data.feature_names

        Util.setup_pytorch(self.number_of_threads, self.random_seed)
        if "chain_names" in encoded_data.info and encoded_data.info["chain_names"] is not None and len(encoded_data.info["chain_names"]) == 2:
            self.chain_names = encoded_data.info["chain_names"]
        else:
            self.chain_names = ["chain_1", "chain_2"]

        self._make_CNN()
        self.CNN.to(device=self.device)

        self.class_mapping = Util.make_binary_class_mapping(encoded_data.labels[label_name], label_name)
        self.label_name = label_name

        self.CNN.train()

        iteration = 0
        loss_function = nn.BCEWithLogitsLoss().to(device=self.device)
        optimizer = torch.optim.Adam(self.CNN.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight_decay, eps=1e-4)
        state = dict(model=copy.deepcopy(self.CNN).state_dict(), optimizer=optimizer, iteration=iteration, best_validation_loss=np.inf)
        train_data, validation_data = self._prepare_and_split_data(encoded_data)

        logging.info("ReceptorCNN: starting training.")
        while iteration < self.iteration_count:
            for examples, labels, example_ids in self._get_data_batch(train_data, self.label_name):

                # Reset gradients
                optimizer.zero_grad()

                # Calculate predictions
                logit_outputs = self.CNN(examples)

                # Calculate losses
                loss = self._compute_loss(loss_function, logit_outputs, labels)

                # Perform update
                loss.backward()
                optimizer.step()

                self.CNN.rescale_weights_for_IGM()

                iteration += 1

                # Calculate scores and loss on training set and validation set
                if iteration % self.evaluate_at == 0 or iteration == self.iteration_count or iteration == 1:
                    logging.info(f"ReceptorCNN: training - iteration {iteration}.")
                    state = self._evaluate_state(state, iteration, loss_function, validation_data)

                if iteration >= self.iteration_count:
                    self.CNN.load_state_dict(state["model"])
                    break

        logging.info("ReceptorCNN: finished training.")

    def fit_by_cross_validation(self, encoded_data: EncodedData, number_of_splits: int = 5, label_name: str = None, cores_for_training: int = -1,
                                optimization_metric=None):
        logging.warning(f"{ReceptorCNN.__name__}: cross_validation is not implemented for this method. Using standard fitting instead...")
        self.fit(encoded_data=encoded_data, label_name=label_name)

    def _get_data_batch(self, encoded_data: EncodedData, label):
        batch_count = int(math.ceil(len(encoded_data.example_ids) / self.batch_size))
        for i in range(batch_count):
            start_index, end_index = int(self.batch_size * i), int(self.batch_size * (i + 1))
            yield encoded_data.examples[start_index: end_index], encoded_data.labels[label][start_index: end_index], \
                  encoded_data.example_ids[start_index: end_index]

    def _prepare_and_split_data(self, encoded_data: EncodedData):
        indices = list(range(len(encoded_data.example_ids)))
        random.shuffle(indices)

        limit = int(len(encoded_data.example_ids) * self.training_percentage)
        train_indices = indices[:limit]
        val_indices = indices[limit:]

        train_data = self._make_encoded_data(encoded_data, train_indices)
        val_data = self._make_encoded_data(encoded_data, val_indices)

        return train_data, val_data

    def _make_encoded_data(self, encoded_data, indices):
        examples = np.swapaxes(encoded_data.examples, 2, 3)
        return EncodedData(examples=torch.from_numpy(examples[indices]).float(),
                           labels={
                               label: torch.from_numpy(np.array([encoded_data.labels[label][i] for i in indices]) == self.class_mapping[1]).float()
                               for label in encoded_data.labels.keys()},
                           example_ids=[encoded_data.example_ids[i] for i in indices], feature_names=encoded_data.feature_names,
                           feature_annotations=encoded_data.feature_annotations, encoding=encoded_data.encoding)

    def _compute_loss(self, loss_function, logit_outputs, labels):
        pred_loss = loss_function(logit_outputs, labels)
        l1reg_loss = (torch.mean(torch.stack([p.abs().float().mean() for p in self.CNN.parameters()])))
        loss = pred_loss + l1reg_loss * self.l1_weight_decay
        return loss

    def _evaluate_state(self, state, iteration, loss_function, validation_data):
        loss = self._evaluate(loss_function, validation_data)
        logging.info(f"ReceptorCNN: current validation loss: {loss}")

        if loss < state["best_validation_loss"]:
            del state["model"]  # remove old model
            state["model"] = copy.deepcopy(self.CNN).state_dict()  # save new model to RAM
            state["iteration"] = iteration
            state["best_validation_loss"] = loss
            logging.info(f"ReceptorCNN: new best validation loss: {loss}")

        return state

    def _evaluate(self, loss_function, data: EncodedData):
        with torch.no_grad():
            self.CNN.to(device=self.device)
            loss_func = loss_function.to(device=self.device)
            loss = 0.

            with torch.no_grad():
                for examples, labels, example_ids in self._get_data_batch(data, self.label_name):
                    logit_outputs = self.CNN(examples)
                    loss += loss_func(logit_outputs, labels) / len(data.example_ids)

        return loss

    def store(self, path:Path, feature_names=None, details_path:Path=None):
        PathBuilder.build(path)

        torch.save(copy.deepcopy(self.CNN).state_dict(), str(path / "CNN.pt"))

        custom_vars = copy.deepcopy(vars(self))
        del custom_vars["CNN"]
        del custom_vars["result_path"]

        custom_vars["background_probabilities"] = custom_vars["background_probabilities"].tolist()
        custom_vars["kernel_size"] = list(custom_vars["kernel_size"])
        custom_vars["sequence_type"] = custom_vars["sequence_type"].name.lower()

        params_path = path / "custom_params.yaml"
        with params_path.open('w') as file:
            yaml.dump(custom_vars, file)

    def load(self, path):
        params_path = path / "custom_params.yaml"
        with params_path.open("r") as file:
            custom_params = yaml.load(file, Loader=yaml.SafeLoader)

        for param, value in custom_params.items():
            if hasattr(self, param):
                setattr(self, param, value)

        self.background_probabilities = np.array(self.background_probabilities)
        self.sequence_type = SequenceType[self.sequence_type.upper()]

        self._make_CNN()
        self.CNN.load_state_dict(torch.load(str(path / "CNN.pt")))

    def _make_CNN(self):
        self.CNN = RCNN(kernel_count=self.kernel_count, kernel_size=self.kernel_size, positional_channels=self.positional_channels,
                        sequence_type=self.sequence_type, background_probabilities=self.background_probabilities, chain_names=self.chain_names)

    def get_model(self, label_names: list = None):
        return vars(self)

    def check_if_exists(self, path):
        return self.CNN is not None

    def get_classes_for_label(self, label):
        if self.label_name == label:
            return np.array(list(self.class_mapping.values()))

    def get_params(self, label):
        params = copy.deepcopy(vars(self))
        params["CNN"] = copy.deepcopy(self.CNN).state_dict()
        return params

    def get_label(self):
        return [self.label_name]

    def get_package_info(self) -> str:
        return Util.get_immuneML_version()

    def get_feature_names(self) -> list:
        return self.feature_names

    def can_predict_proba(self) -> bool:
        return True
