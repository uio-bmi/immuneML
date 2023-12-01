import copy
import logging
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.ml_methods.pytorch_implementations.PyTorchLogisticRegression import PyTorchLogisticRegression
from immuneML.ml_methods.util.Util import Util
from immuneML.util.PathBuilder import PathBuilder


class AtchleyKmerMILClassifier(MLMethod):
    """
    A binary Repertoire classifier which uses the data encoded by :ref:`AtchleyKmer` encoder to predict the repertoire label.

    The original publication:
    Ostmeyer J, Christley S, Toby IT, Cowell LG. Biophysicochemical motifs in T cell receptor sequences distinguish repertoires from tumor-infiltrating
    lymphocytes and adjacent healthy tissue. Cancer Res. Published online January 1, 2019:canres.2292.2018. `doi:10.1158/0008-5472.CAN-18-2292
    <https://cancerres.aacrjournals.org/content/79/7/1671>`_ .

    Specification arguments:

    - iteration_count (int): max number of training iterations

    - threshold (float): loss threshold at which to stop training if reached

    - evaluate_at (int): log model performance every 'evaluate_at' iterations and store the model every 'evaluate_at' iterations if early stopping
      is used

    - use_early_stopping (bool): whether to use early stopping

    - learning_rate (float): learning rate for stochastic gradient descent

    - random_seed (int): random seed used

    - zero_abundance_weight_init (bool): whether to use 0 as initial weight for abundance  term (if not, a random value is sampled from normal
      distribution with mean 0 and variance 1 / total_number_of_features

    - number_of_threads: number of threads to be used for training

    - initialization_count (int): how many times to repeat the fitting procedure from the beginning before choosing the optimal model (trains the model with multiple random initializations)

    - pytorch_device_name (str): The name of the pytorch device to use. This name will be passed to torch.device(pytorch_device_name).

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_kmer_mil_classifier:
            AtchleyKmerMILClassifier:
                iteration_count: 100
                evaluate_at: 15
                use_early_stopping: False
                learning_rate: 0.01
                random_seed: 100
                zero_abundance_weight_init: True
                number_of_threads: 8
                threshold: 0.00001
                initialization_count: 4

    """

    MIN_SEED_VALUE = 1
    MAX_SEED_VALUE = 100000

    def __init__(self, iteration_count: int = None, threshold: float = None, evaluate_at: int = None, use_early_stopping: bool = None,
                 random_seed: int = None, learning_rate: float = None, zero_abundance_weight_init: bool = None, number_of_threads: int = None,
                 result_path: Path = None, initialization_count: int = None, pytorch_device_name: str = None):
        super().__init__()
        self.logistic_regression = None
        self.random_seed = random_seed
        self.iteration_count = iteration_count
        self.threshold = threshold
        self.evaluate_at = evaluate_at
        self.use_early_stopping = use_early_stopping
        self.learning_rate = learning_rate
        self.zero_abundance_weight_init = zero_abundance_weight_init
        self.number_of_threads = number_of_threads
        self.class_mapping = None
        self.input_size = 0
        self.result_path = result_path
        self.feature_names = None
        self.initialization_count = initialization_count
        self.pytorch_device_name = pytorch_device_name

    def _make_log_reg(self):
        return PyTorchLogisticRegression(in_features=self.input_size, zero_abundance_weight_init=self.zero_abundance_weight_init)

    def fit(self, encoded_data: EncodedData, label: Label, optimization_metric=None, cores_for_training: int = 2):
        if encoded_data.example_weights is not None:
            warnings.warn(f"{self.__class__.__name__}: cannot fit this classifier with example weights, fitting without example weights instead... Example weights will still be applied when computing evaluation metrics after fitting.")

        self.feature_names = encoded_data.feature_names

        self.label = label
        self.class_mapping = Util.make_binary_class_mapping(encoded_data.labels[self.label.name], self.label.positive_class)

        mapped_y = Util.map_to_new_class_values(encoded_data.labels[self.label.name], self.class_mapping)
        self.logistic_regression = None
        min_loss = np.inf

        for initialization in range(self.initialization_count):

            random.seed(self.random_seed)
            random_seed = random.randint(AtchleyKmerMILClassifier.MIN_SEED_VALUE, AtchleyKmerMILClassifier.MAX_SEED_VALUE)

            Util.setup_pytorch(self.number_of_threads, random_seed, self.pytorch_device_name)
            self.input_size = encoded_data.examples.shape[1]

            log_reg = self._make_log_reg()
            loss = np.inf

            state = {"loss": loss, "model": None}
            loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean')
            optimizer = torch.optim.SGD(log_reg.parameters(), lr=self.learning_rate)

            for iteration in range(self.iteration_count):

                # reset gradients
                optimizer.zero_grad()

                # compute predictions only for k-mers with max score
                max_logit_indices = self._get_max_logits_indices(encoded_data.examples, log_reg)
                example_count = encoded_data.examples.shape[0]
                examples = torch.from_numpy(encoded_data.examples).float()[torch.arange(example_count).long(), :, max_logit_indices]
                logits = log_reg(examples)

                # compute the loss
                loss = loss_func(logits, torch.tensor(mapped_y).float())

                # perform update
                loss.backward()
                optimizer.step()

                # log current score and keep model for early stopping if specified
                if iteration % self.evaluate_at == 0 or iteration == self.iteration_count - 1:
                    logging.info(f"AtchleyKmerMILClassifier: log loss at iteration {iteration + 1}/{self.iteration_count}: {loss}.")
                    if state["loss"] < loss and self.use_early_stopping:
                        state = {"loss": loss.numpy(), "model": copy.deepcopy(log_reg)}

                if loss < self.threshold:
                    break

                logging.warning(f"AtchleyKmerMILClassifier: the logistic regression model did not converge.")

            if loss > state['loss'] and self.use_early_stopping:
                log_reg.load_state_dict(state["model"])

            if min_loss > loss:
                self.logistic_regression = log_reg
                min_loss = loss

    def _get_max_logits_indices(self, data, log_reg=None):
        with torch.no_grad():
            if log_reg:
                logits = log_reg(torch.from_numpy(np.swapaxes(data, 1, 2).reshape(data.shape[0] * data.shape[2], -1)))
            else:
                logits = self.logistic_regression(torch.from_numpy(np.swapaxes(data, 1, 2).reshape(data.shape[0] * data.shape[2], -1)))
        logits = torch.reshape(logits, (data.shape[0], data.shape[2]))
        max_logits_indices = torch.argmax(logits, dim=1)
        return max_logits_indices.long()

    def predict(self, encoded_data: EncodedData, label: Label):
        predictions_proba = self.predict_proba(encoded_data, label)
        return {label.name: [self.class_mapping[val] for val in (predictions_proba[label.name][label.positive_class] > 0.5).tolist()]}

    def fit_by_cross_validation(self, encoded_data: EncodedData, label: Label = None, optimization_metric: str = None,
                                number_of_splits: int = 5, cores_for_training: int = -1):
        logging.warning(f"AtchleyKmerMILClassifier: fitting by cross validation is not implemented internally for the model, fitting without "
                        f"cross-validation instead.")
        self.fit(encoded_data=encoded_data, label=label)

    def store(self, path: Path, feature_names=None, details_path: Path = None):
        PathBuilder.build(path)
        torch.save(copy.deepcopy(self.logistic_regression).state_dict(), str(path / "log_reg.pt"))
        custom_vars = copy.deepcopy(vars(self))

        coefficients_df = pd.DataFrame(custom_vars["logistic_regression"].linear.weight.detach().numpy(), columns=feature_names)
        coefficients_df["bias"] = custom_vars["logistic_regression"].linear.bias.detach().numpy()
        coefficients_df.to_csv(path / "coefficients.csv", index=False)

        del custom_vars["result_path"]
        del custom_vars["logistic_regression"]
        del custom_vars["label"]

        if self.label:
            custom_vars["label"] = self.label.get_desc_for_storage()

        params_path = path / "custom_params.yaml"
        with params_path.open('w') as file:
            yaml.dump(custom_vars, file)

    def load(self, path):
        params_path = path / "custom_params.yaml"
        with params_path.open("r") as file:
            custom_params = yaml.load(file, Loader=yaml.SafeLoader)

        for param, value in custom_params.items():
            if hasattr(self, param):
                if param == "label":
                    setattr(self, "label", Label(**value))
                else:
                    setattr(self, param, value)

        self.logistic_regression = self._make_log_reg()
        self.logistic_regression.load_state_dict(torch.load(str(path / "log_reg.pt")))

    def check_if_exists(self, path) -> bool:
        return self.logistic_regression is not None

    def get_params(self):
        params = copy.deepcopy(vars(self))
        params["logistic_regression"] = copy.deepcopy(self.logistic_regression).state_dict()
        return params

    def predict_proba(self, encoded_data: EncodedData, label: Label):
        self.logistic_regression.eval()
        example_count = encoded_data.examples.shape[0]
        max_logit_indices = self._get_max_logits_indices(encoded_data.examples)
        with torch.no_grad():
            data = torch.from_numpy(encoded_data.examples).float()[torch.arange(example_count).long(), :, max_logit_indices]
            predictions = torch.sigmoid(self.logistic_regression(data)).numpy()

        return {label.name: {label.positive_class: predictions,
                             label.get_binary_negative_class(): 1 - predictions}}

    def get_label_name(self):
        return self.label.name

    def get_package_info(self) -> str:
        return Util.get_immuneML_version()

    def get_feature_names(self) -> list:
        return self.feature_names

    def can_predict_proba(self) -> bool:
        return True

    def get_class_mapping(self) -> dict:
        return self.class_mapping

    def get_compatible_encoders(self):
        from immuneML.encodings.atchley_kmer_encoding.AtchleyKmerEncoder import AtchleyKmerEncoder
        return [AtchleyKmerEncoder]
