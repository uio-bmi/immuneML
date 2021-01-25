import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_methods.pytorch_implementations.PyTorchLogisticRegression import PyTorchLogisticRegression
from immuneML.ml_methods.util.Util import Util
from immuneML.util.PathBuilder import PathBuilder


class AtchleyKmerMILClassifier(MLMethod):
    """
    Repertoire classifier which uses the data encoded by :ref:`AtchleyKmerEncoder` to predict the repertoire label.

    The original publication:
    Ostmeyer J, Christley S, Toby IT, Cowell LG. Biophysicochemical motifs in T cell receptor sequences distinguish repertoires from tumor-infiltrating
    lymphocytes and adjacent healthy tissue. Cancer Res. Published online January 1, 2019:canres.2292.2018. `doi:10.1158/0008-5472.CAN-18-2292
    <https://cancerres.aacrjournals.org/content/79/7/1671>`_ .

    Arguments:

        iteration_count (int): max number of training iterations

        threshold (float): loss threshold at which to stop training if reached

        evaluate_at (int): log model performance every 'evaluate_at' iterations and store the model every 'evaluate_at' iterations if early stopping
        is used

        use_early_stopping (bool): whether to use early stopping

        learning_rate (float): learning rate for stochastic gradient descent

        random_seed (int): random seed used

        zero_abundance_weight_init (bool): whether to use 0 as initial weight for abundance  term (if not, a random value is sampled from normal
        distribution with mean 0 and variance 1 / total_number_of_features

        number_of_threads: number of threads to be used for training

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

    """

    def __init__(self, iteration_count: int = None, threshold: float = None, evaluate_at: int = None, use_early_stopping: bool = None,
                 random_seed: int = None, learning_rate: float = None, zero_abundance_weight_init: bool = None, number_of_threads: int = None,
                 result_path: Path = None):
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
        self.label_name = None
        self.class_mapping = None
        self.input_size = 0
        self.result_path = result_path
        self.feature_names = None

    def _make_log_reg(self):

        self.logistic_regression = PyTorchLogisticRegression(in_features=self.input_size, zero_abundance_weight_init=self.zero_abundance_weight_init)

    def _check_encoded_data(self, encoded_data: EncodedData):
        assert encoded_data.encoding == 'AtchleyKmerEncoder', f"AtchleyKmerMILClassifier: the encoding is not compatible with the given classifier. " \
                                                              f"Expected AtchleyKmer encoding, got {encoded_data.encoding} instead. "

    def fit(self, encoded_data: EncodedData, label_name: str, cores_for_training: int = 2):
        self.feature_names = encoded_data.feature_names
        self._check_encoded_data(encoded_data)

        Util.setup_pytorch(self.number_of_threads, self.random_seed)
        self.input_size = encoded_data.examples.shape[1]

        self._make_log_reg()

        self.class_mapping = Util.make_binary_class_mapping(encoded_data.labels[label_name], label_name)
        self.label_name = label_name
        loss = np.inf

        state = {"loss": loss, "model": None}
        loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = torch.optim.SGD(self.logistic_regression.parameters(), lr=self.learning_rate)

        for iteration in range(self.iteration_count):

            # reset gradients
            optimizer.zero_grad()

            # compute predictions only for k-mers with max score
            max_logit_indices = self._get_max_logits_indices(encoded_data.examples)
            example_count = encoded_data.examples.shape[0]
            examples = torch.from_numpy(encoded_data.examples).float()[torch.arange(example_count).long(), :, max_logit_indices]
            logits = self.logistic_regression(examples)

            # compute the loss
            loss = loss_func(logits, torch.tensor(encoded_data.labels[self.label_name]).float())

            # perform update
            loss.backward()
            optimizer.step()

            # log current score and keep model for early stopping if specified
            if iteration % self.evaluate_at == 0 or iteration == self.iteration_count - 1:
                logging.info(f"AtchleyKmerMILClassifier: log loss at iteration {iteration+1}/{self.iteration_count}: {loss}.")
                if state["loss"] < loss and self.use_early_stopping:
                    state = {"loss": loss.numpy(), "model": copy.deepcopy(self.logistic_regression)}

            if loss < self.threshold:
                break

        logging.warning(f"AtchleyKmerMILClassifier: the logistic regression model did not converge.")

        if loss > state['loss'] and self.use_early_stopping:
            self.logistic_regression.load_state_dict(state["model"])

    def _get_max_logits_indices(self, data):
        with torch.no_grad():
            logits = self.logistic_regression(torch.from_numpy(np.swapaxes(data, 1, 2).reshape(data.shape[0] * data.shape[2], -1)))
        logits = torch.reshape(logits, (data.shape[0], data.shape[2]))
        max_logits_indices = torch.argmax(logits, dim=1)
        return max_logits_indices.long()

    def predict(self, encoded_data: EncodedData, label_name: str):
        predictions_proba = self.predict_proba(encoded_data, label_name)
        return {self.label_name: [self.class_mapping[val] for val in (predictions_proba[self.label_name][:, 1] > 0.5).tolist()]}

    def fit_by_cross_validation(self, encoded_data: EncodedData, number_of_splits: int = 5, label_name: str = None, cores_for_training: int = -1,
                                optimization_metric=None):
        logging.warning(f"AtchleyKmerMILClassifier: fitting by cross validation is not implemented internally for the model, fitting without "
                        f"cross-validation instead.")
        self.fit(encoded_data, label_name)

    def store(self, path: Path, feature_names=None, details_path: Path = None):
        PathBuilder.build(path)
        torch.save(copy.deepcopy(self.logistic_regression).state_dict(), str(path / "log_reg.pt"))
        custom_vars = copy.deepcopy(vars(self))

        coefficients_df = pd.DataFrame(custom_vars["logistic_regression"].linear.weight.detach().numpy(), columns=feature_names)
        coefficients_df["bias"] = custom_vars["logistic_regression"].linear.bias.detach().numpy()
        coefficients_df.to_csv(path / "coefficients.csv", index=False)

        del custom_vars["result_path"]
        del custom_vars["logistic_regression"]

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

        self._make_log_reg()
        self.logistic_regression.load_state_dict(torch.load(str(path / "log_reg.pt")))

    def get_model(self, label_name: str = None):
        return vars(self)

    def check_if_exists(self, path) -> bool:
        return self.logistic_regression is not None

    def get_classes_for_label(self, label):
        if self.label_name == label:
            return np.array(list(self.class_mapping.values()))

    def get_params(self, label):
        params = copy.deepcopy(vars(self))
        params["logistic_regression"] = copy.deepcopy(self.logistic_regression).state_dict()
        return params

    def predict_proba(self, encoded_data: EncodedData, label_name: str):
        self.logistic_regression.eval()
        example_count = encoded_data.examples.shape[0]
        max_logit_indices = self._get_max_logits_indices(encoded_data.examples)
        with torch.no_grad():
            data = torch.from_numpy(encoded_data.examples).float()[torch.arange(example_count).long(), :, max_logit_indices]
            predictions = torch.sigmoid(self.logistic_regression(data)).numpy()
        return {self.label_name: np.vstack([1 - np.array(predictions), predictions]).T}

    def get_label(self):
        return [self.label_name]

    def get_package_info(self) -> str:
        return Util.get_immuneML_version()

    def get_feature_names(self) -> list:
        return self.feature_names

    def can_predict_proba(self) -> bool:
        return True
