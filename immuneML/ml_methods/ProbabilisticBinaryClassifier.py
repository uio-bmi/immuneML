import copy
import pickle
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml
from scipy.special import beta as beta_func
from scipy.special import betaln as beta_func_ln
from scipy.special import digamma
from scipy.stats import betabinom as beta_binomial

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_methods.util.Util import Util
from immuneML.util.FilenameHandler import FilenameHandler
from immuneML.util.PathBuilder import PathBuilder


class ProbabilisticBinaryClassifier(MLMethod):
    """
    ProbabilisticBinaryClassifier predicts the class assignment in binary classification case based on encoding examples by number of
    successful trials and total number of trials. It models this ratio by one beta distribution per class and predicts the class of the new
    examples using log-posterior odds ratio with threshold at 0.

    ProbabilisticBinaryClassifier is based on the paper (details on the classification can be found in the Online Methods section):
    Emerson, Ryan O., William S. DeWitt, Marissa Vignali, Jenna Gravley, Joyce K. Hu, Edward J. Osborne, Cindy Desmarais, et al.
    ‘Immunosequencing Identifies Signatures of Cytomegalovirus Exposure History and HLA-Mediated Effects on the T Cell Repertoire’.
    Nature Genetics 49, no. 5 (May 2017): 659–65. `doi.org/10.1038/ng.3822 <https://doi.org/10.1038/ng.3822>`_.

    Arguments:

        max_iterations (int): maximum number of iterations while optimizing the parameters of the beta distribution (same for both classes)

        update_rate (float): how much the computed gradient should influence the updated value of the parameters of the beta distribution

        likelihood_threshold (float): at which threshold to stop the optimization (default -1e-10)

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_probabilistic_classifier: # user-defined name of the ML method
            ProbabilisticBinaryClassifier: # method name
                max_iterations: 1000
                update_rate: 0.01

    """

    SMALL_POSITIVE_NUMBER = 1e-15

    def __init__(self, max_iterations: int, update_rate: float, likelihood_threshold: float = None):
        super().__init__()
        self.max_iterations = max_iterations
        self.update_rate = update_rate
        self.N_0 = None
        self.N_1 = None
        self.alpha_0 = None
        self.alpha_1 = None
        self.beta_0 = None
        self.beta_1 = None
        self.likelihood_threshold = likelihood_threshold if likelihood_threshold is not None else -1e-10
        self.class_mapping = None
        self.label_name = None
        self.feature_names = None

    def fit(self, encoded_data: EncodedData, label_name: str, cores_for_training: int = 2):
        self.feature_names = encoded_data.feature_names
        X = encoded_data.examples
        assert X.shape[1] == 2, "ProbabilisticBinaryClassifier: the shape of the input is not compatible with the classifier. " \
                                "The classifier is defined when examples are encoded by two counts: the number of successful trials " \
                                "and the total number of trials. If this is not targeted use-case and the encoding, please consider using " \
                                "another classifier."

        self.class_mapping = Util.make_binary_class_mapping(encoded_data.labels[label_name], label_name)
        self.label_name = label_name
        self.N_0 = int(np.sum(np.array(encoded_data.labels[label_name]) == self.class_mapping[0]))
        self.N_1 = int(np.sum(np.array(encoded_data.labels[label_name]) == self.class_mapping[1]))
        self.alpha_0, self.beta_0 = self._find_beta_distribution_parameters(
            X[np.nonzero(np.array(encoded_data.labels[self.label_name]) == self.class_mapping[0])], self.N_0)
        self.alpha_1, self.beta_1 = self._find_beta_distribution_parameters(
            X[np.nonzero(np.array(encoded_data.labels[self.label_name]) == self.class_mapping[1])], self.N_1)

    def fit_by_cross_validation(self, encoded_data: EncodedData, number_of_splits: int = 5, label_name: str = None, cores_for_training: int = -1,
                                optimization_metric=None):
        warnings.warn("ProbabilisticBinaryClassifier: cross-validation on this classifier is not defined: fitting one model instead...")
        self.fit(encoded_data, label_name)

    def predict(self, encoded_data: EncodedData, label_name: str):
        """
        Predict the class assignment for examples in X (where X is validation or test set - examples not seen during training).

        .. math::

            \\widehat{c} \\, (k, n) = \\left\\{\\begin{matrix} 0, & F(k, n) \\leq 0\\\\ 1, & F(k, n) > 0 \\end{matrix}\\right

        Arguments:

            encoded_data (EncodedData): EncodedData object with examples attribute which is a design matrix of shape
            [number of examples x number of features], where number of features is 2 (the first feature is the number of disease-associated sequences
            and the second is the total number of sequences per example)

            label_name (str): name of the label used for classification (e.g. CMV)

        Returns:

            a dictionary of the following format: {label_name: predictions} where predictions is a list of predicted classes for each example

        """
        X = encoded_data.examples
        self._check_labels(label_name)
        predictions_list = []
        for example in X:
            k, n = example[0], example[1]
            F = self._compute_log_posterior_odds_ratio(k, n)
            predicted_class = int(F > 0)
            predictions_list.append(self.class_mapping[predicted_class])

        return {self.label_name: predictions_list}

    def predict_proba(self, encoded_data: EncodedData, label_name: str):
        """
        Predict the probability of the class for examples in X.

        .. math::

            \\widehat{c} \\, (k, n) = '\\left\\{\\begin{matrix} 0, & F(k, n) \\leq 0\\ 1, & F(k, n) > 0 \\end{matrix}\\right

        Arguments:

            encoded_data (EncodedData): EncodedData object with examples attribute which is a design matrix of shape, where number of features is 2
            (the first feature is the number of disease-associated sequences and the second is the total number of sequences per example)

            label_name (str): name of the label used for classification (e.g. CMV)

        Returns:

            class probabilities for all examples in X

        """
        self._check_labels(label_name)
        X = encoded_data.examples
        class_probabilities = np.zeros((X.shape[0], len(list(self.class_mapping.keys()))), dtype=float)
        for index, example in enumerate(X):
            k, n = example[0], example[1]
            posterior_class_probabilities = self._compute_posterior_class_probability(k, n)
            class_probabilities[index] = posterior_class_probabilities

        return {self.label_name: class_probabilities}

    def _find_beta_distribution_parameters(self, X, N_l: int) -> Tuple[float, float]:
        """
        Function implementing gradient ascent to find parameters of the beta distribution for the given class.
        It maximizes the following log-likelihood:

        .. math::

            l_l (\\alpha, \\beta) = - N_l \\, log \\, B (\\alpha, \\beta) + \\sum_{i: c_i = l} log \\, B(k_i + \\alpha, n_i - k_i + \\beta), l = 0, 1

        Arguments:

            X: design matrix of shape [number of examples x number of features], where number of features is 2
               (the first feature is the number of disease-associated sequences and the second is the total number of sequences per example)

            N_l: number of examples in the given class

        Returns:

             estimated values of alpha and beta for the given class

        """
        k_is, n_is = X[:, 0], X[:, 1]
        alpha, beta = self._initialize_beta_distribution_parameters(k_is, n_is)
        k_is, n_is = self._perform_laplace_smoothing(k_is, n_is)

        for iteration in range(self.max_iterations):

            log_likelihood = - N_l * beta_func(alpha, beta) + np.sum(beta_func_ln(k_is + alpha, n_is - k_is + beta))

            if np.isnan(log_likelihood):
                raise RuntimeError(f"ProbabilisticBinaryClassifier: while estimating beta distribution parameters, "
                                   f"log_likelihood became nan in iteration {iteration}. \nalpha: {alpha}, beta: {beta}")
            elif log_likelihood > self.likelihood_threshold:
                break

            grad_alpha, grad_beta = self._compute_alpha_beta_gradients(N_l, alpha, beta, k_is, n_is)

            alpha = max(alpha + self.update_rate * grad_alpha, ProbabilisticBinaryClassifier.SMALL_POSITIVE_NUMBER)
            beta = max(beta + self.update_rate * grad_beta, ProbabilisticBinaryClassifier.SMALL_POSITIVE_NUMBER)

        return alpha, beta

    def _initialize_beta_distribution_parameters(self, k_is, n_is) -> Tuple[float, float]:
        """
        Function using the method of moments to initialize the parameters of the beta distribution
        (estimating initial values for population from sample values) if variance is not 0,
        otherwise initializes both alpha and beta to 1 making all values in the domain of the distribution to have
        equal density.

        Initial parameter values as per the method of moments:

        .. math::

            \\alpha = \\frac{E[X]^2 * (1-E[X])}{V[X]}-E[X]
            \\beta = (\\frac{E[X](1-E[X])}{V[X]} - 1) * (1 - E[X])

        Arguments:

            k_is: number of disease-associated sequences per example

            n_is: total number of sequences per example

        Returns:

            initial values of parameters alpha and beta

        """
        binomial_proportions_p = k_is / n_is
        mean = binomial_proportions_p.mean()
        variance = binomial_proportions_p.var()
        if variance != 0:
            alpha = np.square(mean) * (1 - mean) / variance - mean
            beta = (mean * (1 - mean) / variance - 1) * (1 - mean)
        else:
            alpha, beta = 1, 1
        return alpha, beta

    def _compute_alpha_beta_gradients(self, N_l, alpha, beta, k_is, n_is) -> Tuple[float, float]:
        """
        Function computing the gradients of alpha and beta parameters of the beta distribution to maximize log-likelihood:

        .. math::

            \\frac{\\partial  l_l}{\\partial \\alpha} = - N_l (\\Psi (\\alpha) - \\Psi (\\alpha + \\beta)) + \\sum_{i:c_i=l}^{} (\\Psi(k_i + \\alpha) - \\Psi(n_i + k_i + \\alpha + \\beta))
            \\frac{\\partial  l_l}{\\partial \\beta} = - N_l (\\Psi(\\beta) - \\Psi(\\alpha + \\beta)) + \\sum_{i:c_i=l} (\\Psi(n_i - k_i + \\beta) - \\Psi(n_i + k_i + \\alpha + \\beta))

        Arguments:

            N_l: number of examples in the current class

            alpha: alpha parameter of beta distribution

            beta: beta parameter of beta distribution

            k_is: array of numbers of disease-associated sequences per training example

            n_is: array of total numbers of sequences per training example

        Returns:

            gradients for alpha and beta

        """
        grad_alpha = - N_l * (digamma(alpha) - digamma(alpha + beta)) \
                     + np.sum([digamma(k_is[i] + alpha) - digamma(n_is[i] + k_is[i] + alpha + beta)
                               for i in range(k_is.shape[0])])

        grad_beta = - N_l * (digamma(beta) - digamma(alpha + beta)) \
                    + np.sum([digamma(n_is[i] - k_is[i] + beta) - digamma(n_is[i] + k_is[i] + alpha + beta)
                              for i in range(k_is.shape[0])])
        return grad_alpha, grad_beta

    def _perform_laplace_smoothing(self, k_is, n_is) -> Tuple[np.array, np.array]:
        """
        Function performing Laplace smoothing of data, where it uses the most deeply sampled example in the class (example with maximum n)
        by adding the ratio of number of disease-associated sequences and total number of sequences for the example with maximum n to the
        number of disease-associated sequences for all examples, and 1 to the total number of sequences for all examples,
        thus regularizing the likelihood computed from these values and potentially avoiding numerical instabilities.

        If n_max is the total number of sequences in the example with the largest number of total sequences, and k_max is the number of
        disease-associated sequences for that same example, then the smoothing is performed in the following way for each example in the
        training dataset:

        .. math::

            k_i' = k_i + k_{max} / n_{max}
            n_i' = n_i + 1

        Arguments:

            k_is: array of numbers of disease-associated sequences per training example
            n_is: array of total numbers of sequences per training example

        Returns:

            Laplace-smoothed values of k_i and n_i

        """
        regularizer_index = np.argmax(n_is)  # index of max n
        regularizer_k = k_is[regularizer_index]  # k corresponding to max n
        regularizer_n = n_is[regularizer_index]  # max n

        regularized_k_is = copy.deepcopy(k_is)
        regularized_n_is = copy.deepcopy(n_is)
        regularized_k_is = regularized_k_is + regularizer_k / regularizer_n
        regularized_n_is = regularized_n_is + 1

        return regularized_k_is, regularized_n_is

    def _compute_posterior_class_probability(self, k, n) -> Tuple[float, float]:
        """
        For given parameters of beta distributions for both classes, computes the posterior class probabilities:

        .. math::

            p(c' = x | n', k')= \\binom{n'}{k'} \\frac{B(k'+\\alpha_x, n' - k' + \\beta_x)}{B(\\alpha_x, \\beta_x)} \\frac{N_x + 1}{N + 2}, x=0,1

        Arguments:

            k: number of disease-associated sequences
            n: total number of sequences

        Returns:

            a tuple of probabilities for negative class and positive class for given example, normalized to sum to 1

        """
        predicted_probability_0 = beta_binomial.pmf(k, n, self.alpha_0, self.beta_0) * (self.N_0 + 1) / (self.N_0 + self.N_1 + 2)
        predicted_probability_1 = beta_binomial.pmf(k, n, self.alpha_1, self.beta_1) * (self.N_1 + 1) / (self.N_0 + self.N_1 + 2)

        normalization_const = predicted_probability_0 + predicted_probability_1

        if np.isnan(normalization_const):
            raise ValueError(f"{ProbabilisticBinaryClassifier.__name__}: encountered nan in predicted posterior class probabilities."
                             f"\nprobability of class 0: {predicted_probability_0}\nprobability of class 1: {predicted_probability_1}\n"
                             f"alpha 0: {self.alpha_0}, beta 0: {self.beta_0}\nalpha 1: {self.alpha_1}, beta 1: {self.beta_1}\n"
                             f"positive example count: {self.N_1}, negative example count: {self.N_0}")
        elif normalization_const == 0:
            warnings.warn(f"{ProbabilisticBinaryClassifier.__name__}: posterior class probabilities for both classes are 0 (k={k}, n={n}). Returning "
                          f"normalized values to indicate that the example could not be classified, by setting both probabilities to 0.5.", RuntimeWarning)
            return 0.5, 0.5

        return predicted_probability_0 / normalization_const, predicted_probability_1 / normalization_const

    def _compute_log_posterior_odds_ratio(self, k, n):
        """
        Function computing log-posterior odds ratio for class assignment for new example with parameters k and n:

        .. math::

            F(k, n) = log \\, p (c=1|k,n) - log \\, p(c=0|k,n)) = log (N_1 + 1) - log(N_0 + 1) + log \\, B(\\alpha_0, \\beta_0) -  log \\, B(\\alpha_1, \\beta_1) +  log \\, B(k + \\alpha_1, n - k + \\beta_1) -  log \\, B(k + \\alpha_0, n-k + \\beta_0)

        Arguments:

            k: number of disease-associated sequences
            n: total number of sequences

        Returns:

            log-posterior odds ratio for class assignment

        """
        return np.log(self.N_1 + 1) - np.log(self.N_0 + 1) \
               + beta_func_ln(self.alpha_0, self.beta_0) - beta_func_ln(self.alpha_1, self.beta_1) \
               + beta_func_ln(k + self.alpha_1, n - k + self.beta_1) \
               - beta_func_ln(k + self.alpha_0, n - k + self.beta_0)

    def get_classes_for_label(self, label):
        if label == self.label_name:
            return np.array(list(self.class_mapping.values()))
        else:
            warnings.warn("ProbabilisticBinaryClassifier: in get_classes_for_label() a label was passed in for which "
                          "the classifier was not trained: returning None...", RuntimeWarning)
            return None

    def _convert_object_to_dict(self):
        content = vars(self)
        result = {}
        for key, value in content.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif value is None or isinstance(value, str) or isinstance(value, dict) or isinstance(value, list) or isinstance(value, Path):
                result[key] = value
            else:
                result[key] = float(value)

        return result

    def store(self, path: Path, feature_names=None, details_path=None):
        content = self._convert_object_to_dict()
        PathBuilder.build(path)
        file_path = path / FilenameHandler.get_filename(self.__class__.__name__, "pickle")

        with file_path.open("wb") as file:
            pickle.dump(content, file)

        if details_path is None:
            params_path = path / FilenameHandler.get_filename(self.__class__.__name__, "yaml")
        else:
            params_path = details_path

        with params_path.open("w") as file:
            desc = {self.label_name: {
                **content,
                "feature_names": feature_names,
                "classes": list(self.class_mapping.values())
            }}
            yaml.dump(desc, file)

    def load(self, path: Path):
        keys = list(vars(self).keys())
        file_path = path / FilenameHandler.get_filename(self.__class__.__name__, "pickle")
        if file_path.is_file():
            with file_path.open("rb") as file:
                content = pickle.load(file)
                assert all(
                    key in keys for key in content.keys()), f"ProbabilisticBinaryClassifier: error while loading from {file_path}: " \
                                                            f"object attributes from file and from the class do not match.\n" \
                                                            f"Attributes from file: {list(content.keys())}\n" \
                                                            f"Attributes for object of class ProbabilisticBinaryClassifier: {keys}"
                for key in content:
                    setattr(self, key, content[key])
        else:
            raise FileNotFoundError(f"{self.__class__.__name__} model could not be loaded from {file_path}. "
                                    f"Check if the path to the {file_path.name} file is properly set.")

    def get_model(self, label_name: str = None):
        return vars(self)

    def get_params(self, label):
        if label == self.label_name:
            return vars(self)
        else:
            warnings.warn("ProbabilisticBinaryClassifier: in get_params() a label was passed in for which "
                          "the classifier was not trained: returning None...", RuntimeWarning)
            return None

    def check_if_exists(self, path):
        vals = vars(self).values()
        if any(val is None for val in vals):
            return False
        else:
            return True

    def _check_labels(self, label_name):
        assert label_name == self.label_name, f"ProbabilisticBinaryClassifier: classifier cannot predict the labels " \
                                              f"on which it was not trained: got: {label_name}, expected: {self.label_name}."

    def get_label(self):
        return [self.label_name]

    def get_package_info(self) -> str:
        return Util.get_immuneML_version()

    def get_feature_names(self) -> list:
        return self.feature_names

    def can_predict_proba(self) -> bool:
        return True
