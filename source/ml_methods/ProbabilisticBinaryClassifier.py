import copy
import os
import pickle
import warnings
from typing import Tuple

import numpy as np
import yaml
from scipy.special import beta as beta_func
from scipy.special import betaln as beta_func_ln
from scipy.special import digamma, comb

from source.ml_methods.MLMethod import MLMethod
from source.util.FilenameHandler import FilenameHandler
from source.util.PathBuilder import PathBuilder


class ProbabilisticBinaryClassifier(MLMethod):
    """
    ProbabilisticBinaryClassifier predicts the class assignment in binary classification case based on encoding examples by number of
    successful trials and total number of trials. It models this ratio by one beta distribution per class and predicts the class of the new
    examples using log-posterior odds ratio with threshold at 0.

    ProbabilisticBinaryClassifier is based on the paper (details on the classification can be found in the Online Methods section):
    Emerson, Ryan O., William S. DeWitt, Marissa Vignali, Jenna Gravley, Joyce K. Hu, Edward J. Osborne, Cindy Desmarais, et al.
    ‘Immunosequencing Identifies Signatures of Cytomegalovirus Exposure History and HLA-Mediated Effects on the T Cell Repertoire’.
    Nature Genetics 49, no. 5 (May 2017): 659–65. https://doi.org/10.1038/ng.3822.
    """

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

    def fit(self, X, y, label_names: list = None, cores_for_training: int = 2):
        assert X.shape[1] == 2, "ProbabilisticBinaryClassifier: the shape of the input is not compatible with the classifier. " \
                                "The classifier is defined when examples are encoded by two counts: the number of successful trials " \
                                "and the total number of trials. If this is not targeted use-case and the encoding, please consider using " \
                                "another classifier."

        self.class_mapping = self._make_class_mapping(y, label_names)
        self.label_name = label_names[0]
        self.N_0 = np.sum(np.array(y[label_names[0]]) == self.class_mapping[0])
        self.N_1 = np.sum(np.array(y[label_names[0]]) == self.class_mapping[1])
        self.alpha_0, self.beta_0 = self._find_beta_distribution_parameters(
            X[np.nonzero(np.array(y[self.label_name]) == self.class_mapping[0])], self.N_0)
        self.alpha_1, self.beta_1 = self._find_beta_distribution_parameters(
            X[np.nonzero(np.array(y[self.label_name]) == self.class_mapping[1])], self.N_1)

    def predict(self, X, label_names: list = None):
        """
        Predict the class assignment for examples in X (where X is validation or test set - examples not seen during training).

        .. math::
            \widehat{c} \, (k, n) = \left\{\begin{matrix} 0, & F(k, n) \leq 0\\ 1, & F(k, n) > 0 \end{matrix}\right

        Arguments:
            X: design matrix of shape [number of examples x number of features], where number of features is 2
               (the first feature is the number of disease-associated sequences and the second is the total number of sequences per example)
            label_names: name of the label used for classification (e.g. CMV)

        Returns:
            class predictions for all examples in X
        """
        self._check_labels(label_names)
        predictions_list = []
        for example in X:
            k, n = example[0], example[1]
            F = self._compute_log_posterior_odds_ratio(k, n)
            predicted_class = int(F > 0)
            predictions_list.append(self.class_mapping[predicted_class])

        return {self.label_name: predictions_list}

    def _find_beta_distribution_parameters(self, X, N_l: int) -> Tuple[float, float]:
        """
        Function implementing gradient ascent to find parameters of the beta distribution for the given class.
        It maximizes the following log-likelihood:

        .. math::
            l_l (\alpha, \beta) = - N_l \, log \, B (\alpha, \beta) + \sum_{i: c_i = l} log \, B(k_i + \alpha, n_i - k_i + \beta), l = 0, 1

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

            alpha = alpha + self.update_rate * grad_alpha
            beta = beta + self.update_rate * grad_beta

        return alpha, beta

    def _initialize_beta_distribution_parameters(self, k_is, n_is) -> Tuple[float, float]:
        """
        Function using the method of moments to initialize the parameters of the beta distribution
        (estimating initial values for population from sample values) if variance is not 0,
        otherwise initializes both alpha and beta to 1 making all values in the domain of the distribution to have
        equal density.

        Initial parameter values as per the method of moments:
        .. math::
            \alpha = \frac{E[X]^2 * (1-E[X])}{V[X]}-E[X]
            \beta = (\frac{E[X](1-E[X])}{V[X]} - 1) * (1 - E[X])

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
            \frac{\partial  l_l}{\partial \alpha} = - N_l (\Psi (\alpha) - \Psi (\alpha + \beta)) + \sum_{i:c_i=l}^{} (\Psi(k_i + \alpha) - \Psi(n_i + k_i + \alpha + \beta))
            \frac{\partial  l_l}{\partial \beta} = - N_l (\Psi(\beta) - \Psi(\alpha + \beta)) + \sum_{i:c_i=l} (\Psi(n_i - k_i + \beta) - \Psi(n_i + k_i + \alpha + \beta))

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
            p(c' = x | n', k')= \binom{n'}{k'} \frac{B(k'+\alpha_x, n' - k' + \beta_x)}{B(\alpha_x, \beta_x)} \frac{N_x + 1}{N + 2}, x=0,1

        Arguments:
            k: number of disease-associated sequences
            n: total number of sequences

        Returns:
            a tuple of probabilities for negative class and positive class for given example
        """
        predicted_probability = comb(n, k) \
                                * beta_func(k + self.alpha_0, n - k + self.beta_0) / beta_func(self.alpha_0, self.beta_0) \
                                * (self.N_0 + 1) / (self.N_0 + self.N_1 + 2)
        return predicted_probability, 1 - predicted_probability

    def _compute_log_posterior_odds_ratio(self, k, n):
        """
        Function computing log-posterior odds ratio for class assignment for new example with parameters k and n:

        .. math::
            F(k, n) = log \, p (c=1|k,n) - log \, p(c=0|k,n)) = log (N_1 + 1) - log(N_0 + 1) + log \, B(\alpha_0, \beta_0) -  log \, B(\alpha_1, \beta_1) +  log \, B(k + \alpha_1, n - k + \beta_1) -  log \, B(k + \alpha_0, n-k + \beta_0)

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

    def fit_by_cross_validation(self, X, y, number_of_splits: int = 5, parameter_grid: dict = None, label_names: list = None):
        warnings.warn("ProbabilisticBinaryClassifier: cross-validation on this classifier is not defined: fitting one model instead...")
        self.fit(X, y, label_names)

    def predict_proba(self, X, labels):
        """
        Predict the probability of the class for examples in X.

        .. math::
            \widehat{c} \, (k, n) = \left\{\begin{matrix} 0, & F(k, n) \leq 0\\ 1, & F(k, n) > 0 \end{matrix}\right

        Arguments:
            X: design matrix of shape [number of examples x number of features], where number of features is 2
               (the first feature is the number of disease-associated sequences and the second is the total number of sequences per example)
            labels: name of the label used for classification (e.g. CMV)

        Returns:
            class probabilities for all examples in X
        """
        self._check_labels(labels)
        class_probabilities = np.zeros((X.shape[0], len(list(self.class_mapping.keys))), dtype=float)
        for index, example in enumerate(X):
            k, n = example[0], example[1]
            posterior_class_probabilities = self._compute_posterior_class_probability(k, n)
            class_probabilities[index] = posterior_class_probabilities

        return {self.label_name: class_probabilities}

    def get_classes_for_label(self, label):
        if label == self.label_name:
            return self.class_mapping.values()
        else:
            warnings.warn("ProbabilisticBinaryClassifier: in get_classes_for_label() a label was passed in for which "
                          "the classifier was not trained: returning None...", RuntimeWarning)
            return None

    def _make_class_mapping(self, y, label_names: list):
        assert len(label_names) == 1, "ProbabilisticBinaryClassifier: more than one label was specified at the time, but this " \
                                      "classifier can handle only binary classification for one label. Try using HPOptimization " \
                                      "instruction which will train different classifiers for all provided labels."
        unique_values = np.unique(y[label_names[0]])
        assert unique_values.shape[0] == 2, "ProbabilisticBinaryClassifier: more than two classes were given to binary classifier. " \
                                            "If there are more than two classes, choose some of the other classifiers."

        if 0 in unique_values and 1 in unique_values:
            mapping = {1: 1, 0: 0}
        elif True in unique_values and False in unique_values:
            mapping = {1: True, 0: False}
        else:
            mapping = {0: unique_values[0], 1: unique_values[1]}

        return mapping

    def store(self, path, feature_names=None, details_path=None):
        content = vars(self)
        PathBuilder.build(path)
        name = FilenameHandler.get_filename(self.__class__.__name__, "pickle")
        with open(path + name, "wb") as file:
            pickle.dump(content, file)

        if details_path is None:
            params_path = path + FilenameHandler.get_filename(self.__class__.__name__, "yaml")
        else:
            params_path = details_path

        with open(params_path, "w") as file:
            desc = {self.label_name: {
                **content,
                "feature_names": feature_names,
                "classes": list(self.class_mapping.values())
            }}
            yaml.dump(desc, file)

    def load(self, path):
        keys = list(vars(self).keys())
        name = FilenameHandler.get_filename(self.__class__.__name__, "pickle")
        if os.path.isfile(path + name):
            with open(path + name, "rb") as file:
                content = pickle.load(file)
                assert all(
                    key in keys for key in content.keys()), f"ProbabilisticBinaryClassifier: error while loading from {path + name}: " \
                                                            f"object attributes from file and from the class do not match.\n" \
                                                            f"Attributes from file: {list(content.keys())}\n" \
                                                            f"Attributes for object of class ProbabilisticBinaryClassifier: {keys}"
                for key in content:
                    setattr(self, key, content[key])
        else:
            raise FileNotFoundError(self.__class__.__name__ + " model could not be loaded from " + str(
                path + name) + ". Check if the path to the " + name + " file is properly set.")

    def get_model(self, label_names: list = None):
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

    def _check_labels(self, labels):
        assert len(labels) == 1 and labels[0] == self.label_name, f"ProbabilisticBinaryClassifier: classifier cannot predict the labels " \
                                                                  f"on which it was not trained: " \
                                                                  f"got labels: {labels}, expected: {self.label_name}."
