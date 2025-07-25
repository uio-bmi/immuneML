import pickle
from pathlib import Path

import numpy as np
from glmnet import LogitNet

from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.bnp_util import write_yaml
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.ml_methods.util.Util import Util
from immuneML.util.PathBuilder import PathBuilder


class LogRegressionCustomPenalty(MLMethod):
    """
    Logistic Regression with custom penalty factors for specific features.

    **Specification arguments**:

    - penalty (str): Type of penalty to use, either 'l1' for Lasso or 'l2' for Ridge.

    - random_state (int): Random seed for reproducibility.

    - non_penalized_features (list): List of feature names that should not be penalized.

    """
    def __init__(self, penalty: str = 'l1', random_state: int = None, non_penalized_features: list = None,
                 name: str = None, label: Label = None):
        super().__init__(name=name, label=label)
        self.penalty = penalty
        self.random_state = random_state
        self.non_penalized_features = non_penalized_features if non_penalized_features is not None else []
        self.model = None
        self.feature_names = None

    def _fit(self, encoded_data: EncodedData, cores_for_training: int = 2):
        X = encoded_data.examples
        y = encoded_data.labels[self.label.name]

        self.feature_names = encoded_data.feature_names

        # Create penalty factor vector
        penalty_factor = np.ones(X.shape[1])
        for idx, feature in enumerate(self.feature_names):
            if feature in self.non_penalized_features:
                penalty_factor[idx] = 0.0

        alpha = 1 if self.penalty == 'l1' else 0.0

        self.model = LogitNet(
            alpha=alpha,
            lambda_path=None,
            n_lambda=100,
            standardize=False,  # already standardized in the encoder
            random_state=self.random_state,
            n_jobs=cores_for_training
        )
        self.model.fit(X, y, relative_penalties=penalty_factor)

    def _predict(self, encoded_data: EncodedData):
        return {self.label.name: self.model.predict(encoded_data.examples)}

    def _predict_proba(self, encoded_data: EncodedData):
        class_names = Util.map_to_old_class_values(self.model.classes_, self.class_mapping)
        probabilities = self.model.predict_proba(encoded_data.examples)
        return {self.label.name: {class_name: probabilities[:, i] for i, class_name in enumerate(class_names)}}

    def store(self, path: Path):
        PathBuilder.build(path)
        write_yaml(path / 'model.yaml', vars(self))
        with open(path / 'model.pkl', 'wb') as f:
            pickle.dump({
                'model': self.model,
                'penalty': self.penalty,
                'random_state': self.random_state,
                'non_penalized_features': self.non_penalized_features,
                'feature_names': self.feature_names
            }, f)

    def load(self, path: Path):
        with open(path / 'model.pkl', 'rb') as f:
            model = pickle.load(f)
            self.model = model['model']
            self.penalty = model['penalty']
            self.C = model['C']
            self.random_state = model['random_state']
            self.non_penalized_features = model['non_penalized_features']
            self.feature_names = model['feature_names']

    def get_params(self, for_refitting=False) -> dict:
        return {
            'penalty': self.penalty,
            'C': self.C,
            'random_state': self.random_state,
            'non_penalized_features': self.non_penalized_features
        }

    def can_predict_proba(self) -> bool:
        return True

    def can_fit_with_example_weights(self) -> bool:
        return False  # LogitNet does not support sample weights

    def get_compatible_encoders(self):
        from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
        return [KmerFrequencyEncoder]
