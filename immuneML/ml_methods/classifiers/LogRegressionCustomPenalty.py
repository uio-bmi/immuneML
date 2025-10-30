import logging
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

    - non_penalized_features (list): List of feature names that should not be penalized.

    - non_penalized_encodings (list): List of encoding names (class names) whose features should not be penalized. This
      parameter can be used only in combination with CompositeEncoder. None fo the features from the specified encodings
      will be penalized. If both non_penalized_features and non_penalized_encodings are provided, the union of the two
      will be used.

    Other supported arguments are inherited from LogitNet of python-glmnet package and will be directly passed to it.
    n_jobs will be overwritten to use the number of CPUs specified for the instruction (e.g. in TrainMLModel).

    **YAML specification:**

    .. code-block:: yaml

        ml_methods:
            custom_log_reg:
                LogRegressionCustomPenalty:
                    alpha: 1
                    n_lambda: 100
                    non_penalized_features: []
                    non_penalized_encodings: ['Metadata']
                    random_state: 42

    """
    def __init__(self, non_penalized_features: list = None, name: str = None, label: Label = None,
                 non_penalized_encodings: list = None, **kwargs):
        super().__init__(name=name, label=label)
        self.non_penalized_features = non_penalized_features if non_penalized_features is not None else []
        self.non_penalized_encodings = non_penalized_encodings if non_penalized_encodings is not None else []

        for ind, encoding in enumerate(self.non_penalized_encodings):
            if 'Encoder' not in encoding:
                self.non_penalized_encodings[ind] = encoding + 'Encoder'

        self.model = None
        self.feature_names = None
        self.kwargs = kwargs

    def _fit(self, encoded_data: EncodedData, cores_for_training: int = 2):
        X = encoded_data.examples
        y = Util.map_to_new_class_values(encoded_data.labels[self.label.name], self.class_mapping)

        self.feature_names = encoded_data.feature_names

        if encoded_data.encoding == 'CompositeEncoder' and self.non_penalized_encodings:
            features_from_non_penalized_encodings = encoded_data.feature_annotations[encoded_data.feature_annotations['encoder'].isin(self.non_penalized_encodings)]['feature'].tolist()
            non_penalized_features = list(set(features_from_non_penalized_encodings))
            non_penalized_features.extend(self.non_penalized_features)
            self.non_penalized_features = list(set(non_penalized_features))

            logging.info(f"{self.__class__.__name__}: inferred non-penalized features: {self.non_penalized_features}")

        # Create penalty factor vector
        penalty_factor = np.ones(X.shape[1])
        for idx, feature in enumerate(self.feature_names):
            if feature in self.non_penalized_features:
                penalty_factor[idx] = 0.0

        self.model = LogitNet(**self.kwargs, **{'n_jobs': cores_for_training})
        self.model.fit(X, y, relative_penalties=penalty_factor)

    def _predict(self, encoded_data: EncodedData):
        predictions = self.model.predict(encoded_data.examples)
        return {self.label.name: Util.map_to_old_class_values(np.array(predictions), self.class_mapping)}

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
                'non_penalized_features': self.non_penalized_features,
                'feature_names': self.feature_names
            }, f)

    def load(self, path: Path):
        with open(path / 'model.pkl', 'rb') as f:
            model = pickle.load(f)
            self.model = model['model']
            self.non_penalized_features = model['non_penalized_features']
            self.feature_names = model['feature_names']

    def get_params(self, for_refitting=False) -> dict:
        return {
            'penalty': self.model.penalty,
            'C': self.model.C,
            'random_state': self.model.random_state,
            'non_penalized_features': self.non_penalized_features
        }

    def can_predict_proba(self) -> bool:
        return True

    def can_fit_with_example_weights(self) -> bool:
        return False  # LogitNet does not support sample weights

    def get_compatible_encoders(self):
        from immuneML.encodings.composite_encoding.CompositeEncoder import CompositeEncoder
        return [CompositeEncoder]
