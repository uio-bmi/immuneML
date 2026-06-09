import copy
import logging
from pathlib import Path

import dill
import yaml

from immuneML.data_model.EncodedData import EncodedData
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.ml_methods.util.Util import Util
from immuneML.util.FilenameHandler import FilenameHandler
from immuneML.util.PathBuilder import PathBuilder


class XGBClassifier(MLMethod):
    """
    Wrapper around `XGBoost's XGBClassifier <https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier>`_.
    All XGBoost constructor parameters can be passed directly and are forwarded to the underlying model unchanged.
    If no parameters are provided, XGBoost's own defaults are used.

    Note: if you are interested in visualising feature contributions of this model,
    consider running the :ref:`SHAPReport` or :ref:`Coefficients` reports.

    **Specification arguments:**

    Any parameter accepted by `xgboost.XGBClassifier <https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier>`_
    can be specified directly (e.g. ``n_estimators``, ``max_depth``, ``learning_rate``).


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_xgb:
                    XGBClassifier:
                        n_estimators: 100
                        max_depth: 6
                        learning_rate: 0.1
                        random_state: 42
                # alternative: use all XGBoost defaults
                my_default_xgb: XGBClassifier

    """

    def __init__(self, name: str = None, label: Label = None, **kwargs):
        super().__init__(name=name, label=label)
        self._parameters = kwargs
        self.model = None

    def _fit(self, encoded_data: EncodedData, cores_for_training: int = 2):
        from xgboost import XGBClassifier as _XGB

        X = encoded_data.examples
        y = Util.map_to_new_class_values(encoded_data.labels[self.label.name], self.class_mapping)

        params = copy.deepcopy(self._parameters)
        params["n_jobs"] = cores_for_training

        self.model = _XGB(**params)

        if encoded_data.example_weights is not None:
            self.model.fit(X, y, sample_weight=encoded_data.example_weights)
        else:
            self.model.fit(X, y)

    def _predict(self, encoded_data: EncodedData):
        predictions = self.model.predict(encoded_data.examples)
        return {self.label.name: Util.map_to_old_class_values(predictions, self.class_mapping)}

    def _predict_proba(self, encoded_data: EncodedData):
        probabilities = self.model.predict_proba(encoded_data.examples)
        class_names = Util.map_to_old_class_values(self.model.classes_, self.class_mapping)
        return {self.label.name: {cls: probabilities[:, i] for i, cls in enumerate(class_names)}}

    def store(self, path: Path):
        PathBuilder.build(path)
        model_path = path / f"{self._filename()}.pickle"
        with model_path.open("wb") as f:
            dill.dump(self.model, f)

        params_path = path / f"{self._filename()}.yaml"
        with params_path.open("w") as f:
            desc = {
                **self.get_params(),
                "feature_names": self.get_feature_names(),
                "classes": self.model.classes_.tolist(),
                "class_mapping": self.class_mapping,
            }
            if self.label is not None:
                desc["label"] = self.label.get_desc_for_storage()
            yaml.dump(desc, f)

    def load(self, path: Path):
        model_path = path / f"{self._filename()}.pickle"
        if not model_path.is_file():
            raise FileNotFoundError(f"XGBClassifier model file not found at {model_path}.")
        with model_path.open("rb") as f:
            self.model = dill.load(f)

        params_path = path / f"{self._filename()}.yaml"
        if params_path.is_file():
            with params_path.open("r") as f:
                desc = yaml.safe_load(f)
                if "label" in desc:
                    self.label = Label(**desc["label"])
                for attr in ["feature_names", "class_mapping"]:
                    if attr in desc:
                        setattr(self, attr, desc[attr])

    def get_params(self, for_refitting=False) -> dict:
        params = copy.deepcopy(self.model.get_params())
        if not for_refitting:
            params["feature_importances"] = self.model.feature_importances_.tolist()
        return params

    def can_predict_proba(self) -> bool:
        return True

    def can_fit_with_example_weights(self) -> bool:
        return True

    def get_package_info(self) -> str:
        xgb_version = ""
        try:
            import xgboost
            xgb_version = f"; xgboost: {xgboost.__version__}"
        except Exception as e:
            logging.warning(f"Could not get xgboost version: {e}")
        return Util.get_immuneML_version() + xgb_version

    def get_compatible_encoders(self):
        from immuneML.encodings.diversity_encoding.EvennessProfileEncoder import EvennessProfileEncoder
        from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
        from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
        from immuneML.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
        from immuneML.encodings.reference_encoding.MatchedSequencesEncoder import MatchedSequencesEncoder
        from immuneML.encodings.reference_encoding.MatchedReceptorsEncoder import MatchedReceptorsEncoder
        from immuneML.encodings.reference_encoding.MatchedRegexEncoder import MatchedRegexEncoder
        from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
        from immuneML.encodings.protein_embedding.ESMCEncoder import ESMCEncoder
        from immuneML.encodings.protein_embedding.ProtT5Encoder import ProtT5Encoder
        from immuneML.encodings.protein_embedding.TCRBertEncoder import TCRBertEncoder
        from immuneML.encodings.diversity_encoding.ShannonDiversityEncoder import ShannonDiversityEncoder
        from immuneML.encodings.composite_encoding.CompositeEncoder import CompositeEncoder
        from immuneML.encodings.baseline_encoding.GeneFrequencyEncoder import GeneFrequencyEncoder
        from immuneML.encodings.baseline_encoding.MetadataEncoder import MetadataEncoder
        from immuneML.encodings.amino_acid_property_encoding.AminoAcidPropertyEncoder import AminoAcidPropertyEncoder
        from immuneML.encodings.sequence_length_encoding.SequenceLengthEncoder import SequenceLengthEncoder
        from immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder import \
            CompAIRRSequenceAbundanceEncoder

        return [KmerFrequencyEncoder, OneHotEncoder, Word2VecEncoder, EvennessProfileEncoder,
                MatchedSequencesEncoder, MatchedReceptorsEncoder, MatchedRegexEncoder, MotifEncoder,
                ESMCEncoder, ProtT5Encoder, TCRBertEncoder, ShannonDiversityEncoder, CompAIRRSequenceAbundanceEncoder,
                CompositeEncoder, GeneFrequencyEncoder, MetadataEncoder, AminoAcidPropertyEncoder,
                SequenceLengthEncoder]

    def _filename(self) -> str:
        return FilenameHandler.get_filename(self.__class__.__name__, "")
