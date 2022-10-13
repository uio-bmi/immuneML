import pandas as pd

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.example_weighting.ExampleWeightingParams import ExampleWeightingParams
from immuneML.example_weighting.ExampleWeightingStrategy import ExampleWeightingStrategy
from immuneML.example_weighting.importance_weighting.ImportanceWeightHelper import ImportanceWeightHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class ImportanceWeighting(ExampleWeightingStrategy):
    '''


    pseudocount_value: only if one of distributions is mutagenesis

    mutagenesis -> assumed positonal independence, like in Mason dataset

    '''

    VALID_DISTRIBUTIONS = ("uniform", "mutagenesis", "olga")

    def __init__(self, baseline_dist, dataset_dist, pseudocount_value, lower_weight_limit, upper_weight_limit, export_weights, name: str = None):
        super().__init__(name)
        self.baseline_dist = baseline_dist
        self.dataset_dist = dataset_dist
        self.pseudocount_value = pseudocount_value
        self.lower_weight_limit = lower_weight_limit
        self.upper_weight_limit = upper_weight_limit
        self.export_weights = export_weights

        self._compute_baseline_probability = self.get_probability_fn(baseline_dist)
        self._compute_dataset_probability = self.get_probability_fn(dataset_dist)

        self.alphabet_size = None
        self.dataset_positional_frequences = None


    @staticmethod
    def _prepare_parameters(baseline_dist, dataset_dist, pseudocount_value, lower_weight_limit, upper_weight_limit, export_weights, name: str = None):
        location = ImportanceWeighting.__name__

        ParameterValidator.assert_in_valid_list(baseline_dist.lower(), valid_values=ImportanceWeighting.VALID_DISTRIBUTIONS,
                                                location=location, parameter_name="baseline_dist")
        ParameterValidator.assert_in_valid_list(dataset_dist.lower(), valid_values=ImportanceWeighting.VALID_DISTRIBUTIONS,
                                                location=location, parameter_name="dataset_dist")

        assert baseline_dist != dataset_dist, f"{location}: baseline_dist cannot be the same as dataset_dist, found: {baseline_dist}"

        ParameterValidator.assert_type_and_value(pseudocount_value, (int, float), location, "pseudocount_value", min_exclusive=0)

        if lower_weight_limit is not None:
            ParameterValidator.assert_type_and_value(lower_weight_limit, (int, float), location, "lower_weight_limit", min_exclusive=0)

        if upper_weight_limit is not None:
            ParameterValidator.assert_type_and_value(upper_weight_limit, (int, float), location, "upper_weight_limit", min_exclusive=0)

        ParameterValidator.assert_type_and_value(export_weights, bool, location, "export_weights")

        return {
            "baseline_dist": baseline_dist.lower(),
            "dataset_dist": dataset_dist.lower(),
            "pseudocount_value" : pseudocount_value,
            "lower_weight_limit": lower_weight_limit,
            "upper_weight_limit": upper_weight_limit,
            "export_weights": export_weights,
            "name": name
        }

    @staticmethod
    def build_object(dataset=None, **params):
        if not isinstance(dataset, SequenceDataset):
            raise ValueError(
                f"{ImportanceWeighting.__name__}: this weighting strategy is not defined for dataset of type {dataset.__class__.__name__}. "
                f"This weighting strategy can only be used for SequenceDatasets.")

        prepared_params = ImportanceWeighting._prepare_parameters(**params)

        return ImportanceWeighting(**prepared_params)

    def compute_weights(self, dataset: SequenceDataset, params: ExampleWeightingParams):
        if params.learn_model:
            self._prepare_weighting_parameters(dataset)

        weights = self.get_weights_for_each_sequence(dataset)

        if self.export_weights:
            export_file = params.result_path / f"dataset_{dataset.identifier}_example_weights.tsv"

            if not export_file.is_file():
                PathBuilder.build(params.result_path)
                self.export_weights_to_file(dataset.get_example_ids(), weights, export_file)

        return weights

    def _prepare_weighting_parameters(self, dataset: SequenceDataset):
        if "mutagenesis" in [self.baseline_dist, self.dataset_dist]:
            self.dataset_positional_frequences = ImportanceWeightHelper.compute_positional_aa_frequences(dataset, pseudocount_value=self.pseudocount_value)

        if "uniform" in [self.baseline_dist, self.dataset_dist]:
            self.alphabet_size = len(EnvironmentSettings.get_sequence_alphabet())

    def get_weights_for_each_sequence(self, dataset: SequenceDataset):
        return CacheHandler.memo_by_params(self._prepare_caching_params(dataset),
                                           lambda: self._compute_weight_for_each_sequence(dataset))

    def _prepare_caching_params(self, dataset: SequenceDataset):
        return ("compute_weight_for_each_sequence",
                ("dataset_identifier", dataset.identifier),
                ("baseline_dist", self.baseline_dist),
                ("dataset_dist", self.dataset_dist),
                ("pseudocount_value", self.pseudocount_value),
                ("lower_weight_limit", self.lower_weight_limit),
                ("upper_weight_limit", self.upper_weight_limit),
                ("example_ids", tuple(dataset.get_example_ids())),
                ("dataset_positional_frequences", tuple(self.dataset_positional_frequences.items()) if self.dataset_positional_frequences is not None else None),
                ("alphabet_size", self.alphabet_size))

    def _compute_weight_for_each_sequence(self, dataset: SequenceDataset):
        return [self._compute_sequence_weight(sequence) for sequence in dataset.get_data()]

    def export_weights_to_file(self, identifiers, weights, result_path):
        df = pd.DataFrame({"identifier": identifiers,
                           "example_weight": weights})

        df.to_csv(result_path, index=False, header=True, sep="\t")


    def _compute_sequence_weight(self, sequence: ReceptorSequence):
        sequence_weight = self._compute_baseline_probability(sequence) / self._compute_dataset_probability(sequence)
        return self._apply_weight_thresholds(sequence_weight)

    def _apply_weight_thresholds(self, sequence_weight):
        if self.lower_weight_limit is not None:
            sequence_weight = max(self.lower_weight_limit, sequence_weight)

        if self.upper_weight_limit is not None:
            sequence_weight = min(self.upper_weight_limit, sequence_weight)

        return sequence_weight

    def get_probability_fn(self, distribution):
        if distribution == "uniform":
            return self._uniform_probability
        elif distribution == "mutagenesis":
            return self._mutagenesis_probability
        elif distribution == "olga":
            return self._olga_probability

    def _uniform_probability(self, sequence: ReceptorSequence):
        return ImportanceWeightHelper.compute_uniform_probability(sequence.get_sequence(), self.alphabet_size)

    def _mutagenesis_probability(self, sequence: ReceptorSequence):
        return ImportanceWeightHelper.compute_mutagenesis_probability(sequence.get_sequence(),
                                                                      self.dataset_positional_frequences)

    def _olga_probability(self, sequence: ReceptorSequence):
        raise NotImplementedError
