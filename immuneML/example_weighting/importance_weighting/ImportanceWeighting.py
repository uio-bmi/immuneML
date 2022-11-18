from functools import partial

import pandas as pd

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.example_weighting.ExampleWeightingParams import ExampleWeightingParams
from immuneML.example_weighting.ExampleWeightingStrategy import ExampleWeightingStrategy
from immuneML.example_weighting.importance_weighting.DistributionParameters import DistributionParameters
from immuneML.example_weighting.importance_weighting.ImportanceWeightHelper import ImportanceWeightHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class ImportanceWeighting(ExampleWeightingStrategy):
    '''

    pseudocount_value: only if one of distributions is mutagenesis

    mutagenesis -> assumed positonal independence, like in Mason dataset


    Arguments:

        baseline_dist

        dataset_dist

        pseudocount_value

        lower_weight_limit

        upper_weight_limit

        export_weights




    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_weighting:
            ImportanceWeighting:
                ...

    '''

    TYPE_UNIFORM = "uniform"
    TYPE_MUTAGENESIS = "mutagenesis"
    TYPE_OLGA = "olga"
    VALID_DISTRIBUTIONS = (TYPE_UNIFORM, TYPE_MUTAGENESIS, TYPE_OLGA)

    def __init__(self, baseline_dist: DistributionParameters, dataset_dist: DistributionParameters,
                 pseudocount_value: int, lower_weight_limit: float, upper_weight_limit: float,
                 export_weights: bool, name: str = None):
        super().__init__(name)
        self.baseline_dist = baseline_dist
        self.dataset_dist = dataset_dist
        self.pseudocount_value = pseudocount_value
        self.lower_weight_limit = lower_weight_limit
        self.upper_weight_limit = upper_weight_limit
        self.export_weights = export_weights

        self._compute_baseline_probability = None
        self._compute_dataset_probability = None

        self.alphabet_size = None
        self.baseline_positional_frequences = None
        self.dataset_positional_frequences = None

    @staticmethod
    def _build_distribution_parameters(dist_yaml, dataset_type):
        if type(dist_yaml) is str:
            return DistributionParameters(distribution_type=dist_yaml, dataset_type=dataset_type)
        else:
            distribution_type = list(dist_yaml.keys())[0]
            return DistributionParameters(distribution_type=distribution_type,
                                          dataset_type=dataset_type,
                                          label_name=dist_yaml[distribution_type]["restrict_to"]["label"],
                                          class_name=dist_yaml[distribution_type]["restrict_to"]["class"])


    @staticmethod
    def _prepare_parameters(baseline_dist, dataset_dist, pseudocount_value: int,
                            lower_weight_limit: float, upper_weight_limit: float, export_weights: bool, name: str = None):
        location = ImportanceWeighting.__name__

        for dist, parameter_name in zip([baseline_dist, dataset_dist], ["baseline_dist", "dataset_dist"]):
            if type(dist) is dict:
                error_mssg = f"{location}: {dist} is not a valid value for parameter {parameter_name}. " \
                             f"Expected either a single value (legal values are: {', '.join(ImportanceWeighting.VALID_DISTRIBUTIONS)}), " \
                             f"or alternatively you may specify the 'restrict_to' parameter for the distribution of type " \
                             f"{ImportanceWeighting.TYPE_MUTAGENESIS}, for example:\n\n" \
                             f"mutagenesis:\n" \
                             f"  restrict_to:\n" \
                             f"    label: CMV\n" \
                             f"    class: -\n"

                assert list(dist.keys()) == [ImportanceWeighting.TYPE_MUTAGENESIS], error_mssg
                assert dist["mutagenesis"].keys() == ["restrict_to"], error_mssg
                assert dist["mutagenesis"]["restrict_to"].keys() == ["label", "class"], error_mssg
                assert type(dist["mutagenesis"]["restrict_to"]["label"]) is str, error_mssg
                assert type(dist["mutagenesis"]["restrict_to"]["class"]) is str, error_mssg
                # todo make error messages more specific from 'restrict_to' and onward

            else:
                ParameterValidator.assert_in_valid_list(dist.lower(), valid_values=ImportanceWeighting.VALID_DISTRIBUTIONS,
                                                        location=location, parameter_name=parameter_name)
        # ParameterValidator.assert_in_valid_list(dataset_dist.lower(), valid_values=ImportanceWeighting.VALID_DISTRIBUTIONS,
        #                                         location=location, parameter_name="dataset_dist")

        assert baseline_dist != dataset_dist, f"{location}: baseline_dist cannot be the same as dataset_dist, found: {baseline_dist}"

        ParameterValidator.assert_type_and_value(pseudocount_value, (int, float), location, "pseudocount_value", min_exclusive=0)

        if lower_weight_limit is not None:
            ParameterValidator.assert_type_and_value(lower_weight_limit, (int, float), location, "lower_weight_limit", min_exclusive=0)

        if upper_weight_limit is not None:
            ParameterValidator.assert_type_and_value(upper_weight_limit, (int, float), location, "upper_weight_limit", min_exclusive=0)

        ParameterValidator.assert_type_and_value(export_weights, bool, location, "export_weights")

        return {
            "baseline_dist": ImportanceWeighting._build_distribution_parameters(baseline_dist, "baseline"),
            "dataset_dist": ImportanceWeighting._build_distribution_parameters(dataset_dist, "dataset"),
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
        self.dataset_positional_frequences = self._get_positional_frequencies(dataset, self.dataset_dist)
        self.baseline_positional_frequences = self._get_positional_frequencies(dataset, self.baseline_dist)

        self.alphabet_size = len(EnvironmentSettings.get_sequence_alphabet())

        self._compute_dataset_probability = self.get_probability_fn(self.dataset_dist)
        self._compute_baseline_probability = self.get_probability_fn(self.baseline_dist)

    def _get_positional_frequencies(self, dataset, dist):
        if dist.distribution_type == ImportanceWeighting.TYPE_MUTAGENESIS:
            if dist.class_name is None:
                return ImportanceWeightHelper.compute_positional_aa_frequences(dataset,
                                                                               pseudocount_value=self.pseudocount_value)
            else:
                varsss = vars(dist)
                label_name = dist.label_name
                class_name = dist.class_name
                result = dataset.get_metadata([label_name], return_df=True)
                result2 = dataset.get_metadata([label_name], return_df=False)
                # todo get subset
        else:
            return None

    def get_weights_for_each_sequence(self, dataset: SequenceDataset):
        return CacheHandler.memo_by_params(self._prepare_caching_params(dataset),
                                           lambda: self._compute_weight_for_each_sequence(dataset))

    def _prepare_caching_params(self, dataset: SequenceDataset):
        return ("compute_weight_for_each_sequence",
                ("dataset_identifier", dataset.identifier),
                ("baseline_dist", vars(self.baseline_dist)),
                ("dataset_dist", vars(self.dataset_dist)),
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

    def get_probability_fn(self, dist):
        if dist.distribution_type == ImportanceWeighting.TYPE_UNIFORM:
            return self._uniform_probability
        elif dist.distribution_type == ImportanceWeighting.TYPE_OLGA:
            return self._olga_probability
        elif dist.distribution_type == ImportanceWeighting.TYPE_MUTAGENESIS:
            if dist.dataset_type == "dataset":
                return partial(self._mutagenesis_probability, positional_frequencies=self.dataset_positional_frequences)
            elif dist.dataset_type == "baseline":
                return partial(self._mutagenesis_probability, positional_frequencies=self.baseline_positional_frequences)

    def _uniform_probability(self, sequence: ReceptorSequence):
        return ImportanceWeightHelper.compute_uniform_probability(sequence.get_sequence(), self.alphabet_size)

    def _mutagenesis_probability(self, sequence: ReceptorSequence, positional_frequencies):
        return ImportanceWeightHelper.compute_mutagenesis_probability(sequence.get_sequence(),
                                                                      positional_frequencies)

    def _olga_probability(self, sequence: ReceptorSequence):
        raise NotImplementedError
