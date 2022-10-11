from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.example_weighting.ExampleWeightingParams import ExampleWeightingParams
from immuneML.example_weighting.ExampleWeightingStrategy import ExampleWeightingStrategy
from immuneML.util.ParameterValidator import ParameterValidator


class ImportanceWeighting(ExampleWeightingStrategy):
    VALID_DISTRIBUTIONS = ("uniform", "mutagenesis", "olga")

    def __init__(self, baseline_dist, dataset_dist, name):
        super().__init__(name)
        self.baseline_dist = baseline_dist
        self.dataset_dist = dataset_dist

    @staticmethod
    def _prepare_parameters(baseline_dist, dataset_dist, name):

        ParameterValidator.assert_in_valid_list(baseline_dist.lower(), valid_values=ImportanceWeighting.VALID_DISTRIBUTIONS,
                                                location=ImportanceWeighting.__name__, parameter_name="baseline_dist")
        ParameterValidator.assert_in_valid_list(dataset_dist.lower(), valid_values=ImportanceWeighting.VALID_DISTRIBUTIONS,
                                                location=ImportanceWeighting.__name__, parameter_name="dataset_dist")

        return {
            "baseline_dist": baseline_dist.lower(),
            "dataset_dist": dataset_dist.lower(),
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

    def compute_weights(self, dataset, params: ExampleWeightingParams):
        raise NotImplementedError