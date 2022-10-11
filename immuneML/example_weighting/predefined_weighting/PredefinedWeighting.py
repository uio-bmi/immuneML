from pathlib import Path

from immuneML.example_weighting.ExampleWeightingParams import ExampleWeightingParams
from immuneML.example_weighting.ExampleWeightingStrategy import ExampleWeightingStrategy
from immuneML.util.ParameterValidator import ParameterValidator


class PredefinedWeighting(ExampleWeightingStrategy):

    def __init__(self, file_path, separator, name):
        super().__init__(name)
        self.file_path = Path(file_path)
        self.separator = separator

    @staticmethod
    def _prepare_parameters(file_path, separator, name):
        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(f"{PredefinedWeighting.__class__.__name__}: example weigths could not be loaded from {file_path}. "
                                    f"Check if the path to the file is properly set.")

        ParameterValidator.assert_type_and_value(separator, str, location=PredefinedWeighting.__name__, parameter_name="separator")

        return {
            "file_path": file_path,
            "separator": separator,
            "name": name
        }

    @staticmethod
    def build_object(dataset=None, **params):
        prepared_params = PredefinedWeighting._prepare_parameters(**params)

        return PredefinedWeighting(**prepared_params)

    def compute_weights(self, dataset, params: ExampleWeightingParams):
        raise NotImplementedError