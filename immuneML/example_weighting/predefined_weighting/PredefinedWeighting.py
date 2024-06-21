from pathlib import Path
import pandas as pd

from immuneML.example_weighting.ExampleWeightingParams import ExampleWeightingParams
from immuneML.example_weighting.ExampleWeightingStrategy import ExampleWeightingStrategy
from immuneML.util.ParameterValidator import ParameterValidator


class PredefinedWeighting(ExampleWeightingStrategy):
    '''

    Example weighting strategy where weights are supplied in a file.

    **Specification arguments:**

    - file_path (Path): Path to the example weights, should contain the columns 'identifier' and 'example_weight':

      ==========  ==============
      identifier  example_weight
      ==========  ==============
      1           0.5
      2           1
      3           1
      ========  ==============

    - separator (str): Column separator in the input file.

    '''

    def __init__(self, file_path, separator, name: str = None):
        super().__init__(name)
        self.file_path = Path(file_path)
        self.separator = separator

    @staticmethod
    def _prepare_parameters(file_path, separator, name: str = None):
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
        weights_df = self._read_example_weights_file()

        return self._get_example_weights(dataset, weights_df)

    def _get_example_weights(self, dataset, weights_df):
        return [self._get_example_weight_by_identifier(example.identifier, weights_df) for example in dataset.get_data()]

    def _read_example_weights_file(self):
        return pd.read_csv(self.file_path, sep=self.separator, usecols=["identifier", "example_weight"])

    def _get_example_weight_by_identifier(self, identifier, weights_df):
        return float(weights_df[weights_df["identifier"] == identifier].example_weight)
