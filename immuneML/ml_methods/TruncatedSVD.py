from sklearn.decomposition import TruncatedSVD as SklearnTruncatedSVD
from immuneML.ml_methods.DimensionalityReduction import DimensionalityReduction
from immuneML.util.PathBuilder import PathBuilder
from pathlib import Path
import yaml
import dill
import pandas as pd

class TruncatedSVD(DimensionalityReduction):
    @classmethod
    def build_object(cls, **kwargs):
        return TruncatedSVD(parameters=kwargs)

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        _parameters = parameters if parameters is not None else {}
        _parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(TruncatedSVD, self).__init__(parameter_grid=_parameter_grid, parameters=_parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        return SklearnTruncatedSVD(**self._parameters)

    def store(self, path: Path, feature_names=None, details_path: Path = None):
        PathBuilder.build(path)
        file_path = path / f"{self._get_model_filename()}.csv"
        df = pd.DataFrame(self.model.components_)
        with file_path.open("wb") as file:
            df.to_csv(file, index=False)

        if details_path is None:
            params_path = path / f"{self._get_model_filename()}.yaml"
        else:
            params_path = details_path

        with params_path.open("w") as file:
            desc = {
                **(self.get_params()),
                "feature_names": feature_names
            }

            yaml.dump(desc, file)

    def load(self, path: Path, details_path: Path = None):
        name = f"{self._get_model_filename()}.csv"
        file_path = path / name
        if file_path.is_file():
            with file_path.open("rb") as file:
                df = pd.read_csv(file)
                components = df.to_numpy()
                self.model.components_ = components
        else:
            raise FileNotFoundError(f"{self.__class__.__name__} model could not be loaded from {file_path}"
                                    f". Check if the path to the {name} file is properly set.")

        if details_path is None:
            params_path = path / f"{self._get_model_filename()}.yaml"
        else:
            params_path = details_path

        if params_path.is_file():
            with params_path.open("r") as file:
                desc = yaml.safe_load(file)
                for param in ["feature_names"]:
                    if param in desc:
                        setattr(self, param, desc[param])
