import numpy as np

from pathlib import Path
from immuneML.ml_methods.GenerativeModel import GenerativeModel
from scripts.specification_util import update_docs_per_mapping
from immuneML.util.PathBuilder import PathBuilder

class PWM(GenerativeModel):

    def get_classes(self) -> list:
        pass

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        parameters = parameters if parameters is not None else {}
        parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(PWM, self).__init__(parameter_grid=parameter_grid, parameters=parameters)


    def _get_ml_model(self, cores_for_training: int = 2, X=None, dataset=None):

        instances = np.array([sequence.get_sequence() for repertoire in dataset.get_data() for sequence in repertoire.sequences])

        print(instances)

        alphabet = ""

        for instance in instances:
            alphabet = "".join(set(instance + alphabet))
            if len(alphabet) == 20:
                break

        print(alphabet)

        matrix = np.zeros(shape=(len(max(instances)), len(alphabet)))
        print(matrix)

        params = self._parameters

        return self.model

    def _fit(self, X, y, cores_for_training: int = 1, dataset=None):
        self.model = self._get_ml_model(cores_for_training, X, dataset)
        self.model.weblogo("mymotif.jpg")
        pwm = self.model.counts.normalize(pseudocounts=0.5)
        print(pwm)
        return pwm

    def get_params(self):
        return self.model.get_params(deep=True)

    def can_predict_proba(self) -> bool:
        raise Exception("can_predict_proba has not been implemented")

    def get_compatible_encoders(self):
        raise Exception("get_compatible_encoders has not been implemented")

    def store(self, path: Path, feature_names=None, details_path: Path = None):
        PathBuilder.build(path)
        file_path = path / f"{self._get_model_filename()}.pickle"
        with file_path.open("wb") as file:
            dill.dump(self.model, file)

        if details_path is None:
            params_path = path / f"{self._get_model_filename()}.yaml"
        else:
            params_path = details_path

        with params_path.open("w") as file:
            desc = {
                **(self.get_params()),
                "feature_names": feature_names,
                "classes": self.model.classes_.tolist(),
                "class_mapping": self.class_mapping,
            }

            if self.label is not None:
                desc["label"] = vars(self.label)

            yaml.dump(desc, file)

    @staticmethod
    def get_documentation():
        doc = str(PWM.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.": GenerativeModel.get_usage_documentation("LSTM"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc