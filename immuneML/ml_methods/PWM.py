
from Bio import motifs
from immuneML.ml_methods.GenerativeModel import GenerativeModel
from scripts.specification_util import update_docs_per_mapping


class PWM(GenerativeModel):

    def get_classes(self) -> list:
        pass

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        parameters = parameters if parameters is not None else {}
        parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(PWM, self).__init__(parameter_grid=parameter_grid, parameters=parameters)


    def _get_ml_model(self, cores_for_training: int = 2, X=None):

        print(X.get_encoded_repertoire(X))

        self.model = motifs.create(X, alphabet="GPAVLIMCFYWHKRQNEDST")

        print(self.model)
        params = self._parameters

        return self.model

    def _fit(self, X, y, cores_for_training: int = 1):
        self.model = self._get_ml_model(X=X)
        print("Her er motif", self.model)

    def get_params(self):
        return self.model.get_params(deep=True)

    def can_predict_proba(self) -> bool:
        raise Exception("can_predict_proba has not been implemented")

    def get_compatible_encoders(self):
        raise Exception("get_compatible_encoders has not been implemented")

    @staticmethod
    def get_documentation():
        doc = str(PWM.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.": GenerativeModel.get_usage_documentation("LSTM"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc