
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


    def _get_ml_model(self, cores_for_training: int = 2, X=None, dataset=None):

        instances = [sequence.get_sequence() for repertoire in dataset.get_data() for sequence in repertoire.sequences]

        self.model = motifs.create(instances, alphabet="GPAVLIMCFYWHKRQNEDST")

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

    @staticmethod
    def get_documentation():
        doc = str(PWM.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.": GenerativeModel.get_usage_documentation("LSTM"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc