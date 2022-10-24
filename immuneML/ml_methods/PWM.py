import csv
import yaml
import datetime

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

        instances = np.array([list(sequence.get_sequence()) for repertoire in dataset.get_data() for sequence in repertoire.sequences])

        alphabet = ""

        for instance in instances:
            for letter in instance:
                alphabet = "".join(set(letter + alphabet))
                if len(alphabet) == 20:
                    break

        self.alphabet = "".join(sorted(alphabet))
        matrix = np.zeros(shape=(instances.shape[1], len(self.alphabet)))

        instances = instances.T
        for x, pos in enumerate(instances):
            for i, element in enumerate(pos):
                for y, char in enumerate(list(self.alphabet)):
                    if element == char:
                        matrix[x][y] += 1
                        break

        matrix = matrix / sum(matrix)

        return matrix

    def _fit(self, X, y, cores_for_training: int = 1, dataset=None):
        self.model = self._get_ml_model(cores_for_training, X, dataset)

        print(self.model)

        return self.model

    def generate(self, length_of_sequences: int = None, amount=10, path_to_model: Path = None):

        if self.model is None:
            model_as_array = []
            with open(path_to_model, 'r') as file:

                reader = csv.reader(file)
                self.alphabet = "".join(next(reader))
                for row in reader:
                    model_as_array.append(row)
            self.model = np.array(model_as_array)

        length_of_sequences = length_of_sequences if length_of_sequences is not None else self.model.shape(0)
        generated_sequences = []
        for i in range(amount):
            sequence = []
            for j in range(length_of_sequences):
                sequence.append(np.random.choice(list(self.alphabet), p=self.model[i]))
            generated_sequences.append(sequence)
        print(generated_sequences)

    def get_params(self):
        return self._parameters

    def can_predict_proba(self) -> bool:
        raise Exception("can_predict_proba has not been implemented")

    def get_compatible_encoders(self):
        raise Exception("get_compatible_encoders has not been implemented")

    def store(self, path: Path, feature_names=None, details_path: Path = None):

        PathBuilder.build(path)

        print(f'{datetime.datetime.now()}: Writing to file')
        file_path = path / f"{self._get_model_filename()}.csv"
        with open(file_path, "w") as file:
            writer = csv.writer(file)
            writer.writerow(list(self.alphabet))
            for row in self.model:
                writer.writerow(row)


        if details_path is None:
            params_path = path / f"{self._get_model_filename()}.yaml"
        else:
            params_path = details_path

        with params_path.open("w") as file:
            desc = {
                **(self.get_params()),
                "feature_names": feature_names,
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