import csv
import yaml
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from immuneML.ml_methods.GenerativeModel import GenerativeModel
from scripts.specification_util import update_docs_per_mapping
from immuneML.util.PathBuilder import PathBuilder

class LSTM(GenerativeModel):

    def get_classes(self) -> list:
        pass

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        parameters = parameters if parameters is not None else {}
        parameter_grid = parameter_grid if parameter_grid is not None else {}
        self.alphabet = ""
        super(LSTM, self).__init__(parameter_grid=parameter_grid, parameters=parameters)


    def _get_ml_model(self, cores_for_training: int = 2, X=None):


        """
        :param cores_for_training:
        :param X:
        :return: keras.Sequential object

        The initial parameters set have been determined through testing on a previous project using LSTM. It is
        worthwhile considering changing these.
        """
        embedding_dim = 256
        rnn_units = 1024
        seq_length = 42  # window size (w)
        batch_size = 64
        buffer_size = 1000

        instances = np.array(
            [list(sequence.get_sequence()) for repertoire in X.get_data() for sequence in repertoire.sequences])

        self.alphabet = sorted(set(instances.view().reshape(instances.shape[0] * instances.shape[
            1])))  # Mashes all data into 1 dimension and uses the set function to find the unique characters

        vocab_size = len(self.alphabet)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
            tf.keras.layers.LSTM(rnn_units,
                                 return_sequences=True,
                                 stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])

        return self.model

    def _fit(self, X, cores_for_training: int = 1):

        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        test_size = int(0.15 * len(X))
        print('Trains batches {}, val batches {}, test batches {}'.format(train_size, val_size, test_size))

        train_dataset = X.take(train_size)
        test_dataset = dataset.skip(train_size)
        val_dataset = test_dataset.skip(val_size)
        test_dataset = test_dataset.take(test_size)

        self.model = self._get_ml_model(cores_for_training, X)

        history = self.model.fit(train_dataset, epochs=nb_epoch,
                            callbacks=[checkpoint_callback],
                            validation_data=val_dataset)
        history_outfile = 'outfiles/%s_history.txt'

        history_contents = []
        for key in history.history:
            for i, val in enumerate(history.history[key]):
                history_content = [key, i + 1, val]
                history_contents.append(history_content)
        historydf = pd.DataFrame(history_contents, columns=['loss_cat', 'epoch', 'value'])
        historydf.to_csv(history_outfile, index=False)

        return self.model

    def generate(self, length_of_sequences: int = None, amount=10, path_to_model: Path = None):

        if self.model is None:
            model_as_array = []
            print(f'{datetime.datetime.now()}: Fetching model...')
            with open(path_to_model, 'r') as file:

                reader = csv.reader(file)
                self.alphabet = "".join(next(reader))
                for row in reader:
                    model_as_array.append(row)
            self.model = np.array(model_as_array)

        length_of_sequences = length_of_sequences if length_of_sequences is not None else self.model.shape[0]
        generated_sequences = []
        for _ in range(amount):
            sequence = []
            for i in range(length_of_sequences):
                sequence.append(np.random.choice(list(self.alphabet), 1, p=self.model[i]/100)[0])
            generated_sequences.append(sequence)

        instances = np.array(generated_sequences)

        matrix = np.zeros(shape=(instances.shape[1], len(self.alphabet)))

        instances = instances.T

        for x, pos in enumerate(instances):
            for i, element in enumerate(pos):
                for y, char in enumerate(list(self.alphabet)):
                    if element == char:
                        matrix[x][y] += 1
                        break

        for ind, row in enumerate(matrix):
            matrix[ind] = matrix[ind] / sum(matrix[ind]) * 100
        matrix = np.around(matrix, 2)
        return_sequences = []
        instances = instances.T

        for row in instances:
            return_sequences.append("".join(row))

        matrix = matrix.T

        return list(matrix), instances, self.alphabet

    def get_params(self):
        return self._parameters

    def can_predict_proba(self) -> bool:
        raise Exception("can_predict_proba has not been implemented")

    def get_compatible_encoders(self):
        raise Exception("get_compatible_encoders has not been implemented")

    def load(self, path: Path, details_path: Path = None):

        name = f"{self._get_model_filename()}.csv"
        file_path = path / name
        if file_path.is_file():
            dataframe = file_path
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

    def store(self, path: Path, feature_names=None, details_path: Path = None):

        PathBuilder.build(path)

        print(f'{datetime.datetime.now()}: Writing to file...')
        file_path = path / f"{self._get_model_filename()}.csv"
        data = {"LSTM": self.model, "alphabet": self.alphabet}
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(file_path)


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
        doc = str(LSTM.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.": GenerativeModel.get_usage_documentation("LSTM"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc