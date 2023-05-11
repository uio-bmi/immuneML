
import datetime

import numpy as np
import pandas as pd

from pathlib import Path
from immuneML.ml_methods.GenerativeModel import GenerativeModel
from scripts.specification_util import update_docs_per_mapping
from immuneML.util.PathBuilder import PathBuilder

class PWM(GenerativeModel):
    """
    This is an implementation of Position Weight Matrices as generative models.
    This ML method creates N number og matrices, where N is equal to the number og unique lengths of sequences found
    in the dataset supplied. If load is specified as the instruction rather than training, a previously produced model
    of matrices is loaded.
    Independent of loading or training, this model is then used to generate a user-specified number of sequences.
    Please see thesis regarding the introduction of generative models to immuneML for more information.

    For usage instructions, check :py:obj:`~immuneML.ml_methods.GenerativeModel.GenerativeModel`.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_pwm:
            PWM:
                # params
                amount: 100 # defaults to 100
                sequence_type: nucleotide # specify according to dataset applied
        # alternative way to define ML method with default values:
        my_default_pwm: PWM

    """

    def get_classes(self) -> list:
        pass

    def __init__(self, **parameters):
        parameters = parameters if parameters is not None else {}
        self.model_weight = []
        self.model = []
        super(PWM, self).__init__(parameters=parameters)

    def make_PWM(self, dataset, length):

        matrix = np.zeros(shape=(length, len(self.alphabet)))

        for sequence in dataset:
            for pos, hot in enumerate(sequence):
                matrix[pos][np.argmax(hot)] += 1

        for ind, row in enumerate(matrix):
            matrix[ind] = matrix[ind] / sum(matrix[ind])

        return matrix

    def _get_ml_model(self, X):
        sequences_by_length = {}

        for sequence in X:
            sequence_without_fill = sequence[np.any(sequence == 1, axis=1)]
            if len(sequence_without_fill) not in sequences_by_length:
                sequences_by_length[len(sequence_without_fill)] = [sequence_without_fill]
            else:
                sequences_by_length[len(sequence_without_fill)].append(sequence_without_fill)

        self.model_weight = np.array([len(i) for i in sequences_by_length.values()]) / sum([len(i) for i in sequences_by_length.values()])

        pwms = []
        for length, sequences in sequences_by_length.items():
            pwms.append(self.make_PWM(sequences, length))

        return pwms

    def _fit(self, X, cores_for_training: int = 1, result_path: Path = None):
        self.model = self._get_ml_model(X)
        return self.model

    def generate(self):
        generated_sequences = []
        for i in range(self._amount):
            pwm = np.random.choice(self.model, p=self.model_weight)
            sequence = ""
            for j in range(pwm.shape[0]):
                sequence = sequence + np.random.choice(self.alphabet, 1, p=pwm[j])[0]
            generated_sequences.append(sequence)

        return generated_sequences

    def get_params(self):
        return self._parameters

    def can_predict_proba(self) -> bool:
        raise NotImplementedError

    def get_compatible_encoders(self):
        from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder

        return [OneHotEncoder]

    def load(self, path: Path, details_path: Path = None):

        csv_files = Path(path).glob('*.csv')

        for file in csv_files:
            df = pd.read_csv(file, index_col=False)
            df.pop('alphabet')

            self.model_weight.append(df.pop('weight')[0])
            self.model.append(np.array(df.values).T)


    def store(self, path: Path, feature_names=None, details_path: Path = None):

        PathBuilder.build(path)

        print(f'{datetime.datetime.now()}: Writing to file...')
        for i, pwm in enumerate(self.model):

            file_path = path / f"{self._get_model_filename()}_{pwm.shape[0]}.csv"
            data = {str(ind + 1): numbers for ind, numbers in enumerate(pwm)}
            data['weight'] = self.model_weight[i]
            data["alphabet"] = list(self.alphabet)
            dataframe = pd.DataFrame(data)
            dataframe.to_csv(file_path, index=False)

    @staticmethod
    def get_documentation():
        doc = str(PWM.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.GenerativeModel.GenerativeModel`.": GenerativeModel.get_usage_documentation("LSTM"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc