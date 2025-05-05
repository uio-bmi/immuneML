import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from immuneML.data_model.SequenceParams import Chain
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.bnp_util import write_yaml, read_yaml, get_sequence_field_name
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.util.PathBuilder import PathBuilder


class PWM(GenerativeModel):
    """

    This is a baseline implementation of a positional weight matrix. It is estimated from a set of sequences for each
    of the different lengths that appear in the dataset.


    **Specification arguments:**

    - locus (str): which chain is generated (for now, it is only assigned to the generated sequences)

    - sequence_type (str): amino_acid or nucleotide

    - region_type (str): which region type to use (e.g., IMGT_CDR3), this is only assigned to the generated sequences


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_pwm:
                    PWM:
                        locus: beta
                        sequence_type: amino_acid
                        region_type: IMGT_CDR3

    """

    @classmethod
    def load_model(cls, path: Path):

        assert path.exists(), f"{cls.__name__}: {path} does not exist."

        model_overview_file = path / 'model_overview.yaml'
        length_probs_file = path / "length_probabilities.yaml"

        for file in [model_overview_file, length_probs_file]:
            assert file.exists(), f"{cls.__name__}: {file} is not a file."

        length_probs = read_yaml(length_probs_file)
        model_overview = read_yaml(model_overview_file)
        pwm_matrix = {}

        pwm = PWM(locus=model_overview['locus'], sequence_type=model_overview['sequence_type'],
                  region_type=model_overview['region_type'])

        for file, length in [(path / f'pwm_len_{length}.csv', length) for length in length_probs.keys()]:
            assert file.exists(), f"{cls.__name__}: {file} is not a file."
            pwm_matrix[length] = pd.read_csv(str(file))

            assert pwm_matrix[length].iloc[:, 0].tolist() == EnvironmentSettings.get_sequence_alphabet(
                pwm.sequence_type), \
                (f"{cls.__name__}: the row names in the PWM for length {length} don't match the expected row names.\n"
                 f"Obtained:\n{pwm_matrix[length].index.tolist()}\nExpected:"
                 f"\n{EnvironmentSettings.get_sequence_alphabet(pwm.sequence_type)}")

            pwm_matrix[length] = pwm_matrix[length].iloc[:, 1:].values

            expected_shape = (len(EnvironmentSettings.get_sequence_alphabet(pwm.sequence_type)), length)
            assert pwm_matrix[length].shape == expected_shape, \
                (f"{cls.__name__}: PWM matrix for length {length} has shape {pwm_matrix[length].shape}, "
                 f"but expected {expected_shape}.")

        pwm.length_probs = length_probs
        pwm.pwm_matrix = pwm_matrix
        return pwm

    def __init__(self, locus, sequence_type: str, region_type: str, name: str = None):
        super().__init__(Chain.get_chain(locus), name=name, region_type=RegionType[region_type.upper()])
        self.sequence_type = SequenceType[sequence_type.upper()]
        self.pwm_matrix = None
        self.length_probs = None

    def fit(self, data: SequenceDataset, path: Path = None):
        sequences = data.get_attribute(get_sequence_field_name(self.region_type, self.sequence_type))
        lengths, counts = np.unique(sequences.lengths, return_counts=True)

        self.length_probs = dict(zip(lengths, counts))
        self.length_probs = {int(length): float(count / sum(counts)) for length, count in self.length_probs.items()}
        self.pwm_matrix = {length: None for length in lengths}

        alphabet = EnvironmentSettings.get_sequence_alphabet(self.sequence_type)

        for length in lengths:
            seq_subset = sequences[sequences.lengths == length]
            self.pwm_matrix[length] = np.zeros((len(alphabet), length))
            for position in range(length):
                for i, letter in enumerate(alphabet):
                    self.pwm_matrix[length][i, position] = np.sum([1 for seq in seq_subset if seq[position] == letter])

            # Normalize counts to obtain probabilities
            total_counts = np.sum(self.pwm_matrix[length], axis=0)
            self.pwm_matrix[length] = self.pwm_matrix[length] / total_counts

    def is_same(self, model) -> bool:
        raise NotImplementedError

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType, compute_p_gen: bool):
        np.random.seed(seed)
        assert sequence_type == self.sequence_type

        sequences = []

        for _ in range(count):
            sequence_length = np.random.choice(list(self.length_probs.keys()), p=list(self.length_probs.values()))
            sequence = "".join(np.random.choice(EnvironmentSettings.get_sequence_alphabet(self.sequence_type),
                                                p=self.pwm_matrix[sequence_length][:, i])
                               for i in range(sequence_length))

            sequences.append(sequence)

        dataset = self._export_gen_dataset(sequences, path)

        return dataset

    def _export_gen_dataset(self, sequences: List[str], path: Path) -> SequenceDataset:
        count = len(sequences)
        df = pd.DataFrame({get_sequence_field_name(self.region_type, self.sequence_type): sequences,
                           'locus': [self.locus.to_string() for _ in range(count)],
                           'gen_model_name': [self.name for _ in range(count)]})

        return SequenceDataset.build_from_partial_df(df, PathBuilder.build(path), 'synthetic_dataset',
                                                     {'gen_model_name': [self.name]}, {'gen_model_name': str})

    def compute_p_gens(self, sequences, sequence_type: SequenceType) -> np.ndarray:
        raise NotImplementedError

    def compute_p_gen(self, sequence: dict, sequence_type: SequenceType) -> float:
        raise NotImplementedError

    def can_compute_p_gens(self) -> bool:
        return True

    def can_generate_from_skewed_gene_models(self) -> bool:
        return False

    def generate_from_skewed_gene_models(self, v_genes: list, j_genes: list, seed: int, path: Path,
                                         sequence_type: SequenceType, batch_size: int, compute_p_gen: bool):
        raise RuntimeError

    def save_model(self, path: Path) -> Path:
        model_path = PathBuilder.build(path / 'model')
        write_yaml(yaml_dict=self.length_probs, filename=model_path / 'length_probabilities.yaml')
        for length in self.pwm_matrix:
            (pd.DataFrame(data=self.pwm_matrix[length], columns=list(range(length)),
                          index=EnvironmentSettings.get_sequence_alphabet(self.sequence_type))
             .to_csv(model_path / f'pwm_len_{length}.csv'))

        write_yaml(filename=model_path / 'model_overview.yaml',
                   yaml_dict={'type': 'PWM', 'locus': self.locus.name, 'sequence_type': self.sequence_type.name,
                              'region_type': self.region_type.name})

        return Path(shutil.make_archive(str(path / 'trained_model'), "zip", str(path / 'model'))).absolute()
