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
    def load_model(cls, path: Path, device: str = None):

        assert path.exists(), f"{cls.__name__}: {path} does not exist."

        model_overview_file = path / 'model_overview.yaml'
        length_probs_file = path / "length_probabilities.yaml"

        for file in [model_overview_file, length_probs_file]:
            assert file.exists(), f"{cls.__name__}: {file} is not a file."

        length_probs = read_yaml(length_probs_file)
        model_overview = read_yaml(model_overview_file)
        pwm_matrix = {}

        pwm = PWM(locus=model_overview['locus'], sequence_type=model_overview['sequence_type'],
                  region_type=model_overview['region_type'], name=model_overview.get('name'))

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
        super().__init__(locus, name=name, region_type=RegionType[region_type.upper()])
        self.sequence_type = SequenceType[sequence_type.upper()]
        self.pwm_matrix = None
        self.length_probs = None

    def fit(self, data: SequenceDataset, path: Path = None):
        self.set_locus(data)
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
        if seed is not None:
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

    def compute_p_gens(self, sequences, sequence_type: SequenceType, sequence_field: str = None) -> np.ndarray:
        """Generation probability per sequence: the probability of its length, times the product of the
        probabilities of its characters at each position in the PWM for that length. Sequences of a length that was
        not seen in training, or with characters outside the alphabet, get a probability of 0."""
        assert sequence_type == self.sequence_type, \
            f"{PWM.__name__}: expected {self.sequence_type}, got {sequence_type}."
        assert self.pwm_matrix is not None, f"{PWM.__name__}: the model has not been fitted or loaded yet."

        seqs = self._extract_sequence_strings(sequences, sequence_type, sequence_field)
        alphabet = EnvironmentSettings.get_sequence_alphabet(self.sequence_type)
        letter_to_index = {letter: i for i, letter in enumerate(alphabet)}
        p_gens = np.zeros(len(seqs))

        for index, seq in enumerate(seqs):
            length = len(seq)
            if length not in self.pwm_matrix or length == 0 or any(letter not in letter_to_index for letter in seq):
                continue

            letter_indices = [letter_to_index[letter] for letter in seq]
            position_probs = self.pwm_matrix[length][letter_indices, np.arange(length)]
            p_gens[index] = self.length_probs[length] * np.prod(position_probs)

        return p_gens

    def compute_p_gen(self, sequence: dict, sequence_type: SequenceType, sequence_field: str = None) -> float:
        return float(self.compute_p_gens([sequence], sequence_type, sequence_field)[0])

    def _extract_sequence_strings(self, sequences, sequence_type: SequenceType, sequence_field: str = None) -> list:
        """Accepts a bionumpy sequence set (e.g., dataset.data), a list of dicts, or a list of strings, and returns
        plain strings."""
        field = sequence_field if sequence_field is not None \
            else get_sequence_field_name(self.region_type, sequence_type)

        def to_str(value):
            return value.to_string() if hasattr(value, 'to_string') else str(value)

        if hasattr(sequences, field):
            return [to_str(value) for value in getattr(sequences, field)]

        return [to_str(seq[field] if field in seq else seq['sequence']) if isinstance(seq, dict) else to_str(seq)
                for seq in sequences]

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
                              'region_type': self.region_type.name, 'name': self.name})

        return Path(shutil.make_archive(str(path / f'trained_model_{self.name}'), "zip", str(path / 'model'))).absolute()
