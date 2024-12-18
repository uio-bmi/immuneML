import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from immuneML.data_model.SequenceParams import Chain, RegionType
from immuneML.data_model.bnp_util import read_yaml, get_sequence_field_name, make_full_airr_seq_set_df, write_yaml
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.util.PathBuilder import PathBuilder


class MarkovModel(GenerativeModel):

    @classmethod
    def load_model(cls, path: Path):
        assert path.exists(), f"{cls.__name__}: {path} does not exist."

        model_overview_file = path / 'model_overview.yaml'
        transition_probs_file = path / "transition_probabilities.yaml"
        initial_probs_file = path / "initial_probabilities.yaml"

        for file in [model_overview_file, transition_probs_file, initial_probs_file]:
            assert file.exists(), f"{cls.__name__}: {file} is not a file."

        transition_probs = read_yaml(transition_probs_file)
        initial_probs = read_yaml(initial_probs_file)
        model_overview = read_yaml(model_overview_file)

        markov_model = MarkovModel(locus=model_overview['locus'],
                                   sequence_type=model_overview['sequence_type'],
                                   region_type=model_overview['region_type'])

        markov_model.transition_probs = transition_probs
        markov_model.initial_probs = initial_probs
        return markov_model

    def __init__(self, locus, sequence_type: str, region_type: str, name: str = None):
        super().__init__(Chain.get_chain(locus), name=name)
        self.sequence_type = SequenceType[sequence_type.upper()]
        self.region_type = RegionType[region_type.upper()]
        self.transition_probs = None
        self.initial_probs = None

    def fit(self, data: SequenceDataset, path: Path = None):
        sequences = data.get_attribute(get_sequence_field_name(self.region_type, self.sequence_type))
        alphabet = EnvironmentSettings.get_sequence_alphabet(self.sequence_type)
        self.transition_probs = {char: {char2: 0 for char2 in alphabet + ["END"]} for char in alphabet}
        self.initial_probs = {char: 0 for char in alphabet}

        for seq in sequences:
            self.initial_probs[seq[0]] += 1
            for i in range(len(seq) - 1):
                self.transition_probs[seq[i]][seq[i + 1]] += 1
            self.transition_probs[seq[-1]]["END"] += 1

        total_sequences = sum(self.initial_probs.values())
        self.initial_probs = {char: count / total_sequences for char, count in self.initial_probs.items()}

        for char in self.transition_probs:
            total_transitions = sum(self.transition_probs[char].values())
            if total_transitions > 0:
                self.transition_probs[char] = {char2: count / total_transitions for char2, count in
                                               self.transition_probs[char].items()}

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType, compute_p_gen: bool):
        np.random.seed(seed)
        assert sequence_type == self.sequence_type
        alphabet = EnvironmentSettings.get_sequence_alphabet(self.sequence_type)
        sequences = []

        for _ in range(count):
            first_char = np.random.choice(alphabet, p=[self.initial_probs[char] for char in alphabet])
            sequence = [first_char]

            while True:
                current_char = sequence[-1]
                next_char = np.random.choice(alphabet + ["END"],
                                             p=[self.transition_probs[current_char][char] for char in
                                                alphabet + ["END"]])
                if next_char == "END":
                    break
                sequence.append(next_char)

            sequences.append("".join(sequence))
        dataset = self._export_gen_dataset(sequences, path)
        return dataset

    def _export_gen_dataset(self, sequences: List[str], path: Path) -> SequenceDataset:
        count = len(sequences)
        df = pd.DataFrame({
            'sequence_id': [i+1 for i in range(count)],  # Unique IDs for sequences
            get_sequence_field_name(self.region_type, self.sequence_type): sequences,
            'locus': [self.locus for _ in range(count)],
            'gen_model_name': [self.name for _ in range(count)]
        })

        df = make_full_airr_seq_set_df(df)

        filename = str(PathBuilder.build(path) / 'synthetic_dataset.tsv')
        df.to_csv(filename, sep='\t', index=False)

        write_yaml(path / 'synthetic_metadata.yaml', {
            'dataset_type': 'SequenceDataset',
            'filename': filename,
            'type_dict_dynamic_fields': {'gen_model_name': 'str'},
            'name': 'synthetic_dataset', 'labels': {'gen_model_name': [self.name]},
            'timestamp': str(datetime.now())
        })

        return SequenceDataset.build(path / 'synthetic_dataset.tsv', path / 'synthetic_metadata.yaml',
                                     'synthetic_dataset')

    def compute_p_gens(self, sequences, sequence_type: SequenceType) -> np.ndarray:
        raise NotImplementedError

    def compute_p_gen(self, sequence: dict, sequence_type: SequenceType) -> float:
        raise NotImplementedError

    def can_compute_p_gens(self) -> bool:
        return False

    def can_generate_from_skewed_gene_models(self) -> bool:
        return False

    def generate_from_skewed_gene_models(self, v_genes: list, j_genes: list, seed: int, path: Path,
                                         sequence_type: SequenceType, batch_size: int, compute_p_gen: bool):
        raise RuntimeError

    def save_model(self, path: Path) -> Path:
        model_path = PathBuilder.build(path / 'model')
        write_yaml(yaml_dict=self.transition_probs, filename=model_path / 'transition_probabilities.yaml')
        write_yaml(yaml_dict=self.initial_probs, filename=model_path / 'initial_probabilities.yaml')

        write_yaml(filename=model_path / 'model_overview.yaml',
                   yaml_dict={'type': 'MarkovModel', 'locus': self.locus.name, 'sequence_type': self.sequence_type.name,
                              'region_type': self.region_type.name})

        return Path(shutil.make_archive(str(path / 'trained_model'), "zip", str(path / 'model'))).absolute()
