import shutil
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from immuneML.data_model.bnp_util import write_yaml, read_yaml
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.ml_methods.pytorch_implementations.SimpleLSTMGenerator import SimpleLSTMGenerator
from immuneML.ml_methods.util.pytorch_util import store_weights
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder


class SimpleLSTM(GenerativeModel):
    """
    This is a simple generative model for receptor sequences based on LSTM.

    Similar models have been proposed in:

    Akbar, R. et al. (2022). In silico proof of principle of machine learning-based antibody design at unconstrained scale. mAbs, 14(1), 2031482. https://doi.org/10.1080/19420862.2022.2031482

    Saka, K. et al. (2021). Antibody design using LSTM based deep generative model from phage display library for affinity maturation. Scientific Reports, 11(1), Article 1. https://doi.org/10.1038/s41598-021-85274-7


    Specification arguments:

    - sequence_type (str): whether the model should work on amino_acid or nucleotide level

    - hidden_size (int): how many LSTM cells should exist per layer

    - num_layers (int): how many hidden LSTM layers should there be

    - num_epochs (int): for how many epochs to train the model

    - learning_rate (float): what learning rate to use for optimization

    - batch_size (int): how many examples (sequences) to use for training for one batch

    - embed_size (int): the dimension of the sequence embedding

    - temperature (float)

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_simple_lstm:
            sequence_type: amino_acid
            hidden_size: 50
            num_layers: 1
            num_epochs: 5000
            learning_rate: 0.001
            batch_size: 100
            embed_size: 100


    """

    @classmethod
    def load_model(cls, path: Path):
        assert path.exists(), f"{cls.__name__}: {path} does not exist."

        model_overview_file = path / 'model_overview.yaml'
        state_dict_file = path / 'state_dict.yaml'

        for file in [model_overview_file, state_dict_file]:
            assert file.exists(), f"{cls.__name__}: {file} is not a file."

        model_overview = read_yaml(model_overview_file)
        lstm = SimpleLSTM(**{k: v for k, v in model_overview.items() if k != 'type'})
        lstm._model = lstm.make_new_model(state_dict_file)
        return lstm

    ITER_TO_REPORT = 100

    def __init__(self, chain: str, sequence_type: str, hidden_size: int, learning_rate: float, num_epochs: int,
                 batch_size: int, num_layers: int, embed_size: int, temperature, device: str, name=None,
                 region_type: str = None):

        super().__init__(Chain.get_chain(chain))
        self._model = None
        self.region_type = RegionType[region_type.upper()] if region_type else None
        self.sequence_type = SequenceType[sequence_type.upper()]
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.temperature = temperature
        self.name = name
        self.device = device
        self.unique_letters = EnvironmentSettings.get_sequence_alphabet(self.sequence_type) + ["*"]
        self.num_letters = len(self.unique_letters)
        self.letter_to_index = {letter: i for i, letter in enumerate(self.unique_letters)}
        self.index_to_letter = {i: letter for letter, i in self.letter_to_index.items()}
        self.loss_summary_path = None

    def make_new_model(self, state_dict_file: Path = None):
        model = SimpleLSTMGenerator(input_size=self.num_letters, hidden_size=self.hidden_size,
                                    embed_size=self.embed_size, output_size=self.num_letters,
                                    batch_size=self.batch_size)

        if isinstance(state_dict_file, Path) and state_dict_file.is_file():
            state_dict = read_yaml(state_dict_file)
            state_dict = {k: torch.as_tensor(v, device=self.device) for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

        model.to(self.device)

        return model

    def fit(self, data, path: Path = None):
        data_loader = self._encode_dataset(data)

        model = self.make_new_model()
        model.train()

        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), self.learning_rate)
        loss_summary = {"loss": [], "epoch": []}

        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(self.num_epochs):
                loss = 0.
                state = model.init_zero_state()
                optimizer.zero_grad()

                for x_batch, y_batch in data_loader:
                    state = state[0][:, :x_batch.size(0), :], state[1][:, :x_batch.size(0), :]

                    outputs, state = model(x_batch, state)
                    loss = loss + criterion(outputs, y_batch)

                loss = loss / len(data_loader.dataset)
                loss.backward()
                optimizer.step()

                loss_summary = self._log_training_progress(loss_summary, epoch, loss)

        self._log_training_summary(loss_summary, path)

        self._model = model

    def _log_training_summary(self, loss_summary, path):
        if path is not None:
            PathBuilder.build(path)
            self.loss_summary_path = path / 'loss_summary.csv'
            pd.DataFrame(loss_summary).to_csv(str(self.loss_summary_path), index=False)

    def _log_training_progress(self, loss_summary, epoch, loss):
        if (epoch + 1) % SimpleLSTM.ITER_TO_REPORT == 0:
            print_log(f"{SimpleLSTM.__name__}: Epoch [{epoch + 1}/{self.num_epochs}]: loss: {loss.item():.4f}",
                      True)
            loss_summary['loss'].append(loss.item())
            loss_summary['epoch'].append(epoch + 1)
        return loss_summary

    def _encode_dataset(self, dataset):
        dataset_attributes = dataset.get_attributes([self.sequence_type.value, 'region_type'], as_list=True)

        unique_region_types = list(set(dataset_attributes['region_type']))
        assert len(unique_region_types) == 1, \
            f'{SimpleLSTM.__name__}: only one region type in the dataset is supported: {unique_region_types}.'
        self.region_type = RegionType[unique_region_types[0].upper()]

        sequences = dataset_attributes[self.sequence_type.value]

        sequences = list(chain.from_iterable(
            [[self.letter_to_index[letter] for letter in seq] + [self.letter_to_index['*']] for seq in sequences]))
        sequences = torch.as_tensor(sequences, device=self.device).long()

        return DataLoader(TensorDataset(sequences[:-1], sequences[1:]), shuffle=True, batch_size=self.batch_size)

    def is_same(self, model) -> bool:
        raise NotImplementedError

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType, compute_p_gen: bool):
        torch.manual_seed(seed)

        self._model.eval()
        prime_str = "CAS"
        input_vector = torch.as_tensor([self.letter_to_index[letter] for letter in prime_str], device=self.device).long()
        predicted = prime_str

        with torch.no_grad():

            state = self._model.init_zero_state(batch_size=1)

            for p in range(len(prime_str) - 1):
                _, state = self._model(input_vector[p], state)

            inp = input_vector[-1]
            gen_seq_count = 0

            while gen_seq_count <= count:
                output, state = self._model(inp, state)

                output_dist = output.data.view(-1).div(self.temperature).exp()
                top_i = torch.multinomial(output_dist, 1)[0].item()

                predicted_char = self.index_to_letter[top_i]
                predicted += predicted_char
                inp = torch.as_tensor(self.letter_to_index[predicted_char], device=self.device).long()
                if predicted_char == "*":
                    gen_seq_count += 1

        print_log(f"{SimpleLSTM.__name__} {self.name}: generated {count} sequences.", True)

        sequences = predicted.split('*')[1:-1]
        return self._export_dataset(sequences, count, path)

    def _export_dataset(self, sequences, count, path):
        sequence_objs = [ReceptorSequence(**{
            self.sequence_type.value: sequence,
            'metadata': SequenceMetadata(region_type=self.region_type.name, chain=self.chain.name)
        }) for i, sequence in enumerate(sequences)]

        dataset = SequenceDataset.build_from_objects(sequence_objs, count, path, 'synthetic_lstm')

        return dataset

    def compute_p_gens(self, sequences, sequence_type: SequenceType) -> np.ndarray:
        raise RuntimeError

    def compute_p_gen(self, sequence: dict, sequence_type: SequenceType) -> float:
        raise RuntimeError

    def can_compute_p_gens(self) -> bool:
        return False

    def can_generate_from_skewed_gene_models(self) -> bool:
        return False

    def generate_from_skewed_gene_models(self, v_genes: list, j_genes: list, seed: int, path: Path,
                                         sequence_type: SequenceType, batch_size: int, compute_p_gen: bool):
        raise RuntimeError

    def save_model(self, path: Path) -> Path:
        model_path = PathBuilder.build(path / 'model')

        skip_keys_for_export = ['_model', 'loss_summary_path', 'index_to_letter', 'letter_to_index',
                                'unique_letters', 'num_letters']
        write_yaml(filename=model_path / 'model_overview.yaml',
                   yaml_dict={**{k: v for k, v in vars(self).items() if k not in skip_keys_for_export},
                              **{'type': self.__class__.__name__, 'region_type': self.region_type.name,
                                 'sequence_type': self.sequence_type.name, 'chain': self.chain.name}})

        store_weights(self._model, model_path / 'state_dict.yaml')

        return Path(shutil.make_archive(str(path / 'trained_model'), 'zip', str(model_path))).absolute()
