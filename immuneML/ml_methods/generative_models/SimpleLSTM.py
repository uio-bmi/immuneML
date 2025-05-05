import logging
import shutil
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

from immuneML.data_model.SequenceParams import RegionType, Chain
from immuneML.data_model.bnp_util import write_yaml, read_yaml, get_sequence_field_name
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
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


    **Specification arguments:**

    - sequence_type (str): whether the model should work on amino_acid or nucleotide level

    - hidden_size (int): how many LSTM cells should exist per layer

    - num_layers (int): how many hidden LSTM layers should there be

    - num_epochs (int): for how many epochs to train the model

    - learning_rate (float): what learning rate to use for optimization

    - batch_size (int): how many examples (sequences) to use for training for one batch

    - embed_size (int): the dimension of the sequence embedding

    - temperature (float): a higher temperature leads to faster yet more unstable learning

    - prime_str (str): the initial sequence to start generating from

    - seed (int): random seed for the model or None

    - iter_to_report (int): number of epochs between training progress reports


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
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

    def __init__(self, locus: str, sequence_type: str, hidden_size: int, learning_rate: float, num_epochs: int,
                 batch_size: int, num_layers: int, embed_size: int, temperature, device: str, name=None,
                 region_type: str = RegionType.IMGT_CDR3.name, prime_str: str = "C", window_size: int = 64,
                 seed: int = None, iter_to_report: int = 1):

        super().__init__(Chain.get_chain(locus), region_type=RegionType.get_object(region_type), name=name, seed=seed)
        self._model = None
        self.sequence_type = SequenceType[sequence_type.upper()] if sequence_type else SequenceType.AMINO_ACID
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.temperature = temperature
        self.prime_str = prime_str
        self.window_size = window_size
        self.device = device
        self.iter_to_report = iter_to_report
        self.unique_letters = EnvironmentSettings.get_sequence_alphabet(self.sequence_type) + ["*"]
        self.num_letters = len(self.unique_letters)
        self.letter_to_index = {letter: i for i, letter in enumerate(self.unique_letters)}
        self.index_to_letter = {i: letter for letter, i in self.letter_to_index.items()}
        self.loss_summary_path = None

    def make_new_model(self, state_dict_file: Path = None):
        from torch import as_tensor
        model = SimpleLSTMGenerator(input_size=self.num_letters, hidden_size=self.hidden_size,
                                    embed_size=self.embed_size, output_size=self.num_letters,
                                    batch_size=self.batch_size, device=self.device)

        if isinstance(state_dict_file, Path) and state_dict_file.is_file():
            state_dict = read_yaml(state_dict_file)
            state_dict = {k: as_tensor(v, device=self.device) for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

        model.to(self.device)

        return model

    def _log_training_progress(self, loss_summary, epoch, loss):
        if (epoch + 1) % self.iter_to_report == 0:
            message = f"{SimpleLSTM.__name__}: Epoch [{epoch + 1}/{self.num_epochs}]: loss: {loss:.4f}"
            print_log(message, True)

            loss_summary['loss'].append(loss)
            loss_summary['epoch'].append(epoch + 1)
        return loss_summary

    def fit(self, data, path: Path = None):
        import torch
        from torch import nn, optim

        if self.seed is not None:
            torch.manual_seed(self.seed)

        data_loader = self._encode_dataset(data)
        model = self.make_new_model()
        model.train()

        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), self.learning_rate)
        loss_summary = {"loss": [], "epoch": []}

        for epoch in range(self.num_epochs):
            epoch_loss = 0.
            num_batches = 0

            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                # Initialize state for each batch
                state = model.init_zero_state(x_batch.size(0))

                optimizer.zero_grad()

                outputs, _ = model(x_batch, state)
                outputs = outputs.reshape(-1, outputs.size(-1))
                y_batch = y_batch.reshape(-1)

                loss = criterion(outputs, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches
            loss_summary = self._log_training_progress(loss_summary, epoch, avg_epoch_loss)

        self._log_training_summary(loss_summary, path)
        self._model = model

    def _log_training_summary(self, loss_summary, path):
        if path is not None:
            try:
                PathBuilder.build(path)
                self.loss_summary_path = path / 'loss_summary.csv'
                pd.DataFrame(loss_summary).to_csv(str(self.loss_summary_path), index=False)
            except Exception as e:
                logging.error(f"{SimpleLSTM.__name__}: failed to save loss summary: {e};\n{loss_summary}")

    def _encode_dataset(self, dataset: SequenceDataset):
        from torch import as_tensor
        from torch.utils.data import DataLoader, TensorDataset

        seq_col = get_sequence_field_name(self.region_type, self.sequence_type)
        sequences = dataset.get_attribute(seq_col).tolist()

        # Flatten all sequences into one long sequence with end tokens
        sequences = list(chain.from_iterable(
            [[self.letter_to_index[letter] for letter in seq] + [self.letter_to_index['*']] for seq in sequences]))

        sequences = as_tensor(sequences, device=self.device).long()

        # Create overlapping windows of size window_size
        # stride=1 means each window overlaps with previous window by window_size-1 elements
        windows = sequences.unfold(0, self.window_size, 1)

        # Create input-target pairs from windows
        x = windows[:, :-1]  # All but last character of each window
        y = windows[:, 1:]  # All but first character of each window

        return DataLoader(TensorDataset(x, y), shuffle=True, batch_size=self.batch_size)

    def is_same(self, model) -> bool:
        raise NotImplementedError

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType, compute_p_gen: bool,
                           max_failed_trials: int = 1000):
        import torch

        torch.manual_seed(seed)

        self._model.eval()
        input_vector = torch.as_tensor([self.letter_to_index[letter] for letter in self.prime_str],
                                       device=self.device).long()
        predicted = self.prime_str
        failed_trials = 0
        total_trials = 0

        with torch.no_grad():
            state = self._model.init_zero_state(batch_size=1)

            # Add sequence and batch dimensions for LSTM input
            for p in range(len(self.prime_str) - 1):
                inp = input_vector[p].unsqueeze(0).unsqueeze(0)
                _, state = self._model(inp, state)

            inp = input_vector[-1].unsqueeze(0).unsqueeze(0)
            gen_seq_count = 0

            while gen_seq_count < count and failed_trials < max_failed_trials:
                output, state = self._model(inp, state)

                scaled_logits = output[0, 0] / self.temperature
                scaled_logits = torch.clamp(scaled_logits, min=-100, max=100)
                output_dist = torch.nn.functional.softmax(scaled_logits, dim=0)

                try:
                    output_dist = output_dist + 1e-10
                    output_dist = output_dist / output_dist.sum()
                    top_i = torch.multinomial(output_dist, 1)[0].item()
                except RuntimeError as e:
                    logging.warning(f"{SimpleLSTM.__name__}: Error sampling from distribution: {e}; "
                                    f"Using argmax instead.")
                    logging.debug(f"{SimpleLSTM.__name__}: Distribution: {output_dist}\n"
                                  f"Sum: {output_dist.sum()}, Min: {output_dist.min()}, Max: {output_dist.max()}")
                    top_i = scaled_logits.argmax().item()

                predicted_char = self.index_to_letter[top_i]
                predicted += predicted_char
                inp = torch.as_tensor(self.letter_to_index[predicted_char], device=self.device).long()
                inp = inp.unsqueeze(0).unsqueeze(0)

                if predicted_char == "*":
                    last_seq = predicted.split('*')[-2]
                    if len(last_seq) > 0:
                        gen_seq_count += 1
                        print_log(f"Generated valid sequence {gen_seq_count}/{count}", True)

                    else:
                        failed_trials += 1
                        if failed_trials % 10 == 0:
                            print_log(f"Warning: LSTM model generated {failed_trials} empty sequences", True)

                total_trials += 1

        print_log(
            f"{SimpleLSTM.__name__} {self.name}: generated {gen_seq_count} sequences with {failed_trials} failed attempts.",
            True)

        dataset = self._export_dataset(predicted, path)
        return dataset

    def _export_dataset(self, predicted, path):
        sequences = [seq for seq in predicted.split('*') if len(seq) > 0]
        count = len(sequences)
        df = pd.DataFrame({get_sequence_field_name(self.region_type, self.sequence_type): sequences,
                           'locus': [self.locus.to_string() for _ in range(count)],
                           'gen_model_name': [self.name for _ in range(count)]})

        return SequenceDataset.build_from_partial_df(df, PathBuilder.build(path), 'synthetic_lstm_dataset',
                                                     {'gen_model_name': [self.name]}, {'gen_model_name': str})


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
                                 'sequence_type': self.sequence_type.name, 'locus': self.locus.name}})

        store_weights(self._model, model_path / 'state_dict.yaml')

        return Path(shutil.make_archive(str(path / 'trained_model'), 'zip', str(model_path))).absolute()
