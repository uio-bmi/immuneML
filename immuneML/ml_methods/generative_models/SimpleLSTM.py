from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.ml_methods.pytorch_implementations.SimpleLSTMGenerator import SimpleLSTMGenerator
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder


class SimpleLSTM(GenerativeModel):
    """
    This is a simple generative model for receptor sequences based on LSTM.

    Similar models have been proposed in:

    Akbar, R. et al. (2022). In silico proof of principle of machine learning-based antibody design at unconstrained scale. mAbs, 14(1), 2031482. https://doi.org/10.1080/19420862.2022.2031482

    Saka, K. et al. (2021). Antibody design using LSTM based deep generative model from phage display library for affinity maturation. Scientific Reports, 11(1), Article 1. https://doi.org/10.1038/s41598-021-85274-7


    Arguments:

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
        raise NotImplementedError

    PREVIEW_SEQ_COUNT = 10
    ITER_TO_REPORT = 100

    def __init__(self, chain, sequence_type: SequenceType, hidden_size, learning_rate, num_epochs, batch_size, num_layers,
                 embed_size, temperature, name=None):
        super().__init__(chain)
        self._model = None
        self.num_letters = len(EnvironmentSettings.get_sequence_alphabet(sequence_type)) + 1
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.temperature = temperature
        self.sequence_type = sequence_type
        self.name = name
        self.unique_letters = EnvironmentSettings.get_sequence_alphabet(self.sequence_type) + ["*"]
        self.letter_to_index = {letter: i for i, letter in enumerate(self.unique_letters)}
        self.index_to_letter = {i: letter for letter, i in self.letter_to_index.items()}
        self.loss_summary_path = None

    def fit(self, data, path: Path = None):
        data_loader = self._encode_dataset(data)

        model = SimpleLSTMGenerator(input_size=self.num_letters, hidden_size=self.hidden_size,
                                    embed_size=self.embed_size,
                                    output_size=self.num_letters, batch_size=self.batch_size)
        model.train()

        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), self.learning_rate)
        loss_summary = {"loss": [], "epoch": []}

        for epoch in range(self.num_epochs):
            loss = 0.
            state = model.init_zero_state()
            optimizer.zero_grad()

            for x_batch, y_batch in data_loader:
                state = state[0][:, :x_batch.size(0), :], state[1][:, :x_batch.size(0), :]

                outputs, state = model(x_batch, state)
                loss += criterion(outputs, y_batch)

            loss /= len(data_loader.dataset)
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
            loss_summary['epoch'].append(epoch+1)
        return loss_summary

    def _encode_dataset(self, dataset):
        sequences = dataset.get_attribute(self.sequence_type.value)

        sequences = list(chain.from_iterable(
            [[self.letter_to_index[letter] for letter in seq] + [self.letter_to_index['*']] for seq in
             sequences.tolist()]))
        sequences = torch.as_tensor(sequences).long()

        return DataLoader(TensorDataset(sequences[:-1], sequences[1:]), shuffle=True, batch_size=self.batch_size)

    def is_same(self, model) -> bool:
        raise NotImplementedError

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType, compute_p_gen: bool):
        torch.manual_seed(seed)

        self._model.eval()
        prime_str = "CAS"
        input_vector = torch.as_tensor([self.letter_to_index[letter] for letter in prime_str]).long()
        predicted = prime_str

        with torch.no_grad():

            state = self._model.init_zero_state(batch_size=1)

            for p in range(len(prime_str) - 1):
                _, state = self._model(input_vector[p], state)

            inp = input_vector[-1]
            gen_seq_count = 0
            probability_scores = [1.]

            while gen_seq_count <= count:
                output, state = self._model(inp, state)

                output_dist = output.data.view(-1).div(self.temperature).exp()
                top_i = torch.multinomial(output_dist, 1)[0].item()
                probability_scores[-1] *= torch.nn.functional.softmax(output, dim=-1).flatten()[top_i].item()

                predicted_char = self.index_to_letter[top_i]
                predicted += predicted_char
                inp = torch.as_tensor(self.letter_to_index[predicted_char]).long()
                if predicted_char == "*":
                    gen_seq_count += 1
                    probability_scores.append(1.)

        generated_sequences = pd.DataFrame({sequence_type.value: predicted.split("*")[1:-1],
                                            'log_probability': np.log(probability_scores[1:-1])})
        generated_sequences.to_csv(str(path), sep='\t', index=False)

        print_log(f"{SimpleLSTM.__name__} {self.name}: generated {count} sequences stored at {path}.", True)
        print_log(f"Preview:\n{generated_sequences.head(SimpleLSTM.PREVIEW_SEQ_COUNT)}", True)

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
        raise NotImplementedError
