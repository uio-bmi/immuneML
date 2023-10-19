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


class SimpleLSTM(GenerativeModel):

    PREVIEW_SEQ_COUNT = 10
    ITER_TO_REPORT = 100

    def __init__(self, chain, sequence_type: SequenceType, hidden_size, learning_rate, num_epochs, batch_size, embed_size):
        super().__init__(chain)
        self._model = None
        self.num_letters = len(EnvironmentSettings.get_sequence_alphabet(sequence_type)) + 1
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.max_length = 10
        self.sequence_type = sequence_type
        self.unique_letters = EnvironmentSettings.get_sequence_alphabet(self.sequence_type) + ["*"]
        self.letter_to_index = {letter: i for i, letter in enumerate(self.unique_letters)}
        self.index_to_letter = {i: letter for letter, i in self.letter_to_index.items()}

    def fit(self, data):
        data_loader = self._encode_dataset(data)

        model = SimpleLSTMGenerator(input_size=self.num_letters, hidden_size=self.hidden_size, output_size=self.num_letters, embed_size=self.embed_size)
        model.train()

        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), self.learning_rate)

        for epoch in range(self.num_epochs):
            loss = 0.
            state = model.init_zero_state()
            optimizer.zero_grad()

            for x_batch, y_batch in data_loader:

                outputs, state = model(x_batch, state)
                loss += criterion(outputs, y_batch)

            loss /= len(data_loader.dataset)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % SimpleLSTM.ITER_TO_REPORT == 0:
                print_log(f"{SimpleLSTM.__name__}: Epoch [{epoch+1}/{self.num_epochs}]: loss: {loss.item():.4f}", True)

        self._model = model

    def _encode_dataset(self, dataset):
        sequences = dataset.get_attribute(self.sequence_type.value)
        self.max_length = sequences.lengths.max() + 1

        sequences = list(chain.from_iterable([[self.letter_to_index[letter] for letter in seq] + [self.letter_to_index['*']] for seq in sequences.tolist()]))
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

            state = self._model.init_zero_state()

            for p in range(len(prime_str) - 1):
                _, state = self._model(input_vector[p], state)

            inp = input_vector[-1]
            gen_seq_count = 0
            temperature = 1.

            while gen_seq_count < count:
                output, state = self._model(inp, state)

                output_dist = output.data.view(-1).div(temperature).exp()
                top_i = torch.multinomial(output_dist, 1)[0].item()

                predicted_char = self.index_to_letter[top_i]
                predicted += predicted_char
                inp = torch.as_tensor(self.letter_to_index[predicted_char]).long()
                if predicted_char == "*":
                    gen_seq_count += 1

        generated_sequences = predicted.split("*")[:-1]
        pd.DataFrame({sequence_type.value: generated_sequences}).to_csv(str(path), sep='\t', index=False)

        print_log(f"{SimpleLSTM.__name__}: generated {count} sequences stored at {path}. "
                  f"Preview:\n{generated_sequences[:SimpleLSTM.PREVIEW_SEQ_COUNT]}", True)

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
        pass
