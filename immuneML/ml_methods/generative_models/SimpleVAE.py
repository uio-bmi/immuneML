from pathlib import Path

import numpy as np
import torch.optim
from torch.nn.functional import cross_entropy, one_hot
from torch.utils.data import DataLoader, TensorDataset

from immuneML import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.ml_methods.pytorch_implementations.SimpleVAEGenerator import Encoder, Decoder, SimpleVAEGenerator, \
    vae_cdr3_loss
from immuneML.util.StringHelper import StringHelper


class SimpleVAE(GenerativeModel):
    """
    SimpleVAE is a generative model on sequence level that relies on variational autoencoder. This type of model was
    proposed by Davidsen et al. 2019, and this implementation is inspired by their original implementation available
    at https://github.com/matsengrp/vampire.

    References:

    Davidsen, K., Olson, B. J., DeWitt, W. S., III, Feng, J., Harkins, E., Bradley, P., & Matsen, F. A., IV. (2019).
    Deep generative models for T cell receptor protein sequences. eLife, 8, e46935. https://doi.org/10.7554/eLife.46935

    Arguments:

    """

    def __init__(self, chain, beta, latent_dim, linear_nodes_count, num_epochs, batch_size, j_gene_embed_dim,
                 v_gene_embed_dim, cdr3_embed_dim):
        super().__init__(chain)
        self.sequence_type = SequenceType.AMINO_ACID
        self.num_epochs = num_epochs
        self.vocab_size = len(EnvironmentSettings.get_sequence_alphabet(self.sequence_type)) + 1  # includes gap
        self.vocab = sorted((EnvironmentSettings.get_sequence_alphabet(self.sequence_type) + [Constants.GAP_LETTER]))
        self.beta = beta
        self.cdr3_embed_dim = cdr3_embed_dim
        self.latent_dim = latent_dim
        self.j_gene_embed_dim, self.v_gene_embed_dim = j_gene_embed_dim, v_gene_embed_dim
        self.linear_nodes_count = linear_nodes_count
        self.batch_size = batch_size
        self.max_cdr3_len, self.unique_v_genes, self.unique_j_genes = None, None, None
        self.model = None

        # hard-coded in the original implementation
        self.v_gene_loss_weight = 0.8138
        self.j_gene_loss_weight = 0.1305

    def _make_empty_model(self):

        assert self.unique_v_genes is not None and self.unique_j_genes is not None, \
            f'{SimpleVAE.__name__}: cannot generate empty model since unique V and J genes are not set.'

        encoder = Encoder(self.vocab_size, self.cdr3_embed_dim, len(self.unique_v_genes), self.v_gene_embed_dim,
                          len(self.unique_j_genes), self.j_gene_embed_dim, self.latent_dim, self.max_cdr3_len,
                          self.linear_nodes_count)

        decoder = Decoder(self.latent_dim, self.linear_nodes_count, self.max_cdr3_len, self.vocab_size,
                          len(self.unique_v_genes), len(self.unique_j_genes))

        return SimpleVAEGenerator(encoder, decoder)

    def fit(self, data, path: Path = None):

        data_loader = self._encode_dataset(data)
        model = self._make_empty_model()

        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(self.num_epochs):
            for batch in data_loader:
                cdr3_input, v_gene_input, j_gene_input = batch
                cdr3_output, v_gene_output, j_gene_output, z = model(cdr3_input, v_gene_input, j_gene_input)
                loss = (vae_cdr3_loss(cdr3_output, cdr3_input, self.max_cdr3_len, z[0], z[1], self.beta)
                        + cross_entropy(v_gene_output, v_gene_input) * self.v_gene_loss_weight
                        + cross_entropy(j_gene_output, j_gene_input)) * self.j_gene_loss_weight

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model = model

    def _encode_dataset(self, dataset):
        data = dataset.get_attributes([self.sequence_type.value, 'v_call', 'j_call'], as_list=True)
        self.unique_v_genes = sorted(list(set([el.split("*")[0] for el in data['v_call']])))
        self.unique_j_genes = sorted(list(set([el.split("*")[0] for el in data['j_call']])))
        self.max_cdr3_len = max(len(el) for el in data[self.sequence_type.value])

        encoded_v_genes = one_hot(torch.as_tensor([self.unique_v_genes.index(v_gene.split("*")[0]) for v_gene in data['v_call']]), num_classes=len(self.unique_v_genes))
        encoded_j_genes = one_hot(torch.as_tensor([self.unique_j_genes.index(j_gene.split("*")[0]) for j_gene in data['j_call']]), num_classes=len(self.unique_j_genes))
        padded_encoded_cdr3s = one_hot(torch.as_tensor([
            [self.vocab.index(letter) for letter in StringHelper.pad_sequence_in_the_middle(seq, self.max_cdr3_len, Constants.GAP_LETTER)]
            for seq in data[self.sequence_type.value]]))

        pytorch_dataset = PyTorchSequenceDataset({'v_gene': encoded_v_genes, 'j_gene': encoded_j_genes,
                                                  'cdr3': padded_encoded_cdr3s})

        return DataLoader(pytorch_dataset, shuffle=True, batch_size=self.batch_size)

    def is_same(self, model) -> bool:
        pass

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType, compute_p_gen: bool):
        pass

    def compute_p_gens(self, sequences, sequence_type: SequenceType) -> np.ndarray:
        pass

    def compute_p_gen(self, sequence: dict, sequence_type: SequenceType) -> float:
        pass

    def can_compute_p_gens(self) -> bool:
        pass

    def can_generate_from_skewed_gene_models(self) -> bool:
        pass

    def generate_from_skewed_gene_models(self, v_genes: list, j_genes: list, seed: int, path: Path,
                                         sequence_type: SequenceType, batch_size: int, compute_p_gen: bool):
        pass

    def save_model(self, path: Path) -> Path:
        pass


class PyTorchSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data['cdr3'][index], self.data['v_gene'][index], self.data['j_gene'][index]
