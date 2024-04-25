import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch.optim
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy, one_hot
from torch.utils.data import DataLoader

from immuneML import Constants
from immuneML.data_model.bnp_util import write_yaml, read_yaml
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.ml_methods.pytorch_implementations.SimpleVAEGenerator import Encoder, Decoder, SimpleVAEGenerator, \
    vae_cdr3_loss
from immuneML.ml_methods.util.pytorch_util import store_weights
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.StringHelper import StringHelper


class SimpleVAE(GenerativeModel):
    """
    SimpleVAE is a generative model on sequence level that relies on variational autoencoder. This type of model was
    proposed by Davidsen et al. 2019, and this implementation is inspired by their original implementation available
    at https://github.com/matsengrp/vampire.

    References:

    Davidsen, K., Olson, B. J., DeWitt, W. S., III, Feng, J., Harkins, E., Bradley, P., & Matsen, F. A., IV. (2019).
    Deep generative models for T cell receptor protein sequences. eLife, 8, e46935. https://doi.org/10.7554/eLife.46935


    **Specification arguments:**

    - chain (str): which chain the sequence come from, e.g., TRB

    - beta (float): VAE hyperparameter that balanced the reconstruction loss and latent dimension regularization

    - latent_dim (int): latent dimension of the VAE

    - linear_nodes_count (int): in linear layers, how many nodes to use

    - num_epochs (int): how many epochs to use for training

    - batch_size (int): how many examples to consider at the same time

    - j_gene_embed_dim (int): dimension of J gene embedding

    - v_gene_embed_dim (int): dimension of V gene embedding

    - cdr3_embed_dim (int): dimension of the cdr3 embedding

    - pretrains (int): how many times to attempt pretraining to initialize the weights and use warm-up for the beta hyperparameter before the main training process

    - warmup_epochs (int): how many epochs to use for training where beta hyperparameter is linearly increased from 0 up to its max value; this is in addition to num_epochs set above

    - patience (int): number of epochs to wait before the training is stopped when the loss is not improving

    - iter_count_prob_estimation (int): how many iterations to use to estimate the log probability of the generated sequence (the more iterations, the better the estimated log probability)

    - vocab (list): which letters (amino acids) are allowed - this is automatically filled for new models (no need to set)

    - max_cdr3_len (int): what is the maximum cdr3 length - this is automatically filled for new models (no need to set)

    - unique_v_genes (list): list of allowed V genes (this will be automatically filled from the dataset if not provided here manually)

    - unique_j_genes (list): list of allowed J genes (this will be automatically filled from the dataset if not provided here manually)

    - device (str): name of the device where to train the model (e.g., cpu)


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_vae:
                    SimpleVAE:
                        chain: beta
                        beta: 0.75
                        latent_dim: 20
                        linear_nodes_count: 75
                        num_epochs: 5000
                        batch_size: 10000
                        j_gene_embed_dim: 13
                        v_gene_embed_dim: 30
                        cdr3_embed_dim: 21
                        pretrains: 10
                        warmup_epochs: 20
                        patience: 20
                        device: cpu

    """

    @classmethod
    def load_model(cls, path: Path):
        assert path.exists(), f"{cls.__name__}: {path} does not exist."

        model_overview_file = path / 'model_overview.yaml'
        state_dict_file = path / 'state_dict.yaml'

        for file in [model_overview_file, state_dict_file]:
            assert file.exists(), f"{cls.__name__}: {file} is not a file."

        model_overview = read_yaml(model_overview_file)
        vae = SimpleVAE(**{k: v for k, v in model_overview.items() if k != 'type'})
        vae.model = vae.make_new_model(state_dict_file)

        return vae

    def __init__(self, chain, beta, latent_dim, linear_nodes_count, num_epochs, batch_size, j_gene_embed_dim, pretrains,
                 v_gene_embed_dim, cdr3_embed_dim, warmup_epochs, patience, iter_count_prob_estimation, device,
                 vocab=None, max_cdr3_len=None, unique_v_genes=None, unique_j_genes=None, name: str = None):
        super().__init__(chain)
        self.sequence_type = SequenceType.AMINO_ACID
        self.iter_count_prob_estimation = iter_count_prob_estimation
        self.num_epochs = num_epochs
        self.pretrains = pretrains
        self.region_type = RegionType.IMGT_CDR3  # TODO: check if they use cdr3 or junction in the original paper
        self.vocab = vocab if vocab is not None else (
            sorted((EnvironmentSettings.get_sequence_alphabet(self.sequence_type) + [Constants.GAP_LETTER])))
        self.vocab_size = len(self.vocab)
        self.beta = beta
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.cdr3_embed_dim = cdr3_embed_dim
        self.latent_dim = latent_dim
        self.j_gene_embed_dim, self.v_gene_embed_dim = j_gene_embed_dim, v_gene_embed_dim
        self.linear_nodes_count = linear_nodes_count
        self.batch_size = batch_size
        self.device = device
        self.max_cdr3_len, self.unique_v_genes, self.unique_j_genes = max_cdr3_len, unique_v_genes, unique_j_genes
        self.name = name
        self.model = None

        # hard-coded in the original implementation
        self.v_gene_loss_weight = 0.8138
        self.j_gene_loss_weight = 0.1305

        self.loss_path = None

    def make_new_model(self, initial_values_path: Path = None):

        assert self.unique_v_genes is not None and self.unique_j_genes is not None, \
            f'{SimpleVAE.__name__}: cannot generate empty model since unique V and J genes are not set.'

        encoder = Encoder(self.vocab_size, self.cdr3_embed_dim, len(self.unique_v_genes), self.v_gene_embed_dim,
                          len(self.unique_j_genes), self.j_gene_embed_dim, self.latent_dim, self.max_cdr3_len,
                          self.linear_nodes_count)

        decoder = Decoder(self.latent_dim, self.linear_nodes_count, self.max_cdr3_len, self.vocab_size,
                          len(self.unique_v_genes), len(self.unique_j_genes))

        vae = SimpleVAEGenerator(encoder, decoder)

        if initial_values_path and initial_values_path.is_file():
            state_dict = read_yaml(filename=initial_values_path)
            state_dict = {k: torch.as_tensor(v, device=self.device) for k, v in state_dict.items()}
            vae.load_state_dict(state_dict)

        vae.to(self.device)

        return vae

    def fit(self, data, path: Path = None):

        data_loader = self.encode_dataset(data)
        # TODO: split the data to train and validation? or have external validation/test dataset

        pretrained_weights_path = self._pretrain(data_loader=data_loader, path=path)

        model = self.make_new_model(pretrained_weights_path)
        model.train()

        optimizer = torch.optim.Adam(model.parameters())
        losses = []
        epoch = 1
        loss_decreasing = True

        while epoch <= self.num_epochs and loss_decreasing:
            loss = self._train_for_epoch(data_loader, model, self.beta, optimizer)
            losses.append(loss.item())
            print_log(f"{SimpleVAE.__name__}: epoch: {epoch}, loss: {loss}.")

            if min(losses) == loss.item():
                store_weights(model, path / 'state_dict.yaml')

            if epoch > self.patience and all(x <= y for x, y in zip(losses[-self.patience:], losses[-self.patience:][1:])):
                loss_decreasing = False

            epoch += 1

        pd.DataFrame({'epoch': list(range(1, epoch)), 'loss': losses}).to_csv(str(path / 'training_losses.csv'),
                                                                              index=False)
        self.loss_path = path / 'training_losses.csv'

        self.model = self.make_new_model(path / 'state_dict.yaml')

    def _pretrain(self, data_loader, path: Path):

        pretrained_weights_path = PathBuilder.build(path) / 'pretrained_warmup_weights.yaml'

        for pretrain_index in range(self.pretrains):
            model = self.make_new_model()
            optimizer = torch.optim.Adam(model.parameters())
            beta = 0 if self.warmup_epochs > 0 else self.beta
            best_val_loss = np.inf

            for epoch in range(self.warmup_epochs + 1):
                loss = self._train_for_epoch(data_loader, model, beta, optimizer)
                val_loss = loss.item()
                beta = self._update_beta_on_epoch_end(epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    store_weights(model, pretrained_weights_path)

        return pretrained_weights_path

    def _train_for_epoch(self, data_loader, model, beta, optimizer):
        loss = None
        for batch in data_loader:
            cdr3_input, v_gene_input, j_gene_input = batch
            cdr3_output, v_gene_output, j_gene_output, z = model(cdr3_input, v_gene_input, j_gene_input)
            loss = (vae_cdr3_loss(cdr3_output, cdr3_input, self.max_cdr3_len, z[0], z[1], beta)
                    + cross_entropy(v_gene_output, v_gene_input.float()) * self.v_gene_loss_weight
                    + cross_entropy(j_gene_output, j_gene_input.float())) * self.j_gene_loss_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss

    def encode_dataset(self, dataset, batch_size=None, shuffle=True):
        data = dataset.get_attributes([self.sequence_type.value, 'v_call', 'j_call'], as_list=True)
        if self.unique_v_genes is None:
            self.unique_v_genes = sorted(list(set([el.split("*")[0] for el in data['v_call']])))
        if self.unique_j_genes is None:
            self.unique_j_genes = sorted(list(set([el.split("*")[0] for el in data['j_call']])))
        if self.max_cdr3_len is None:
            self.max_cdr3_len = max(len(el) for el in data[self.sequence_type.value])

        encoded_v_genes = one_hot(
            torch.as_tensor([self.unique_v_genes.index(v_gene.split("*")[0]) for v_gene in data['v_call']],
                            device=self.device),
            num_classes=len(self.unique_v_genes))
        encoded_j_genes = one_hot(
            torch.as_tensor([self.unique_j_genes.index(j_gene.split("*")[0]) for j_gene in data['j_call']],
                            device=self.device),
            num_classes=len(self.unique_j_genes))
        padded_encoded_cdr3s = one_hot(torch.as_tensor([
            [self.vocab.index(letter) for letter in
             StringHelper.pad_sequence_in_the_middle(seq, self.max_cdr3_len, Constants.GAP_LETTER)]
            for seq in data[self.sequence_type.value]], device=self.device), num_classes=self.vocab_size)

        pytorch_dataset = PyTorchSequenceDataset({'v_gene': encoded_v_genes, 'j_gene': encoded_j_genes,
                                                  'cdr3': padded_encoded_cdr3s})

        return DataLoader(pytorch_dataset, shuffle=shuffle, batch_size=batch_size if batch_size else self.batch_size)

    def is_same(self, model) -> bool:
        raise RuntimeError

    def _update_beta_on_epoch_end(self, epoch):
        new_beta = self.beta
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            new_beta *= epoch / self.warmup_epochs
            logging.info(f'{SimpleVAE.__name__}: epoch {epoch}: beta updated to {new_beta}.')
        return new_beta

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType, compute_p_gen: bool):
        torch.manual_seed(seed)
        self.model.eval()

        with torch.no_grad():
            z_sample = torch.as_tensor(np.random.normal(0, 1, size=(count, self.latent_dim)),
                                       device=self.device).float()
            sequences, v_genes, j_genes = self.model.decode(z_sample)

        seq_objs = [ReceptorSequence(**{
            self.sequence_type.value: ''.join([self.vocab[Categorical(letter).sample()] for letter in sequences[i]])
                                     .replace(Constants.GAP_LETTER, ''),
            'metadata': SequenceMetadata(v_call=self.unique_v_genes[Categorical(v_genes[i]).sample()],
                                         j_call=self.unique_j_genes[Categorical(j_genes[i]).sample()], chain=self.chain,
                                         region_type=self.region_type.name)
        }) for i in range(count)]

        for obj in seq_objs:
            log_prob = self.compute_p_gen({self.sequence_type.value: obj.get_attribute(self.sequence_type.value),
                                           'v_call': obj.metadata.v_call, 'j_call': obj.metadata.j_call},
                                          self.sequence_type)
            obj.metadata.custom_params = {'log_prob': log_prob}

        dataset = SequenceDataset.build_from_objects(seq_objs, count, path, 'synthetic_vae')

        return dataset

    def compute_p_gens(self, sequences, sequence_type: SequenceType) -> np.ndarray:
        pass

    def compute_p_gen(self, sequence: dict, sequence_type: SequenceType) -> float:
        with torch.no_grad():
            encoded_v_genes = one_hot(
                torch.as_tensor([self.unique_v_genes.index(sequence['v_call'].split("*")[0])]),
                num_classes=len(self.unique_v_genes))
            encoded_j_genes = one_hot(
                torch.as_tensor([self.unique_j_genes.index(sequence['j_call'].split("*")[0])]),
                num_classes=len(self.unique_j_genes))
            padded_encoded_cdr3s = one_hot(torch.as_tensor([
                [self.vocab.index(letter) for letter in
                 StringHelper.pad_sequence_in_the_middle(sequence[self.sequence_type.value], self.max_cdr3_len,
                                                         Constants.GAP_LETTER)]]), num_classes=self.vocab_size)

            log_prob_estimates = []

            for _ in range(self.iter_count_prob_estimation):
                z_mean, z_log_var = self.model.encode(padded_encoded_cdr3s, encoded_v_genes, encoded_j_genes)
                z_sd = (z_log_var / 2).exp()
                z_sample = torch.as_tensor(np.array([scipy.stats.norm.rvs(loc=z_mean, scale=z_sd)])).float()
                aa_probs, v_gene_probs, j_gene_probs = self.model.decode(z_sample)
                aa_probs, v_gene_probs, j_gene_probs = aa_probs.numpy(), v_gene_probs.numpy(), j_gene_probs.numpy()

                log_p_x_given_z = (np.sum(np.log(np.sum(aa_probs[0] * padded_encoded_cdr3s[0].numpy(), axis=1))) +
                                   np.log(np.sum(v_gene_probs[0] * encoded_v_genes[0].numpy())) +
                                   np.log(np.sum(j_gene_probs[0] * encoded_j_genes[0].numpy())))

                log_p_z = np.sum(scipy.stats.norm.logpdf(z_sample[0], 0, 1))
                log_q_z_given_x = np.sum(scipy.stats.norm.logpdf(z_sample[0], z_mean[0], z_sd[0]))

                log_imp_weight = log_p_z - log_q_z_given_x
                log_prob_estimates.append(float(log_p_x_given_z + log_imp_weight))

        return sum(log_prob_estimates) / self.iter_count_prob_estimation

    def can_compute_p_gens(self) -> bool:
        return True

    def can_generate_from_skewed_gene_models(self) -> bool:
        return False

    def generate_from_skewed_gene_models(self, v_genes: list, j_genes: list, seed: int, path: Path,
                                         sequence_type: SequenceType, batch_size: int, compute_p_gen: bool):
        raise RuntimeError

    def save_model(self, path: Path) -> Path:
        model_path = PathBuilder.build(path / 'model')

        skip_export_keys = ['model', 'loss_path', 'j_gene_loss_weight', 'v_gene_loss_weight', 'region_type',
                            'sequence_type', 'vocab_size']
        write_yaml(filename=model_path / 'model_overview.yaml',
                   yaml_dict={**{k: v for k, v in vars(self).items() if k not in skip_export_keys},
                              **{'type': self.__class__.__name__}})

        store_weights(self.model, model_path / 'state_dict.yaml')

        return Path(shutil.make_archive(str(path / 'trained_model'), 'zip', str(model_path))).absolute()


class PyTorchSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['cdr3'])

    def __getitem__(self, index):
        return self.data['cdr3'][index], self.data['v_gene'][index], self.data['j_gene'][index]

    def get_v_genes(self):
        return self.data['v_gene']

    def get_j_genes(self):
        return self.data['j_gene']
