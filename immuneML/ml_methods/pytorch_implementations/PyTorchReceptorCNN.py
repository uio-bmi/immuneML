import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType


class PyTorchReceptorCNN(nn.Module):

    def __init__(self, kernel_count: int, kernel_size, positional_channels: int, sequence_type: SequenceType, background_probabilities, chain_names):
        super(PyTorchReceptorCNN, self).__init__()
        self.background_probabilities = background_probabilities
        self.threshold = 0.1
        self.pseudocount = 0.05
        self.in_channels = len(EnvironmentSettings.get_sequence_alphabet(sequence_type)) + positional_channels
        self.positional_channels = positional_channels
        self.max_information_gain = self.get_max_information_gain()
        self.chain_names = chain_names

        self.conv_chain_1 = [f"chain_1_kernel_{size}" for size in kernel_size]
        self.conv_chain_2 = [f"chain_2_kernel_{size}" for size in kernel_size]

        for size in kernel_size:
            # chain 1
            setattr(self, f"chain_1_kernel_{size}", nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_count, kernel_size=size,
                                                              bias=True))
            getattr(self, f"chain_1_kernel_{size}").weight.data. \
                normal_(0.0, np.sqrt(1 / np.prod(getattr(self, f"chain_1_kernel_{size}").weight.shape)))

            # chain 2
            setattr(self, f"chain_2_kernel_{size}", nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_count, kernel_size=size,
                                                              bias=True))
            getattr(self, f"chain_2_kernel_{size}").weight.data. \
                normal_(0.0, np.sqrt(1 / np.prod(getattr(self, f"chain_2_kernel_{size}").weight.shape)))

        self.fully_connected = nn.Linear(in_features=kernel_count * len(kernel_size) * 2, out_features=1, bias=True)
        self.fully_connected.weight.data.normal_(0.0, np.sqrt(1 / np.prod(self.fully_connected.weight.shape)))

    def get_max_information_gain(self):
        """
        Information gain corresponds to Kullback-Leibler divergence between the observed probability p of an option (e.g. amino acid) ond null
        (or background) probability q of the option:

        .. math::

            KL(p||q) = \\sum_n p_n \\, log_2 \\, \\frac{p_n}{q_n}
                = \\sum_n p_n \\, log_2 \\, p_n - \\sum_n p_n \\, log_2 \\, q_n
                = \\sum_n p_n \\, log_2 \\, p_n - log_2 \\, q_n

            log_2 \\, q_n < 0 (1)
            \\sum_n p_n \\, log_2 \\, p_n < 0  (2)

            (1) \\wedge (2) \\Rightarrow max(KL(p||q)) = - log_2 \\, q_n

        Returns:

            max information gain given background probabilities

        """
        if all(self.background_probabilities[i] == self.background_probabilities[0] for i in range(len(self.background_probabilities))):
            return - np.log2(self.background_probabilities[0])
        else:
            raise NotImplementedError("ReceptorCNN: non-uniform background probabilities are currently not supported.")

    def forward(self, x):

        # creates batch_size x kernel_count representation of chain 1 and chain 2 by applying kernels, followed by relu and global max pooling
        chain_1 = torch.cat([torch.max(relu(conv_kernel(x[:, 0])), dim=2)[0] for conv_kernel in
                             [getattr(self, name) for name in self.conv_chain_1]], dim=1)
        chain_2 = torch.cat([torch.max(relu(conv_kernel(x[:, 0])), dim=2)[0] for conv_kernel in
                             [getattr(self, name) for name in self.conv_chain_2]], dim=1)

        # creates a single representation of the receptor from chain representations
        receptor = torch.cat([chain_1, chain_2], dim=1).squeeze()

        # predict the class of the receptor through the final fully-connected layer based on the inferred representations
        predictions = self.fully_connected(receptor).squeeze()
        return predictions

    def rescale_weights_for_IGM(self):
        for name in self.conv_chain_1 + self.conv_chain_2:
            value = getattr(self, name)
            value.weight = self._rescale_chain(value.weight)

    def _rescale_chain(self, weight_parameter):
        # dimension for the content (without positional information):
        dim = self.in_channels - self.positional_channels

        # enforce non-negativity constraint
        weight_chain = relu(weight_parameter[:, :dim, :])

        # add pseudocount for positions where the total sum is below threshold
        weight_chain[torch.sum(weight_chain, dim=1, keepdim=True).expand_as(weight_chain) < self.threshold] += self.pseudocount

        # rescale chain weights to represent IGM
        weight_chain = weight_chain / torch.sum(weight_chain, dim=1, keepdim=True).expand_as(weight_chain) * self.max_information_gain

        # append positional kernel values to rescaled weights and convert back to parameter
        weight_parameter = nn.Parameter(torch.cat([weight_chain, weight_parameter[:, dim:, :]], dim=1))
        return weight_parameter
