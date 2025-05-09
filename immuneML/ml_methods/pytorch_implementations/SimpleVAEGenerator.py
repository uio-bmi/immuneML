import torch
from torch import nn
from torch.nn.functional import cross_entropy, elu


class XavierLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class UniformLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)
        nn.init.uniform_(self.weight, a=-0.05, b=0.05)


class Encoder(nn.Module):

    def __init__(self, vocab_size, cdr3_embed_dim, n_v_genes, v_gene_embed_dim, n_j_genes, j_gene_embed_dim,
                 latent_dim, max_cdr3_len, linear_nodes_count):
        super().__init__()

        # params
        self.vocab_size = vocab_size
        self.cdr3_embed_dim = cdr3_embed_dim
        self.max_cdr3_len = max_cdr3_len

        # input layers
        self.cdr3_embedding = UniformLinear(vocab_size, cdr3_embed_dim)
        self.v_gene_embedding = XavierLinear(n_v_genes, v_gene_embed_dim)
        self.j_gene_embedding = XavierLinear(n_j_genes, j_gene_embed_dim)

        # encoding layers
        self.encoder_linear_layer_1 = XavierLinear(cdr3_embed_dim * max_cdr3_len + v_gene_embed_dim + j_gene_embed_dim,
                                                   linear_nodes_count)
        self.encoder_linear_layer_2 = XavierLinear(linear_nodes_count, linear_nodes_count)

        # latent layers
        self.z_mean = XavierLinear(linear_nodes_count, latent_dim)
        self.z_log_var = XavierLinear(linear_nodes_count, latent_dim)

    def forward(self, cdr3_input, v_gene_input, j_gene_input):
        # input processing
        cdr3_embedding = self.cdr3_embedding(cdr3_input.float())
        cdr3_embedding_flat = cdr3_embedding.view(-1, self.vocab_size * self.max_cdr3_len)
        v_gene_embedding = elu(self.v_gene_embedding(v_gene_input.float()))
        j_gene_embedding = elu(self.j_gene_embedding(j_gene_input.float()))

        # encoding
        merged_embedding = torch.cat([cdr3_embedding_flat, v_gene_embedding, j_gene_embedding], dim=1)
        encoder_linear_1 = elu(self.encoder_linear_layer_1(merged_embedding))
        encoder_linear_2 = elu(self.encoder_linear_layer_2(encoder_linear_1))

        # latent
        z_mean = self.z_mean(encoder_linear_2)
        z_log_var = self.z_log_var(encoder_linear_2)

        return z_mean, z_log_var


class Decoder(nn.Module):

    def __init__(self, latent_dim, linear_nodes_count, max_cdr3_len, vocab_size, n_v_genes, n_j_genes):
        super().__init__()

        # params
        self.latent_dim = latent_dim
        self.linear_nodes_count = linear_nodes_count
        self.max_cdr3_len = max_cdr3_len
        self.vocab_size = vocab_size

        # latent layers
        self.decoder_linear_1 = XavierLinear(latent_dim, linear_nodes_count)
        self.decoder_linear_2 = XavierLinear(linear_nodes_count, linear_nodes_count)

        # decoding layers
        self.cdr3_output = XavierLinear(linear_nodes_count, self.max_cdr3_len * self.vocab_size)
        self.v_gene_output = XavierLinear(linear_nodes_count, n_v_genes)
        self.j_gene_output = XavierLinear(linear_nodes_count, n_j_genes)

    def forward(self, z):
        # latent
        decoder_linear_1 = elu(self.decoder_linear_1(z))
        decoder_linear_2 = elu(self.decoder_linear_2(decoder_linear_1))

        # decoding
        cdr3_output = self.cdr3_output(decoder_linear_2).view(-1, self.max_cdr3_len, self.vocab_size)

        v_gene_output = self.v_gene_output(decoder_linear_2)
        j_gene_output = self.j_gene_output(decoder_linear_2)

        return cdr3_output, v_gene_output, j_gene_output


class SimpleVAEGenerator(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, cdr3_input, v_gene_input, j_gene_input):
        z_mean, z_log_var = self.encode(cdr3_input, v_gene_input, j_gene_input)

        # reparameterization trick
        epsilon = torch.randn(z_mean.size()).to(z_mean.device)
        z = z_mean + torch.exp(z_log_var / 2) * epsilon

        cdr3_output, v_gene_output, j_gene_output = self.decode(z)
        return cdr3_output, v_gene_output, j_gene_output, (z_mean, z_log_var)

    def decode(self, z):
        return self.decoder(z)

    def encode(self, cdr3_input, v_gene_input, j_gene_input):
        z_mean, z_log_var = self.encoder(cdr3_input, v_gene_input, j_gene_input)
        return z_mean, z_log_var

    def encoding_func(self, cdr3_input, v_gene_input, j_gene_input):
        z_mean, z_log_var = self.encoder(cdr3_input, v_gene_input, j_gene_input)
        epsilon = torch.randn(z_mean.size()).to(z_mean.device)
        return z_mean + torch.exp(z_log_var / 2) * epsilon


def vae_cdr3_loss(cdr3_output, cdr3_input, max_cdr3_len, z_mean, z_log_var, beta):
    # format: batch_size, sequence_length, vocab_size
    vocab_size = cdr3_output.size(-1)

    # Cross entropy: mean across all positions, then multiply by sequence length
    cdr3_input_modified = torch.argmax(cdr3_input, dim=-1)
    xent_loss = max_cdr3_len * cross_entropy(cdr3_output.view(-1, vocab_size),
                                             cdr3_input_modified.view(-1),
                                             reduction='mean')  # mean across all positions

    # KL loss: sum across latent dimension
    kl_loss = -0.5 * torch.sum(1 + z_log_var - torch.square(z_mean) - z_log_var.exp(), dim=-1) * beta

    # Sum the losses
    return xent_loss + torch.mean(kl_loss)  # mean of KL loss to match cross entropy scale
