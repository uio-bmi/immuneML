import torch
from torch import nn


class SimpleLSTMGenerator(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, output_size, batch_size, num_layers=1):
        super(SimpleLSTMGenerator, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.embed = nn.Embedding(num_embeddings=input_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=self.num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, features, hidden_and_cell_state):
        features = features.view(1, -1)

        embedded = self.embed(features)

        output, hidden_and_cell_state = self.lstm(embedded, hidden_and_cell_state)

        output = output.squeeze(0)
        output = self.fc(output)
        return output, hidden_and_cell_state

    def init_zero_state(self, batch_size=None):
        init_hidden = torch.zeros(self.num_layers, batch_size if batch_size else self.batch_size, self.hidden_size)
        init_cell = torch.zeros(self.num_layers, batch_size if batch_size else self.batch_size, self.hidden_size)
        return init_hidden, init_cell
