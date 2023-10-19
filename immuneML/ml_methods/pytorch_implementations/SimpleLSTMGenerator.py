import torch
from torch import nn


class SimpleLSTMGenerator(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, output_size):
        super(SimpleLSTMGenerator, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = 1

        self.embed = nn.Embedding(num_embeddings=input_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=self.num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, features, hidden_and_cell_state):
        features = features.view(1, -1)

        embedded = self.embed(features)

        output, hidden_and_cell_state = self.lstm(embedded, hidden_and_cell_state)

        output.squeeze_(0)
        output = self.fc(output)
        return output, hidden_and_cell_state

    def init_zero_state(self):
        init_hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        init_cell = torch.zeros(self.num_layers, 1, self.hidden_size)
        return (init_hidden, init_cell)
