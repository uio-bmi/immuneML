import torch
from torch import nn


class SimpleLSTMGenerator(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, output_size, batch_size, num_layers=1, device: str = 'cpu'):
        super(SimpleLSTMGenerator, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device

        self.embed = nn.Embedding(num_embeddings=input_size, embedding_dim=embed_size)
        nn.init.normal_(self.embed.weight)

        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        self.fc = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, features, hidden_and_cell_state):
        # features shape: (batch_size, seq_len)
        features = features.transpose(0, 1)  # Convert to (seq_len, batch_size)
        embedded = self.embed(features)  # (seq_len, batch_size, embed_size)
        
        output, hidden_and_cell_state = self.lstm(embedded, hidden_and_cell_state)
        # output shape: (seq_len, batch_size, hidden_size)
        
        output = self.fc(output)  # (seq_len, batch_size, output_size)
        output = output.transpose(0, 1)  # Convert back to (batch_size, seq_len, output_size)
        return output, hidden_and_cell_state

    def init_zero_state(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        init_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        init_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        return init_hidden, init_cell
