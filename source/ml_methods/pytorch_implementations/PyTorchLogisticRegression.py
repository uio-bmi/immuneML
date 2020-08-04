import torch


class PyTorchLogisticRegression(torch.nn.Module):

    def __init__(self, in_features: int, abundance_weight_init: float, use_batch_norm: bool):
        super().__init__()

        self.use_batch_norm = use_batch_norm
        self.batch_norm = torch.nn.BatchNorm1d(in_features)

        self.linear = torch.nn.Linear(in_features, 1, bias=True)
        self.linear.weight.data[-1] = abundance_weight_init

    def forward(self, x):
        if self.use_batch_norm:
            x = self.batch_norm(x)
        return self.linear(x).squeeze()
