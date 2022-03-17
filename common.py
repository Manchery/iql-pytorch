import torch.nn as nn
from typing import Callable, Optional

from torch.nn.modules.dropout import Dropout


class MLP(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        n_layers,
        activations: Callable = nn.ReLU,
        activate_final: int = False,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()

        self.affines = []
        self.affines.append(nn.Linear(in_dim, hidden_dim))
        for i in range(n_layers-2):
            self.affines.append(nn.Linear(hidden_dim, hidden_dim))
        self.affines.append(nn.Linear(hidden_dim, out_dim))
        self.affines = nn.ModuleList(self.affines)

        self.activations = activations()
        self.activate_final = activate_final
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = Dropout(self.dropout_rate)

    def forward(self, x):
        for i in range(len(self.affines)):
            x = self.affines[i](x)
            if i != len(self.affines)-1 or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = self.dropout(x)
        return x
