import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

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

        self.afflines = []
        self.afflines.append(nn.Linear(in_dim, hidden_dim))
        for i in range(n_layers-2):
            self.afflines.append(nn.Linear(hidden_dim, hidden_dim))
        self.afflines.append(nn.Linear(hidden_dim, out_dim))
        self.afflines = nn.ModuleList(self.afflines)

        self.activations = activations()
        self.activate_final = activate_final
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = Dropout(self.dropout_rate)

    def forward(self, x):
        for i in range(len(self.afflines)):
            x = self.afflines[i](x)
            if i != len(self.afflines)-1 or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = self.dropout(x)
        return x
