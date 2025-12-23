import torch
import torch.nn as nn
import torch.nn.functional as F


class C(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size

        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = torch.tanh(self.layer(x))
        return out
    
    @property
    def weights(self):
        return self.layer.weight
    
    @weights.setter
    def weights(self, new_weights):
        self.layer.weight = nn.Parameter(new_weights)

    @property
    def bias(self):
        return self.layer.bias
    
    @bias.setter
    def bias(self, new_bias):
        self.layer.bias = nn.Parameter(new_bias)