"""
Various helper network modules
"""
import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """ a simple MLP"""

    def __init__(self, in_dim, sizes, out_dim, non_linearity):
        super().__init__()
        self.non_linearity = non_linearity
        self.in_layer = nn.Linear(in_dim, sizes[0])
        self.out_layer = nn.Linear(sizes[-1], out_dim)
        self.layers = nn.ModuleList([nn.Linear(sizes[index], sizes[index + 1]) for index in range(len(sizes) - 1)])

    def forward(self, x):
        x = self.non_linearity(self.in_layer(x))
        for index, layer in enumerate(self.layers):
            if ((index % 2) == 0):
                x = self.non_linearity(layer(x))
        x = self.out_layer(x)
        return x

class MLPR(nn.Module):
    def __init__(self, in_dim, sizes, out_dim, non_linearity, residual=True):
        super().__init__()
        self.in_dim = in_dim
        self.sizes = sizes
        self.out_dim = out_dim
        self.non_linearity = non_linearity
        self.residual = residual
        self.in_layer = nn.Linear(in_dim, self.sizes[0])
        self.out_layer = nn.Linear(self.sizes[-1], out_dim)
        self.layers = nn.ModuleList([nn.Linear(sizes[index], sizes[index + 1]) for index in range(len(sizes) - 1)])

    def forward(self, x):
        x = self.non_linearity(self.in_layer(x))

        for index, layer in enumerate(self.layers):
            if ((index % 2) == 0):
                residual = x
                x = self.non_linearity(layer(x))
            else:
                x = self.non_linearity(residual + layer(x))

        x = self.out_layer(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        output = self.logic(self.fc(x))
        return output


# class Classifier(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dims=False):
#         super().__init__()

#         if hidden_dims:
#             raise NotImplementedError
#         else:
#             self.classifier = nn.Linear(input_dim, output_dim)

#     def forward(self, input_data):
#         output = self.classifier(input_data)
#         return output

class GSModule(nn.Module):
    def __init__(self, vertices, out_dim):
        super(GSModule, self).__init__()
        self.individuals = nn.ModuleList()
        assert vertices.dim() == 2, 'invalid shape : {:}'.format(vertices.shape)
        self.out_dim = out_dim
        self.require_adj = False
        for i in range(vertices.shape[0]):
            layer = LinearModule(vertices[i], out_dim)
            self.individuals.append(layer)

    def forward(self, semantic_vec):
        responses = [indiv(semantic_vec) for indiv in self.individuals]
        global_semantic = sum(responses)
        return global_semantic

class LinearModule(nn.Module):
    def __init__(self, vertice, out_dim):
        super(LinearModule, self).__init__()
        self.register_buffer('vertice', vertice.clone())
        self.fc = nn.Linear(vertice.numel(), out_dim)

    def forward(self, semantic_vec):
        input_offsets = semantic_vec - self.vertice
        response = F.relu(self.fc(input_offsets))
        return response
