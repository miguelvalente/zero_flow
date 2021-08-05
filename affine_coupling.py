
import torch
from torch import nn
from nets import MLP
from transform import Transform
import einops
from torch.utils import checkpoint

class AffineCoupling(Transform):
    def __init__(self, input_dim, split_dim, hidden_dims, non_linearity, scale_fn_type='sigmoid', eps=1E-1, context_dim=0, event_dim=-1):
        super().__init__()
        self.event_dim = event_dim
        self.input_dim = input_dim
        self.split_dim = input_dim // 2
        self.context_dim = context_dim
        self.scale_fn_type = scale_fn_type
        out_dim = (self.input_dim - self.split_dim) * 2

        self.nn = MLP(self.split_dim, hidden_dims, out_dim, non_linearity)

        if self.scale_fn_type == 'exp':
            self.scale_fn = lambda x: torch.exp(x)
        elif self.scale_fn_type == 'sigmoid':
            self.scale_fn = lambda x: (2 * torch.sigmoid(x) - 1) * (1 - eps) + 1
        else:
            raise Exception('Invalid scale_fn_type')

    def forward(self, x, context=None):
        x2_size = self.split_dim
        x1, x2 = x.split([self.split_dim, x2_size], dim=self.event_dim)

        shift, scale = self.nn(x1).split([x2_size, x2_size], dim=-1)

        scale = self.scale_fn(scale)

        y1 = x1
        y2 = x2 * scale + shift

        ldj = torch.einsum('bn->b', torch.log(scale))
        return torch.cat([y1, y2], dim=self.event_dim), ldj

    def inverse(self, y, context=None):
        y2_size = self.split_dim
        y1, y2 = y.split([self.split_dim, y2_size], dim=self.event_dim)

        shift, scale = self.nn(y1).split([y2_size, y2_size], dim=-1)

        scale = self.scale_fn(scale)

        x1 = y1
        x2 = (y2 - shift) / scale

        return torch.cat([x1, x2], dim=self.event_dim)
