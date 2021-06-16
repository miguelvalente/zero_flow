import csv
import pickle
import torch


def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def mean_except_batch(x, num_dims=1):
    '''
    Averages all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_mean: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).mean(-1)

def reduce_mean_masked(x, mask, axis):
    x = x * mask.float()
    m = x.sum(axis=axis) / mask.sum(axis=axis).float()
    return m

def reduce_sum_masked(x, mask, axis):
    x = x * mask
    m = x.sum(axis=axis)
    return m

if __name__ == '__main__':
    x = torch.rand((2, 10, 10))
    mask = torch.ones(2, 10, 10)
    reduce_sum_masked(x, mask, axis=1)
