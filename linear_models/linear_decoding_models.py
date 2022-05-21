import torch.nn as nn
import torch

from typing import List, Dict


class ClosedFormLinearModel(nn.Module):

    def __init__(self, n_cells: int, height: int, width: int,
                 train_with_gradient_descent: bool = False):
        super(ClosedFormLinearModel, self).__init__()

        self.linear_filters = nn.Parameter(torch.empty((n_cells, height, width), dtype=torch.float32),
                                           requires_grad=train_with_gradient_descent)
        nn.init.uniform_(self.linear_filters, a=-0.01, b=0.01)

    def set_linear_filters(self, filters: torch.Tensor):
        self.linear_filters.data[:] = filters.data[:]

    def forward(self, batched_spikes: torch.Tensor) -> torch.Tensor:
        '''
        :param batched_spikes: shape (batch, n_cells)
        :return:
        '''
        return torch.einsum('bn,nhw->bhw', batched_spikes, self.linear_filters)

    def solve(self, all_spike_vectors: torch.Tensor, images: torch.Tensor) -> None:
        '''
        :param all_spike_vectors: shape (n_images, n_cells)
        :param images: shape (n_images, height, width)
        :return: None
        '''
        self.solve_l2reg(all_spike_vectors, images, 0.0)

    def solve_l2reg(self,
                    all_spike_vectors: torch.Tensor,
                    images: torch.Tensor,
                    lambda_l2: float) -> None:
        '''
        :param all_spike_vectors: shape (n_images, n_cells)
        :param images: shape (n_images, height, width)
        :param lambda_l2:
        :return:
        '''

        n_flashes, height, width = images.shape

        n_cells, height, width = self.linear_filters.shape

        x_t_y = all_spike_vectors.permute((1, 0)) @ images.reshape((n_flashes, -1))  # shape (n_cells, n_pixels)

        to_solve = all_spike_vectors.permute((1, 0)) @ all_spike_vectors  # shape (n_cells, n_cells)
        to_solve[range(n_cells), range(n_cells)] += lambda_l2

        beta_hat = torch.linalg.solve(to_solve, x_t_y)

        self.linear_filters[...] = beta_hat.reshape((n_cells, height, width))
