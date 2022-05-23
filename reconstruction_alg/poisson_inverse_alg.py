import torch
import torch.nn as nn

import numpy as np

from optim.optim_base import BatchParallelUnconstrainedProblem
from optim.admm_optim import BatchParallel_HQS_Z_Problem, BatchParallel_HQS_X_Problem

from optim.unconstrained_optim import FistaSolverParams, batch_parallel_unconstrained_solve

from simple_priors.gaussian_prior import ConvPatch1FGaussianPrior

from typing import Tuple, List, Dict, Optional, Callable, Union


class BatchPoissonEncodingLoss(nn.Module):

    def __init__(self,
                 lnp_filters: np.ndarray,
                 lnp_biases: np.ndarray,
                 n_problems: int = 1,
                 dtype: torch.dtype = torch.float32):
        '''
        Calculates the Poisson encoding loss

        :param lnp_filters: shape (n_cells, height, width)
        :param lnp_biases: shape (n_cells, )
        :param n_problems:
        :param dtype:
        :return:
        '''

        super().__init__()

        self.n_problems = n_problems
        self.n_cells, self.height, self.width = lnp_filters.shape
        self.n_pixels = self.height * self.width

        lnp_filters_flat = lnp_filters.reshape(self.n_cells, -1)

        # shape (n_cells, n_pixels)
        self.register_buffer('lnp_filters',
                             torch.tensor(lnp_filters_flat, dtype=dtype))

        # shape (n_cells, )
        self.register_buffer('lnp_biases',
                             torch.tensor(lnp_biases, dtype=dtype))

    def calculate_loss(self,
                       batched_images: torch.Tensor,
                       batched_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param batched_images: shape (batch, height, width)
        :param batched_spikes: shape (batch, n_cells)
        :return:
        '''

        batched_images_flat = batched_images.reshape(-1, self.n_pixels)

        # shape (1, n_cells, n_pixels) @ (n_problems, n_pixels, 1) -> (n_problems, n_cells, 1)
        # -> (n_problems, n_cells)
        generator_signal = (self.lnp_filters[None, :, :] @ batched_images_flat[:, :, None]).squeeze(2) \
                           + self.lnp_biases[None, :]

        # shape (n_problems, n_cells)
        exp_gen_sig= torch.exp(generator_signal)

        # shape (n_problems, n_cells)
        spike_prod= generator_signal * batched_spikes

        # shape (n_problems, n_cells)
        loss_per_problem_per_cell = exp_gen_sig - spike_prod

        # shape (n_problems, )
        loss_per_problem = torch.sum(loss_per_problem_per_cell, dim=1)

        return loss_per_problem

    def forward(self,
                batched_images: torch.Tensor,
                batched_spikes: torch.Tensor) -> torch.Tensor:
        return self.calculate_loss(batched_images, batched_spikes)


class BatchPoissonProxProblem(BatchParallelUnconstrainedProblem, BatchParallel_HQS_X_Problem):

    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 lnp_filters: np.ndarray,
                 lnp_biases: np.ndarray,
                 batch_size: int,
                 prox_rho: float= 1.0,
                 dtype: torch.dtype = torch.float32,
                 prox_nearto: Optional[np.ndarray] = None):

        super().__init__()

        self.batch_size = batch_size
        _, self.height, self.width= lnp_filters.shape
        self.poisson_loss = BatchPoissonEncodingLoss(lnp_filters,
                                                     lnp_biases,
                                                     n_problems=batch_size,
                                                     dtype=dtype)

        if prox_nearto is None:
            self.register_buffer('z_const_tensor',
                                 torch.zeros((self.batch_size, self.height, self.width), dtype=dtype))
        else:
            self.register_buffer('z_const_tensor', torch.tensor(prox_nearto, dtype=dtype))

        self.rho = prox_rho

        #### OPTIMIZATION VARIABLES ##########################################################
        self.reconstructed_images = nn.Parameter(torch.empty((self.batch_size, self.height, self.width), dtype=dtype),
                                                 requires_grad=True)
        nn.init.uniform_(self.reconstructed_images, a=-0.1, b=0.1)

    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def reinitialize_variables(self, initialized_z_const: Optional[torch.Tensor] = None) -> None:
        # nn.init.normal_(self.z_const_tensor, mean=0.0, std=1.0)
        if initialized_z_const is not None:
            self.z_const_tensor.data[:] = initialized_z_const.data[:]
        else:
            self.z_const_tensor.data[:] = 0.0

        nn.init.normal_(self.reconstructed_images, mean=0.0, std=0.5)

    @property
    def n_problems(self) -> int:
        return self.batch_size

    def assign_z(self, prox_to: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = prox_to.data[:]

    def get_reconstructed_image(self) -> torch.Tensor:
        return self.reconstructed_images.detach().clone()

    def get_reconstructed_image_np(self) -> np.ndarray:
        return self.reconstructed_images.detach().cpu().numpy()

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:

        # shape (batch, height, width)
        image_imshape = args[self.IMAGE_IDX_ARGS]

        # shape (batch, n_cells)
        observed_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape (batch, )
        encoding_loss = self.poisson_loss(image_imshape, observed_spikes)

        # shape (batch, height, width)
        prox_diff = image_imshape - self.z_const_tensor

        # shape (batch, )
        prox_loss = 0.5 * self.rho * torch.sum(prox_diff * prox_diff, dim=(1, 2))

        return encoding_loss + prox_loss


