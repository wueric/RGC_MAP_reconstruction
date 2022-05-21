import torch
import torch.nn as nn

import numpy as np

from convex_solver_base.optim_base import BatchParallelProxProblem

from typing import Tuple


class PixelwiseLinearL1Regression(BatchParallelProxProblem):

    CELL_COEFFS_IDX_ARGS = 0

    # shape (n_pixels, n_trials)
    IMAGES_KWARGS = 'images'

    # shape (n_cells, n_trials)
    SPIKES_KWARGS = 'spikes'

    def __init__(self,
                 im_shape: Tuple[int, int],
                 n_cells: int,
                 l1_sparsity : float):

        super().__init__()

        self.height, self.width = im_shape
        self.n_pixels = self.height * self.width
        self.n_cells = n_cells

        self.l1_sparsity = l1_sparsity

        # OPTIMIZATION VARIABLE
        self.coeffs_by_pixel = nn.Parameter(
            torch.empty((self.n_pixels, self.n_cells), dtype=torch.float32),
            requires_grad=True)
        nn.init.uniform_(self.coeffs_by_pixel, a=-1e-2, b=1e-2)

    def compute_test_loss(self,
                          test_spikes: torch.Tensor,
                          test_images_flat: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # shape (n_pixels, n_cells) @ (n_cells, n_trials)
            # -> (n_pixels, n_trials)
            reconstruction_flat = self.coeffs_by_pixel @ test_spikes

            # -> (n_pixels, n_trials)
            diff = test_images_flat - reconstruction_flat

            return torch.mean(diff * diff)

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:

        # shape (n_pixels, n_trials)
        images = kwargs[self.IMAGES_KWARGS]

        # shape (n_cells, n_trials)
        spikes = kwargs[self.SPIKES_KWARGS]

        # shape (n_pixels, n_cells)
        coeffs = args[self.CELL_COEFFS_IDX_ARGS]

        # shape (n_pixels, n_cells) @ (n_cells, n_trials)
        # -> (n_pixels, n_trials)
        reconstruction_flat = coeffs @ spikes

        # -> (n_pixels, n_trials)
        diff = images - reconstruction_flat

        # shape (n_pixels, )
        loss = torch.sum(diff * diff, dim=1)

        return loss

    def get_filter_coeffs(self) -> np.ndarray:
        return self.coeffs_by_pixel.detach().cpu().numpy()

    def get_filter_coeffs_imshape(self) -> np.ndarray:

        coeffs_np = self.get_filter_coeffs()
        coeffs_imshape = coeffs_np.transpose(1, 0).reshape(self.n_cells, self.height, self.width)
        return coeffs_imshape

    def _prox_proj(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:

        with torch.no_grad():
            coeffs= args[self.CELL_COEFFS_IDX_ARGS]

            prox_proj_filters = torch.clamp_min_(coeffs - self.l1_sparsity, 0.0) \
                                - torch.clamp_min_(-coeffs - self.l1_sparsity, 0.0)

            return (prox_proj_filters, )

    @property
    def n_problems(self) -> int:
        return self.n_pixels