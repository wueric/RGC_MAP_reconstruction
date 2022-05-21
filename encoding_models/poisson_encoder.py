import torch
import torch.nn as nn

import numpy as np

from typing import Tuple, Union, Optional, List, Dict
from collections import namedtuple

from convex_solver_base.optim_base import BatchParallelUnconstrainedProblem, BatchParallelProxProblem

ScaledPoissonFittedParams = namedtuple('ScaledPoissonFittedParams', ['filter', 'bias', 'loss'])


class SharedStimMulticellScaleOnlyPoisson(BatchParallelUnconstrainedProblem):
    FILTER_SCALE_IDX_ARGS = 0
    BIAS_IDX_ARGS = 1

    STIMULUS_KWARGS = 'shared_stimulus'
    SPIKES_KWARGS = 'multicell_spikes'

    def __init__(self,
                 multicell_sta_prior_filters: Union[np.ndarray, torch.Tensor],
                 filter_init_range: Tuple[float, float] = (-1.0, 1.0),
                 dtype: torch.dtype = torch.float32):

        super().__init__()

        self.n_cells, self.n_pixels = multicell_sta_prior_filters.shape

        #### Constants ###########################################
        if isinstance(multicell_sta_prior_filters, np.ndarray):
            self.register_buffer('prior_filters',
                                 torch.tensor(multicell_sta_prior_filters, dtype=dtype))
        else:
            self.register_buffer('prior_filters', multicell_sta_prior_filters)

        #### Optimization variables ################################
        self.filter_scale = nn.Parameter(torch.empty((self.n_cells, 1), dtype=dtype),
                                         requires_grad=True)
        nn.init.uniform_(self.filter_scale, a=filter_init_range[0], b=filter_init_range[1])

        self.bias = nn.Parameter(torch.empty((self.n_cells, 1), dtype=dtype),
                                 requires_grad=True)
        nn.init.uniform_(self.bias, a=filter_init_range[0], b=filter_init_range[1])

    @property
    def n_problems(self) -> int:
        return self.n_cells

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:

        # shape (n_cells, 1)
        filter_scale = args[self.FILTER_SCALE_IDX_ARGS]

        # shape (n_cells, 1)
        bias = args[self.BIAS_IDX_ARGS]

        # shape (n_trials, n_pixels)
        stimuli = kwargs[self.STIMULUS_KWARGS]

        # shape (n_cells, n_trials)
        multicell_spikes = kwargs[self.SPIKES_KWARGS]

        # shape (n_cells, n_pixels)
        filters_scaled = filter_scale * self.prior_filters

        # shape (1, n_trials, 1, n_pixels) @ (n_cells, 1, n_pixels, 1) ->
        # (n_cells, n_trials, 1, 1) -> (n_cells, n_trials)
        encoded_inner_prod = (stimuli[None, :, None, :] @ filters_scaled[:, None, :, None]
                              + bias[:, None, :, None]).squeeze(3).squeeze(2)

        # shape (n_cells, n_trials)
        exp_gen_sig = torch.exp(encoded_inner_prod)

        # shape (n_cells, n_trials)
        spike_prod = encoded_inner_prod * multicell_spikes

        # shape (n_cells, n_trials)
        loss_per_trial = exp_gen_sig - spike_prod

        # shape (n_cells, )
        return torch.mean(loss_per_trial, dim=1)

    def get_filters_and_biases(self) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            # shape (n_cells, n_pixels)
            filters_scaled = self.prior_filters * self.filter_scale[:, :]
            return filters_scaled.detach().cpu().numpy(), self.bias.detach().cpu().numpy()


class SharedStimMulicellPoisson(BatchParallelProxProblem):
    FILTERS_IDX_ARGS = 0
    BIAS_IDX_ARGS = 1

    STIMULUS_KWARGS = 'shared_stimulus'
    SPIKES_KWARGS = 'multicell_spikes'

    def __init__(self,
                 sta_prior_filters: np.ndarray,
                 l1_weight: float,
                 l2_weight: float,
                 filter_init_range: Tuple[float, float] = (-1.0, 1.0),
                 initial_filter_values: Optional[np.ndarray] = None,
                 initial_bias_values: Optional[np.ndarray] = None):

        super().__init__()

        self.n_cells, self.n_pixels = sta_prior_filters.shape
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

        # constants
        self.register_buffer('prior_filter', torch.tensor(sta_prior_filters, dtype=torch.float32))

        # optimization variables
        if initial_filter_values is None:
            self.filters = nn.Parameter(torch.empty(sta_prior_filters.shape, dtype=torch.float32),
                                        requires_grad=True)
            nn.init.uniform_(self.filters, a=filter_init_range[0], b=filter_init_range[1])
        else:
            self.filters = nn.Parameter(torch.tensor(initial_filter_values, dtype=torch.float32),
                                        requires_grad=True)

        if initial_bias_values is None:
            self.bias = nn.Parameter(torch.empty((self.n_cells, 1), dtype=torch.float32),
                                     requires_grad=True)
            nn.init.uniform_(self.bias, a=filter_init_range[0], b=filter_init_range[1])
        else:
            self.bias = nn.Parameter(torch.tensor(initial_bias_values, dtype=torch.float32),
                                     requires_grad=True)

    @property
    def n_problems(self) -> int:
        return self.n_cells

    def _prox_proj(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:

        new_filters = args[self.FILTERS_IDX_ARGS]
        new_biases = args[self.BIAS_IDX_ARGS]

        with torch.no_grad():
            prox_proj_filters = torch.clamp_min_(new_filters - self.l1_weight, 0.0) \
                                - torch.clamp_min_(-new_filters - self.l1_weight, 0.0)
            return prox_proj_filters, new_biases.detach().clone()

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:

        stimuli = kwargs[self.STIMULUS_KWARGS]
        multicell_spikes = kwargs[self.SPIKES_KWARGS]

        filters = args[self.FILTERS_IDX_ARGS]
        biases = args[self.BIAS_IDX_ARGS]

        # shape (1, n_trials, 1, n_pixels) @ (n_cells, 1, n_pixels, 1) ->
        # (n_cells, n_trials, 1, 1)
        encoded_inner_prod = stimuli[None, :, None, :] @ filters[:, None, :, None] \
                             + biases[:, None, :, None]

        # shape (n_cells, n_trials)
        encoded_inner_prod = encoded_inner_prod.squeeze(3).squeeze(2)

        # shape (n_cells, n_trials)
        exp_gen_sig = torch.exp(encoded_inner_prod)

        # shape (n_cells, n_trials)
        spike_prod = encoded_inner_prod * multicell_spikes

        # shape (n_cells, n_trials)
        loss_per_trial = exp_gen_sig - spike_prod

        # shape (n_cells, )
        mean_encoding_loss = torch.mean(loss_per_trial, dim=1)

        # now we have to add the L2 loss term
        # shape (n_cells, n_pixels)
        diff_prior = filters - self.prior_filter

        # shape (n_cells)
        l2_penalty = 0.5 * self.l2_weight * torch.sum(diff_prior * diff_prior, dim=1)

        return mean_encoding_loss + l2_penalty

    def get_filters_and_biases(self) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            # shape (n_cells, n_pixels)
            return self.filters.detach().cpu().numpy(), self.bias.detach().cpu().numpy()


class MulticellPoissonLoss(nn.Module):

    def __init__(self,
                 spatial_filters: np.ndarray,
                 spatial_bias: np.ndarray):

        super().__init__()

        # shape (n_cells, n_pixels)
        self.register_buffer('spatial_filters', torch.tensor(spatial_filters, dtype=torch.float32))

        # shape (n_cells, 1)
        self.register_buffer('bias', torch.tensor(spatial_bias, dtype=torch.float32))

    def forward(self,
                stimulus_frame: torch.Tensor,
                observed_spikes: torch.Tensor):
        '''

        :param observed_spikes: shape (batch, n_cells)
        :param stimulus_frame: shape (batch, n_pixels)
        :return:
        '''

        # shape (1, n_cells, n_pixels) @ (batch, n_pixels, 1)
        # -> (batch, n_cells, 1) -> (batch, n_cells)
        spat_inner_product = (self.spatial_filters[None, :, :] @ stimulus_frame[:, :, None]).squeeze(2)
        spat_inner_product = spat_inner_product + self.bias.T

        # shape (batch, n_cells)
        exp_gen_sig = torch.exp(spat_inner_product)

        # shape (batch, n_cells)
        spike_prod = spat_inner_product * observed_spikes

        # shape (batch, n_cells)
        encoding_loss = exp_gen_sig - spike_prod

        # shape (batch, )
        encoding_loss_per_cell = torch.mean(encoding_loss, dim=0)

        return encoding_loss_per_cell


def reinflate_uncropped_poisson_model(
        models: Dict[str, List[ScaledPoissonFittedParams]],
        cell_type_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    filter_list, bias_list = [], []

    for cell_type in cell_type_order:
        fitted_models = models[cell_type]

        for fitted_model in fitted_models:
            filter_list.append(fitted_model.filter)
            bias_list.append(fitted_model.bias)

    return np.array(filter_list, dtype=np.float32), np.array(bias_list, dtype=np.float32)
