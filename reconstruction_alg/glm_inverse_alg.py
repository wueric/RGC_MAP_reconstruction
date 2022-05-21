import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import numpy as np

from typing import Dict, List, Tuple, Callable, Union, Optional, Any

from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from reconstruction_alg.hqs_alg import HQS_X_Problem, BatchParallel_HQS_X_Problem
from convex_solver_base.optim_base import SingleUnconstrainedProblem, BatchParallelUnconstrainedProblem

from simple_priors.gaussian_prior import Full1FGaussianPrior, ConvPatch1FGaussianPrior

from dataclasses import dataclass


# we need an intermediate representation of ALL of the GLMs
# where we can just do either matrix multiplication or selection
# to grab the neighbors

#### some order-of-magnitude stuff ###############################################
# 700 cells, 250 sample filters, each cell needs its own set of coupling filters
# so naively this is 700 x 700 x 250 which is ~500 MB where the vast majority of
# entries is zero

# each cell needs its own feedback filter, 700 x 250 which is < 1 MB

# each cell has its own spatial filter, 128 x 80 x 700 which is ~30 MB

# so the biggest computational load is going to be dealing with the coupling filters
# We know that the coupling filters are going to be sparse, since each cell only
# couples to the nearby cells

# alternatively, we can densify the filter matrix
# the filter matrix becomes 700 x 250 x (max num coupled cells ~ 30) -> ~20 MB
# and use torch.take_along_dim to grab the relevant spike trains

def make_cosine_bump_family(a: float,
                            c: float,
                            n_basis: int,
                            t_timesteps: np.ndarray) -> np.ndarray:
    log_term = a * np.log(t_timesteps + c)  # shape (n_timesteps, )
    phases = np.r_[0.0:n_basis * np.pi / 2:np.pi / 2]  # shape (n_basis, )
    log_term_with_phases = -phases[:, None] + log_term[None, :]  # shape (n_basis, n_timesteps)

    should_keep = np.logical_and(log_term_with_phases >= -np.pi,
                                 log_term_with_phases <= np.pi)

    cosine_all = 0.5 * np.cos(log_term_with_phases) + 0.5
    cosine_all[~should_keep] = 0.0

    return cosine_all


def batch_bernoulli_spiking_neg_ll_loss(generator_sig: torch.Tensor,
                                        spike_vector: torch.Tensor) -> torch.Tensor:
    '''

    :param generator_sig: shape (batch, n_bins)
    :param spike_vector: shape (batch, n_bins)
    :return:
    '''

    prod = generator_sig * spike_vector
    log_sum_exp_term = torch.log(1.0 + torch.exp(generator_sig))
    return torch.mean(torch.sum(log_sum_exp_term - prod, dim=1), dim=0)


def batch_poisson_spiking_neg_ll_loss(generator_sig: torch.Tensor,
                                      spike_vector: torch.Tensor) -> torch.Tensor:
    '''

    :param generator_sig: shape (batch, n_bins)
    :param spike_vector: shape (batch, n_bins)
    :return:
    '''

    prod = generator_sig * spike_vector
    return torch.mean(torch.exp(generator_sig) - prod, dim=(0, 1))


def make_batch_binomial_spiking_neg_ll_loss(binom_max: int) \
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

    def batch_binomial_spiking_neg_ll_loss(generator_sig: torch.Tensor,
                                           spike_vector: torch.Tensor) -> torch.Tensor:
        '''

        :param generator_sig: shape (batch, n_bins)
        :param spike_vector: shape (batch, n_bins)
        :return:
        '''

        prod = generator_sig * spike_vector

        log_term = binom_max * torch.log(1.0 + torch.exp(generator_sig))

        return torch.mean(torch.sum(log_term - prod, dim=1), dim=0)

    return batch_binomial_spiking_neg_ll_loss


@dataclass
class FittedGLM:
    cell_id: int
    spatial_weights: np.ndarray
    spatial_bias: np.ndarray

    timecourse_weights: np.ndarray

    feedback_weights: np.ndarray

    coupling_cells_weights: Tuple[np.ndarray, np.ndarray]

    fitting_params: Dict[str, Any]

    train_loss: float
    test_loss: float


@dataclass
class FittedGLMFamily:
    fitted_models: Dict[int, FittedGLM]

    spatial_basis: Union[np.ndarray, None]
    timecourse_basis: np.ndarray
    feedback_basis: np.ndarray
    coupling_basis: np.ndarray


def _count_max_coupled_cells(fit_glms: Dict[str, FittedGLMFamily]) -> int:
    '''
    Counts the maximum number of coupled cells over every cell that has
        been fit (traverses over the cell types)

    :param fit_glms: fitted GLM families for each cell type, str key is cell type
    :return: int, maximum number of coupled cells for ANY model that has been fit
    '''

    max_coupled_cells = 0
    for ct, glm_family in fit_glms.items():
        for cell_id, fitted_glm in glm_family.fitted_models.items():
            max_coupled_cells = max(max_coupled_cells,
                                    fitted_glm.coupling_cells_weights[0].shape[0])

    return max_coupled_cells


@dataclass
class FullResolutionCompactModel:
    '''
    Used as an intermediate representation of the paramters of a single
        cell GLM for spot-checking the model

    Note that the model here is fitted on the full resolution stimulus
        with no spatial basis
    '''

    spatial_filter: np.ndarray  # shape (height, width), same shape as the stimulus
    timecourse_filter: np.ndarray  # shape (n_bins, )
    feedback_filter: np.ndarray  # shape (n_bins, )

    coupling_params: Tuple[np.ndarray, np.ndarray]
    # first array is the cell id, integer-valued, shape (n_cells, )
    # array has shape (n_cells, n_bins) which are the coupling filters
    # for the respective cells

    bias: np.ndarray


def _extract_full_res_glm_params(glm_family: FittedGLMFamily,
                                 cell_id: int) -> FullResolutionCompactModel:

    fitted_glm = glm_family.fitted_models[cell_id]

    # shape (n_coupling_basis, n_bins_filter)
    coupling_basis = glm_family.coupling_basis

    # shape (n_feedback_basis, n_bins_filter)
    feedback_basis = glm_family.feedback_basis

    # shape (n_timecourse_basis, n_bins_filter)
    timecourse_basis = glm_family.timecourse_basis

    # coupling_weights shape (n_coupled_cells, n_coupling_basis)
    # coupling_ids shape (n_coupled_cells, ); these are reference dataset cell ids
    coupling_weights, coupling_ids = fitted_glm.coupling_cells_weights

    # (n_coupled_cells, n_coupling_basis) @ (n_coupling_basis, n_bins_filter)
    # -> (n_coupled_cells, n_bins_filter)
    coupling_filters_cell = coupling_weights @ coupling_basis

    # shape (1, n_basis_stim_time)
    timecourse_weights = fitted_glm.timecourse_weights

    # shape (1, n_basis_stim_time) @ (n_timecourse_basis, n_bins_filter)
    # -> (1, n_bins_filter) -> (n_bins_filter, )
    timecourse_filter = (timecourse_weights @ timecourse_basis).squeeze(0)

    # shape (1, n_basis_feedback)
    feedback_weights = fitted_glm.feedback_weights

    # shape (1, n_basis_feedback) @ (n_basis_feedback, n_bins_filter)
    # -> (1, n_bins_filter) -> (n_bins_filter, )
    feedback_filter = (feedback_weights @ feedback_basis).squeeze(0)

    # shape (height, width)
    full_spatial_filter = fitted_glm.spatial_weights

    bias = fitted_glm.spatial_bias

    return FullResolutionCompactModel(
        full_spatial_filter,
        timecourse_filter,
        feedback_filter,
        (coupling_ids, coupling_filters_cell),
        bias
    )


@dataclass
class PackedGLMTensors:
    spatial_filters: np.ndarray  # shape (n_cells, height, width)
    timecourse_filters: np.ndarray  # shape (n_cells, n_bins_filter)
    feedback_filters: np.ndarray  # shape (n_cells, n_bins_filter)
    coupling_filters: np.ndarray  # shape (n_cells, max_coupled_cells, n_bins_filter)
    coupling_indices: np.ndarray  # shape (n_cells, max_coupled_cells)
    bias: np.ndarray  # shape (n_cells, )


def _compute_raw_coupling_indices(cell_ordering: OrderedMatchedCellsStruct,
                                  coupling_params: Tuple[np.ndarray, np.ndarray]) \
        -> np.ndarray:
    '''

    :param cell_ordering:
    :param coupling_params:
    :return:
    '''

    coupling_cell_ids = coupling_params[0]
    coupling_indices = np.zeros_like(coupling_cell_ids, dtype=np.int64)
    for ix in range(coupling_cell_ids.shape[0]):
        coupling_indices[ix] = cell_ordering.get_concat_idx_for_cell_id(coupling_cell_ids[ix])
    return coupling_indices


def _build_padded_coupling_sel_and_filters(
        raw_coupling_filters: np.ndarray,
        raw_coupling_indices: np.ndarray,
        max_coupled_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Builds the coupling selector matrix and the coupling filter matrix

    Note that because all of the filters that correspond to unused slots
        are set to zero, we don't care about what the selection indices are
        for unused slots, since we multiply those (randomly-selected) spikes by
        zero anyway.

    :param raw_coupling_filters:
    :param raw_coupling_indices:
    :param max_coupled_cells:
    :return:
    '''

    n_coupling_filters, n_bins = raw_coupling_filters.shape

    padded_coupling_sel = np.zeros((max_coupled_cells,), dtype=np.int64)
    padded_coupling_sel[:n_coupling_filters] = raw_coupling_indices

    padded_coupling_filters = np.zeros((max_coupled_cells, n_bins), dtype=np.float32)
    padded_coupling_filters[:n_coupling_filters, :] = raw_coupling_filters

    return padded_coupling_sel, padded_coupling_filters


def convert_glm_family_to_np(glm_family: FittedGLMFamily,
                             spat_shape: Optional[Tuple[int, int]] = None) -> FittedGLMFamily:
    '''
    Helper function; converts glm_family components to np.ndarray from torch.Tensor on GPU
        if necessary (this is necessary due to a bug in the full res fitting code, and because
        refitting GLMs is a multi-day affair)

    Rebuilds the object no matter what, even if no conversions are necessary
    :param glm_family:
    :return:
    '''

    converted_params_dict = {}
    for cell_id, fg in glm_family.fitted_models.items():
        spat_w = fg.spatial_weights.detach().cpu().numpy() if isinstance(fg.spatial_weights,
                                                                         torch.Tensor) else fg.spatial_weights
        if spat_shape is not None:
            spat_w = spat_w.reshape(spat_shape)

        spat_b = fg.spatial_bias.detach().cpu().numpy() if isinstance(fg.spatial_bias,
                                                                      torch.Tensor) else fg.spatial_bias

        time_w = fg.timecourse_weights.detach().cpu().numpy() if isinstance(fg.timecourse_weights,
                                                                            torch.Tensor) else fg.timecourse_weights

        feedback_w = fg.feedback_weights.detach().cpu().numpy() if isinstance(fg.feedback_weights,
                                                                              torch.Tensor) else fg.feedback_weights

        coupling_w0 = fg.coupling_cells_weights[0].detach().cpu().numpy() if isinstance(fg.coupling_cells_weights[0],
                                                                                        torch.Tensor) else \
        fg.coupling_cells_weights[0]
        coupling_w1 = fg.coupling_cells_weights[1].detach().cpu().numpy() if isinstance(fg.coupling_cells_weights[1],
                                                                                        torch.Tensor) else \
        fg.coupling_cells_weights[1]

        converted_params_dict[cell_id] = FittedGLM(cell_id,
                                                   spat_w, spat_b, time_w, feedback_w, (coupling_w0, coupling_w1),
                                                   fg.fitting_params, fg.train_loss, fg.test_loss)

    # spatial_basis may be an np.ndarray, None, or torch.Tensor
    # convert to np.ndarray if torch.Tensor, otherwise pass through
    spatial_basis = glm_family.spatial_basis.detach().cpu().numpy() if isinstance(glm_family.spatial_basis,
                                                                                  torch.Tensor) else glm_family.spatial_basis
    timecourse_basis = glm_family.timecourse_basis.detach().cpu().numpy() if isinstance(glm_family.timecourse_basis,
                                                                                        torch.Tensor) else glm_family.timecourse_basis
    feedback_basis = glm_family.feedback_basis.detach().cpu().numpy() if isinstance(glm_family.feedback_basis,
                                                                                    torch.Tensor) else glm_family.feedback_basis
    coupling_basis = glm_family.coupling_basis.detach().cpu().numpy() if isinstance(glm_family.coupling_basis,
                                                                                    torch.Tensor) else glm_family.coupling_basis
    
    del glm_family
    
    return FittedGLMFamily(
        converted_params_dict,
        spatial_basis,
        timecourse_basis,
        feedback_basis,
        coupling_basis
    )


def make_full_res_packed_glm_tensors(ordered_cells: OrderedMatchedCellsStruct,
                                      fit_glms: Dict[str, FittedGLMFamily]) \
        -> PackedGLMTensors:
    max_coupled_cells = _count_max_coupled_cells(fit_glms)

    cell_type_ordering = ordered_cells.get_cell_types()

    idx_sel_list = []  # type: List[np.ndarray]
    coupling_filters_list = []  # type: List[np.ndarray]
    timecourse_filters_list = []  # type: List[np.ndarray]
    feedback_filters_list = []  # type: List[np.ndarray]
    spatial_filters_list = []  # type: List[np.ndarray]
    bias_list = []  # type: List[np.ndarray]

    for cell_type in cell_type_ordering:

        glm_family = fit_glms[cell_type]
        glm_family = convert_glm_family_to_np(glm_family)

        for idx, cell_id in enumerate(ordered_cells.get_reference_cell_order(cell_type)):

            compact_model = _extract_full_res_glm_params(glm_family, cell_id)

            bias = compact_model.bias

            raw_coupling_indices = _compute_raw_coupling_indices(ordered_cells,
                                                                 compact_model.coupling_params)

            coupling_idx_padded, coupling_filters_padded = _build_padded_coupling_sel_and_filters(
                compact_model.coupling_params[1],
                raw_coupling_indices,
                max_coupled_cells
            )

            idx_sel_list.append(coupling_idx_padded)
            coupling_filters_list.append(coupling_filters_padded)
            timecourse_filters_list.append(compact_model.timecourse_filter)
            feedback_filters_list.append(compact_model.feedback_filter)
            spatial_filters_list.append(compact_model.spatial_filter)
            bias_list.append(bias)

    # now stack all of the arrays together
    spatial_filters_stacked = np.stack(spatial_filters_list, axis=0)
    idx_sel_stacked = np.stack(idx_sel_list, axis=0)
    coupling_filters_stacked = np.stack(coupling_filters_list, axis=0)
    timecourse_filters_stacked = np.stack(timecourse_filters_list, axis=0)
    feedback_filters_stacked = np.stack(feedback_filters_list, axis=0)
    bias_stacked = np.stack(bias_list, axis=0)

    packed_glm_tensors = PackedGLMTensors(spatial_filters_stacked,
                                          timecourse_filters_stacked,
                                          feedback_filters_stacked,
                                          coupling_filters_stacked,
                                          idx_sel_stacked,
                                          bias_stacked)

    return packed_glm_tensors


def per_bin_bernoulli_neg_log_likelihood(generator_signal: torch.Tensor,
                                         observed_spikes: torch.Tensor) -> torch.Tensor:
    '''

    Per-bin Bernoulli negative log-likelihood loss with sigmoid nonlinearity is

    \log (1 + \exp g[n]) - y[n] g[n] = g[n] \log (\exp -g[n] + 1) - y[n] g[n]

    :param generator_signal: shape (n_cells, n_bins)
    :param observed_spikes: shape (n_cells, n_bins)
    :return: per bin negative log-likelihood, shape (n_bins, )
    '''

    prod = generator_signal * observed_spikes
    log_sum_exp_term = torch.log(torch.exp(generator_signal) + 1)

    per_cell_loss_per_bin = log_sum_exp_term - prod

    # sum the log-likelihood over the cells, leave the bin dimension alone
    loss_per_bin = torch.sum(per_cell_loss_per_bin, dim=0)

    return loss_per_bin


def batch_per_bin_bernoulli_neg_log_likelihood(generator_signal: torch.Tensor,
                                               observed_spikes: torch.Tensor) -> torch.Tensor:
    '''

    :param generator_signal: shape (batch, n_cells, n_bins)
    :param observed_spikes: shape (batch, n_cells, n_bins)
    :return: shape (batch, n_bins)
    '''
    prod = generator_signal * observed_spikes
    log_sum_exp_term = torch.log(torch.exp(generator_signal) + 1)

    per_cell_loss_per_bin = log_sum_exp_term - prod

    # sum the log-likelihood over the cells, leave the bin dimension alone
    loss_per_bin = torch.sum(per_cell_loss_per_bin, dim=1)

    return loss_per_bin



def batch_bernoulli_spike_generation(gensig: torch.Tensor) -> torch.Tensor:
    '''
    Simulates Bernoulli spiking process for a single bin with sigmoid
        nonlinearity

    (can only do a single bin because GLM has feedback filter, so future
     probabilities depend on what happens in the present)

    :param spike_rate: shape (batch, n_repeats)
    :return: shape (batch, n_repeats), 0 or 1-valued tensor
    '''

    return torch.bernoulli(torch.sigmoid(gensig))


class TrialGLMSeparableFullSim(nn.Module):

    def __init__(self,
                 stacked_spatial_filters: np.ndarray,
                 stacked_timecourse_filters: np.ndarray,
                 stacked_feedback_filters: np.ndarray,
                 stacked_coupling_filters: np.ndarray,
                 coupling_idx_sel: np.ndarray,
                 stacked_bias: np.ndarray,
                 spike_generation_fn: Callable[[torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):
        '''

        :param stacked_spatial_filters: shape (n_cells, height, width); spatial filters, one per cell/GLM
        :param stacked_timecourse_filters: shape (n_cells, n_bins_filter); timecourse filters, one per cell/GLM
        :param stacked_feedback_filters: shape (n_cells, n_bins_filter); feedback filters, one per cell/GLM
        :param stacked_coupling_filters: shape (n_cells, max_coupled_cells, n_bins_filter); coupling_filters,
            may contain unused slots since each cell may be coupled to a different number of other cells
        :param coupling_idx_sel: shape (n_cells, max_coupled_cells); positions marked 0 correspond to unused slots, the
            first valid cell corresponds to index 1
        :param stacked_bias: shape (n_cells, 1)
        :param n_bins_reconstruction: int, number of frames that we want to reconstruct; each reconstructed frame
            should correspond to a single timebin

            n_bins_reconstruction should satisfy
                n_bins_reconstruction + n_bins_filter = n_bins_spikes

        :param pre_flash_stimulus: shape (n_bins_filter, n_pixels), the exact stimulus for the time period directly
            preceding the period that we are trying to reconstruct; used to set the initial conditions of the GLM

            For the flashed trials, this should be a matrix of zeros.

        '''

        super().__init__()

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_bins_filter = stacked_timecourse_filters.shape[1]
        self.max_coupled_cells = stacked_coupling_filters.shape[1]
        self.n_pixels = self.height * self.width

        self.dtype = dtype
        self.spike_generation_fn = spike_generation_fn

        stacked_flat_spat_filters = stacked_spatial_filters.reshape(self.n_cells, -1)

        ##### GLM parameters as torch buffers #############################################
        # shape (n_cells, n_pixels)
        self.register_buffer('stacked_flat_spat_filters', torch.tensor(stacked_flat_spat_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_timecourse_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_timecourse_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_timecourse_filters', torch.tensor(stacked_timecourse_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_feedback_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_feedback_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_feedback_filters', torch.tensor(stacked_feedback_filters, dtype=dtype))

        # shape (n_cells, max_coupled_cells, n_bins_filter)
        assert stacked_coupling_filters.shape == (self.n_cells, self.max_coupled_cells, self.n_bins_filter), \
            f'stacked_coupling_filters must have shape {(self.n_cells, self.max_coupled_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_coupling_filters', torch.tensor(stacked_coupling_filters, dtype=dtype))

        # shape (n_cells, 1)
        assert stacked_bias.shape == (self.n_cells, 1), f'stacked_bias must have shape {(self.n_cells, 1)}'
        self.register_buffer('stacked_bias', torch.tensor(stacked_bias, dtype=dtype))

        # shape (n_cells, max_coupled_cells), integer LongTensor
        assert coupling_idx_sel.shape == (self.n_cells, self.max_coupled_cells), \
            f'coupling_idx_sel must have shape {(self.n_cells, self.max_coupled_cells)}'
        self.register_buffer('coupled_sel', torch.tensor(coupling_idx_sel, dtype=torch.long))

    def compute_stimulus_time_component(self,
                                        stim_time: torch.Tensor) -> torch.Tensor:
        '''
        Convolves the timecourse of each cell with the time component of the stimulus
        
        :param stim_time: shape (n_bins_observed, )
        :return: shape (n_cells, n_bins_observed); convolution of the timecourse of each cell with the
            time component of the stimulus
        '''

        # shape (1, 1, n_bins_observed) \ast (n_cells, 1, n_bins_filter)
        # -> (1, n_cells, n_bins_observed - n_bins_filter)
        conv_extra_dims = F.conv1d(stim_time[None, None, :],
                                   self.stacked_timecourse_filters[:, None, :])
        return conv_extra_dims.squeeze(0)

    def compute_stimulus_spat_component(self,
                                        spat_stim_flat: torch.Tensor) -> torch.Tensor:
        '''
        Applies the spatial filters and biases to the stimuli
        
        :param spat_stim_flat: shape (batch, n_pixels)
        :return: shape (batch, n_cells) ;
            result of applying the spatial filter for each cell onto each stimulus image
        '''

        # shape (batch, n_pixels) @ (n_pixels, n_cells)
        # -> (batch, n_cells)
        spat_filt_applied = spat_stim_flat @ self.stacked_flat_spat_filters.T
        return spat_filt_applied

    def compute_coupling_exp_arg(self,
                                 all_observed_spikes: torch.Tensor) -> torch.Tensor:
        '''
        Computes the coupling component of the generator signal from the real data
        
        Implementation: Gather using the specified indices, then a 1D conv

        :param all_observed_spikes: shape (batch, n_cells, n_bins_observed); observed spike trains for
            all of the cells, for every stimulus image
        :return: shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        '''

        batch_size, _, n_bins_observed = all_observed_spikes.shape

        # we want an output set of spike trains with shape
        # (batch, n_cells, max_coupled_cells, n_bins_observed)

        # we need to pick our data out of all_observed_spikes, which has shape
        # (batch, n_cells, n_bins_observed)
        # using indices contained in self.coupled_sel, which has shape
        # (n_cells, max_coupled_cells), which contains indices from 0 to (n_cells - 1)

        # in order to use gather, the number of dimensions of each need to match
        # (we need 4 total dimensions)

        # shape (batch_size, n_cells, max_coupled_cells, n_bins_observed), index dimension is dim2 max_coupled_cells
        indices_repeated = self.coupled_sel[None, :, :, None].expand(batch_size, -1, -1, n_bins_observed)

        # shape (batch_size, n_cells, n_cells, n_bins_observed)
        observed_spikes_repeated = all_observed_spikes[:, None, :, :].expand(-1, self.n_cells, -1, -1)

        # shape (batch_size, n_cells, max_coupled_cells, n_bins_observed)
        selected_spike_trains = torch.gather(observed_spikes_repeated, 2, indices_repeated)

        # now we have to do a 1D convolution with the coupling filters
        # the intended output has shape
        # (batch, n_cells, n_bins_observed - n_bins_filter + 1)

        # the coupling filters are in self.stacked_coupling_filters and have shape
        # (n_cells, n_coupled_cells, n_bins_filter)

        # this looks like it needs to be a grouped 1D convolution with some reshaping,
        # since we convolve along time, need to sum over the coupled cells, but have
        # an extra batch dimension

        # we do a 1D convolution, with n_cells different groups

        # shape (batch_size, n_cells * max_coupled_cells, n_bins_observed)
        selected_spike_trains_reshape = selected_spike_trains.reshape(batch_size, -1, n_bins_observed)

        # (batch_size, n_cells * max_coupled_cells, n_bins_observed) \ast (n_cells, n_coupled_cells, n_bins_filter)
        # -> (batch_size, n_cells, n_bins_filter)
        coupling_conv = F.conv1d(selected_spike_trains_reshape,
                                 self.stacked_coupling_filters,
                                 groups=self.n_cells)

        return coupling_conv

    def ar_all_but_one_sim_spikes(self,
                                  batched_image_flattened: torch.Tensor,
                                  stim_time: torch.Tensor,
                                  observed_spikes: torch.Tensor,
                                  n_repeats: int,
                                  debug: bool = False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        '''

        Method for simulating spikes (useful for debugging computation of the generator signal and the
            loss function)

        The idea: given a stimulus image, simulate the spikes for every cell, using the real observed
            spikes for the coupling signal but keeping the AR structure for the cell being simulated
            (hence the all_but_one)

        :param image_flattened: shape (batch, n_pixels), note that each frame is a different image,
            since we are assuming stimulus space-time separability and the time-component is known and fixed
        :param observed_spikes: shape (batch, n_cells, n_bins_total) Initial set of spikes required to set the
            initial conditions for the cell
        :param n_repeats: number of repeats for each image
        :return: shape (batch, n_cells, n_repeats, n_bins_total)
        '''

        with torch.no_grad():
            batch, n_pixels = batched_image_flattened.shape
            n_bins = stim_time.shape[0]

            # first compute all of the components of the generator signal
            # that do not depend on time

            # -> (n_cells, n_bins - n_bins_filter + 1)
            filtered_stim_time = self.compute_stimulus_time_component(stim_time)

            # -> (batch, n_cells)
            filtered_stim_spat = self.compute_stimulus_spat_component(batched_image_flattened)

            # shape (batch, n_cells, 1) * (1, n_cells, n_bins - n_bins_filter + 1)
            # -> (batch, n_cells, n_bins - n_bins_filter + 1)
            stimulus_gensig_contrib = filtered_stim_spat[:, :, None] * filtered_stim_time[None, :, :] \
                                      + self.stacked_bias[None, :, :]

            # shape (batch, n_cells, n_bins - n_bins_filter + 1)
            coupling_exp_arg = self.compute_coupling_exp_arg(observed_spikes)

            print(coupling_exp_arg.shape)

            # shape (batch, n_cells, n_bins - n_bins_filter + 1)
            generator_signal_piece = stimulus_gensig_contrib + coupling_exp_arg

            len_bins_gensig = generator_signal_piece.shape[2]
            len_bins_sim = len_bins_gensig - 1

            output_bins_acc = torch.zeros((batch, self.n_cells, n_repeats, n_bins), dtype=self.dtype,
                                          device=generator_signal_piece.device)

            # copy the spikes that occur before the simulation period
            output_bins_acc[:, :, :, :self.n_bins_filter] = observed_spikes[:, :, None, :self.n_bins_filter]

            for i in range(len_bins_sim):
                # (batch, n_cells, n_repeats, n_bins_filter) @ (1, n_cells, n_bins_filter, 1)
                # -> (batch, n_cells, n_repeats, 1) -> (batch, n_cells, n_repeats)
                batched_feedback_value = (output_bins_acc[:, :, :, i:i + self.n_bins_filter] @
                                          self.stacked_feedback_filters[None, :, :, None]).squeeze(3)

                # shape (batch, n_cells, 1) + (batch, n_cells, n_repeats)
                # -> (batch, n_cells, n_repeats)
                relev_gen_sig = generator_signal_piece[:, :, i][:, :, None] + batched_feedback_value

                output_bins_acc[:, :, :, i + self.n_bins_filter] = self.spike_generation_fn(relev_gen_sig)

            if debug:
                return output_bins_acc, stimulus_gensig_contrib, coupling_exp_arg
            return output_bins_acc


class KnownSeparableTrialGLMLoss_Precompute(nn.Module):

    def __init__(self,
                 stacked_spatial_filters: np.ndarray,
                 stacked_timecourse_filters: np.ndarray,
                 stacked_feedback_filters: np.ndarray,
                 stacked_coupling_filters: np.ndarray,
                 coupling_idx_sel: np.ndarray,
                 stacked_bias: np.ndarray,
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):
        '''

        :param stacked_spatial_filters: shape (n_cells, height, width); spatial filters, one per cell/GLM
        :param stacked_timecourse_filters: shape (n_cells, n_bins_filter); timecourse filters, one per cell/GLM
        :param stacked_feedback_filters: shape (n_cells, n_bins_filter); feedback filters, one per cell/GLM
        :param stacked_coupling_filters: shape (n_cells, max_coupled_cells, n_bins_filter); coupling_filters,
        :param coupling_idx_sel: shape (n_cells, max_coupled_cells);
        :param stacked_bias: shape (n_cells, 1)
        :param stimulus_time_component: shape (n_bins, )
        '''
        super().__init__()

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_bins_filter = stacked_timecourse_filters.shape[1]
        self.max_coupled_cells = stacked_coupling_filters.shape[1]
        self.n_pixels = self.height * self.width
        self.n_bins_total = stimulus_time_component.shape[0]
        self.n_bins_reconstruction = self.n_bins_total - self.n_bins_filter

        self.dtype = dtype
        self.spiking_loss_fn = spiking_loss_fn

        # fixed temporal component of the stimulus
        self.register_buffer('stim_time_component', torch.tensor(stimulus_time_component, dtype=dtype))

        ##### GLM parameters as torch buffers #############################################
        # shape (n_cells, n_pixels)
        stacked_flat_spat_filters = stacked_spatial_filters.reshape(self.n_cells, -1)
        self.register_buffer('stacked_flat_spat_filters', torch.tensor(stacked_flat_spat_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_timecourse_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_timecourse_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_timecourse_filters', torch.tensor(stacked_timecourse_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_feedback_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_feedback_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_feedback_filters', torch.tensor(stacked_feedback_filters, dtype=dtype))

        # shape (n_cells, max_coupled_cells, n_bins_filter)
        assert stacked_coupling_filters.shape == (self.n_cells, self.max_coupled_cells, self.n_bins_filter), \
            f'stacked_coupling_filters must have shape {(self.n_cells, self.max_coupled_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_coupling_filters', torch.tensor(stacked_coupling_filters, dtype=dtype))

        # shape (n_cells, 1)
        assert stacked_bias.shape == (self.n_cells, 1), f'stacked_bias must have shape {(self.n_cells, 1)}'
        self.register_buffer('stacked_bias', torch.tensor(stacked_bias, dtype=dtype))

        # shape (n_cells, max_coupled_cells), integer LongTensor
        assert coupling_idx_sel.shape == (self.n_cells, self.max_coupled_cells), \
            f'coupling_idx_sel must have shape {(self.n_cells, self.max_coupled_cells)}'
        self.register_buffer('coupled_sel', torch.tensor(coupling_idx_sel, dtype=torch.long))

        # Create buffers for pre-computed quantities (stuff that doesn't for a fixed spike train)
        self.register_buffer('precomputed_feedback_coupling_gensig',
                             torch.zeros((self.n_cells, self.n_bins_total - self.n_bins_filter + 1),
                                         dtype=torch.float32))

        self.register_buffer('precomputed_timecourse_contrib',
                             torch.zeros((self.n_cells, self.n_bins_total - self.n_bins_filter + 1)))

    def precompute_gensig_components(self,
                                     all_observed_spikes: torch.Tensor) -> None:
        fc_pc = self._precompute_feedback_coupling_gensig_components(all_observed_spikes)
        t_pc = self._precompute_timecourse_component(self.stim_time_component)

        self.precomputed_feedback_coupling_gensig.data[:] = fc_pc.data[:]
        self.precomputed_timecourse_contrib.data[:] = t_pc.data[:]

    def compute_feedback_exp_arg(self,
                                 all_observed_spikes: torch.Tensor) -> torch.Tensor:
        '''
        Computes the feedback component of the generator signal from the real data for every cell

        Implementation: 1D conv with groups
        :param all_observed_spikes:  shape (n_cells, n_bins_observed), observed spike trains for
            all of the cells, for just the image being reconstructed
        :return: shape (n_cells, n_bins_observed - n_bins_filter + 1)
        '''

        # the feedback filters have shape
        # shape (n_cells, n_bins_filter), one for every cell

        # the observed spikes have shape (n_cells, n_bins_observed)

        # we want an output with shape (n_cells, n_bins_observed - n_bins_filter + 1)

        # (1, n_cells, n_bins_observed) \ast (n_cells, 1, n_bins_filter)
        # -> (1, n_cells, n_bins_observed - n_bins_filter + 1)
        # -> (n_cells, n_bins_observed - n_bins_filter + 1)
        conv_padded = F.conv1d(all_observed_spikes[None, :, :],
                               self.stacked_feedback_filters[:, None, :],
                               groups=self.n_cells).squeeze(0)

        return conv_padded

    def compute_coupling_exp_arg(self,
                                 all_observed_spikes: torch.Tensor) -> torch.Tensor:
        '''
        Computes the coupling component of the generator signal from the real data for every cell

        Implementation: Gather using the specified indices, then a 1D conv

        :param all_observed_spikes: shape (n_cells, n_bins_observed); observed spike trains for
            all of the cells, for just the image being reconstructed
        :return: shape (n_cells, n_bins_observed - n_bins_filter + 1)
        '''

        _, n_bins_observed = all_observed_spikes.shape

        # we want an output set of spike trains with shape
        # (n_cells, max_coupled_cells, n_bins_observed)

        # we need to pick our data out of all_observed_spikes, which has shape
        # (n_cells, n_bins_observed)
        # using indices contained in self.coupled_sel, which has shape
        # (n_cells, max_coupled_cells), which contains indices from 0 to (n_cells - 1)

        # in order to use gather, the number of dimensions of each need to match
        # (we need 3 total dimensions)

        # shape (n_cells, max_coupled_cells, n_bins_observed), index dimension is dim1 max_coupled_cells
        indices_repeated = self.coupled_sel[:, :, None].expand(-1, -1, n_bins_observed)

        # shape (n_cells, n_cells, n_bins_observed)
        observed_spikes_repeated = all_observed_spikes[None, :, :].expand(self.n_cells, -1, -1)

        # shape (n_cells, max_coupled_cells, n_bins_observed)
        selected_spike_trains = torch.gather(observed_spikes_repeated, 1, indices_repeated)

        # now we have to do a 1D convolution with the coupling filters
        # the intended output has shape
        # (n_cells, n_bins_observed - n_bins_filter + 1)

        # the input is in selected_spike_trains and has shape
        # (n_cells, max_coupled_cells, n_bins_observed)

        # the coupling filters are in self.stacked_coupling_filters and have shape
        # (n_cells, n_coupled_cells, n_bins_filter)

        # this looks like it needs to be a grouped 1D convolution with some reshaping,
        # since we convolve along time, need to sum over the coupled cells, but have
        # an extra batch dimension

        # we do a 1D convolution, with n_cells different groups

        # shape (1, n_cells * max_coupled_cells, n_bins_observed)
        selected_spike_trains_reshape = selected_spike_trains.reshape(1, -1, n_bins_observed)

        # (1, n_cells * max_coupled_cells, n_bins_observed) \ast (n_cells, n_coupled_cells, n_bins_filter)
        # -> (1, n_cells, n_bins_filter) -> (n_cells, n_bins_observed - n_bins_filter + 1)
        coupling_conv = F.conv1d(selected_spike_trains_reshape,
                                 self.stacked_coupling_filters,
                                 groups=self.n_cells).squeeze(0)

        return coupling_conv

    def _precompute_feedback_coupling_gensig_components(self,
                                                        all_cells_spiketrain: torch.Tensor) -> torch.Tensor:
        '''
        Precompute the feedback and coupling components of the generator signal,
            since given a fixed observedspike train, these components do not depend
            on the stimulus at all. When doing reconstruction, the only thing that
            changes is the spatial component of the stimulus.
        :param all_cells_spiketrain:
        :return: shape (n_cells, n_bins_observed - n_bins_filter + 1); sum of the coupling and feedback
            components to the generator signal
        '''

        with torch.no_grad():
            coupling_component = self.compute_coupling_exp_arg(all_cells_spiketrain)
            feedback_component = self.compute_feedback_exp_arg(all_cells_spiketrain)
            return coupling_component + feedback_component

    def _precompute_timecourse_component(self,
                                         stim_time: torch.Tensor) -> torch.Tensor:
        '''
        Convolves the timecourse of each cell with the time component of the stimulus

        :param stim_time: shape (n_bins_observed, )
        :return: shape (n_cells, n_bins_observed - n_bins_filter + 1);
            convolution of the timecourse of each cell with the time component of the stimulus
        '''

        with torch.no_grad():
            # shape (1, 1, n_bins_observed) \ast (n_cells, 1, n_bins_filter)
            # -> (1, n_cells, n_bins_observed - n_bins_filter + 1)
            conv_extra_dims = F.conv1d(stim_time[None, None, :],
                                       self.stacked_timecourse_filters[:, None, :])
            return conv_extra_dims.squeeze(0)

    def compute_stimulus_spat_component(self,
                                        spat_stim_flat: torch.Tensor) -> torch.Tensor:
        '''
        Applies the spatial filters and biases to the stimuli

        :param spat_stim_flat: shape (n_pixels, )
        :return: shape (n_cells, ) ;
            result of applying the spatial filter for each cell onto each stimulus image
        '''

        # shape (1, n_pixels) @ (n_pixels, n_cells)
        # -> (1, n_cells) -> (n_cells, )
        spat_filt_applied = (spat_stim_flat[None, :] @ self.stacked_flat_spat_filters.T).squeeze(0)
        return spat_filt_applied

    def gen_sig(self, image_flattened: torch.Tensor) -> torch.Tensor:
        '''
        The generator signal no longer explicitly depends on the observed spike train,
            since for a fixed spike train we can precompute that component

        :param image_flattened: shape (n_pixels, )
        :param observed_spikes: shape (n_cells, n_bins_observed)
        :return: shape (n_cells, n_bins_observed - n_bins_filter + 1)
        '''

        # shape (n_cells, )
        gensig_spat_component = self.compute_stimulus_spat_component(image_flattened)

        # shape (n_cells, 1) * (n_cells, n_bins_observed - n_bins_filter + 1)
        # -> (n_cells, n_bins_observed - n_bins_filter + 1)
        gensig_spat_time = gensig_spat_component[:, None] * self.precomputed_timecourse_contrib + self.stacked_bias

        # shape (n_cells, n_bins_observed - n_bins_filter + 1)
        total_gensig = gensig_spat_time + self.precomputed_feedback_coupling_gensig

        return total_gensig

    def image_loss(self, image_flattened: torch.Tensor, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param image_flattened: shape (n_pixels, )
        :param observed_spikes_null_padded:  torch.Tensor, shape (n_cells, n_bins_observed), first row is the
            NULL cell with no spikes observed
        '''

        # shape (n_cells, n_bins_observed - n_bins_filter + 1)
        generator_signal = self.gen_sig(image_flattened)

        loss_per_timestep = self.spiking_loss_fn(generator_signal[:, :-1],
                                                 observed_spikes[:, self.n_bins_filter:])
        return loss_per_timestep

    def calculate_loss(self, image_imshape: torch.Tensor, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param image_imshape: shape (height, width)
        :param observed_spikes:  shape (n_cells, n_bins_observed)
        :return:
        '''

        # shape (n_pixels, )
        image_flat = image_imshape.reshape(self.n_pixels)

        loss_per_bin = self.image_loss(image_flat, observed_spikes)

        return torch.sum(loss_per_bin, dim=0)

    def image_gradient(self, image_imshape: torch.Tensor, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param image_imshape: shape (height, width)
        :param observed_spikes: shape (n_cells, n_bins_observed)
        :return: shape (height, width)
        '''
        image_flattened = image_imshape.reshape(self.height * self.width, )
        image_flattened.requires_grad_(True)

        # shape (n_bins_reconstructed, )
        loss_per_bin = self.image_loss(image_flattened, observed_spikes)

        total_loss = torch.sum(loss_per_bin, dim=0)

        gradient_image_flat, = autograd.grad(total_loss, image_flattened)

        gradient_imshape = gradient_image_flat.reshape(self.height, self.width)

        return -gradient_imshape


class KnownSeparableTrialGLM_Precompute_HQS_XProb(SingleUnconstrainedProblem,
                                                  HQS_X_Problem):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 stacked_spatial_filters: np.ndarray,
                 stacked_timecourse_filters: np.ndarray,
                 stacked_feedback_filters: np.ndarray,
                 stacked_coupling_filters: np.ndarray,
                 coupling_idx_sel: np.ndarray,
                 stacked_bias: np.ndarray,
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 hqs_rho: float,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.rho = hqs_rho
        self.valid_prox_arg = False

        self.glm_encoding_loss = KnownSeparableTrialGLMLoss_Precompute(
            stacked_spatial_filters,
            stacked_timecourse_filters,
            stacked_feedback_filters,
            stacked_coupling_filters,
            coupling_idx_sel,
            stacked_bias,
            stimulus_time_component,
            spiking_loss_fn,
            dtype=dtype
        )

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape

        ### HQS constants #####################################################
        self.register_buffer('z_const_tensor', torch.zeros((self.height, self.width), dtype=dtype))

        ### OPTIMIZATION VARIABLES ############################################
        self.x_image = nn.Parameter(torch.empty((self.height, self.width), dtype=dtype),
                                    requires_grad=True)
        nn.init.normal_(self.x_image, mean=0.0, std=0.25)  # FIXME we may want a different noise initialization strategy

    def precompute_gensig_components(self,
                                     all_observed_spikes: torch.Tensor) -> None:
        return self.glm_encoding_loss.precompute_gensig_components(all_observed_spikes)

    def assign_z(self, z: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = z.data

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        # shape (height, width)
        image_imshape = args[self.IMAGE_IDX_ARGS]

        # shape (n_cells, )
        observed_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape ()
        encoding_loss = self.glm_encoding_loss.calculate_loss(image_imshape, observed_spikes)

        prox_diff = image_imshape - self.z_const_tensor
        prox_loss = 0.5 * self.rho * torch.sum(prox_diff * prox_diff)

        return encoding_loss + prox_loss

    def reinitialize_variables(self,
                               initialized_z_const: Optional[torch.Tensor] = None) -> None:
        # nn.init.normal_(self.z_const_tensor, mean=0.0, std=1.0)
        if initialized_z_const is None:
            self.z_const_tensor.data[:] = 0.0
        else:
            self.z_const_tensor.data[:] = initialized_z_const.data[:]

        nn.init.normal_(self.x_image, mean=0.0, std=0.5)

    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def get_reconstructed_image(self) -> np.ndarray:
        return self.x_image.detach().cpu().numpy()


class BatchKnownSeparableTrialGLMLoss_Precompute(nn.Module):

    def __init__(self,
                 batch: int,
                 stacked_spatial_filters: np.ndarray,
                 stacked_timecourse_filters: np.ndarray,
                 stacked_feedback_filters: np.ndarray,
                 stacked_coupling_filters: np.ndarray,
                 coupling_idx_sel: np.ndarray,
                 stacked_bias: np.ndarray,
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):

        super().__init__()

        self.batch = batch
        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_bins_filter = stacked_timecourse_filters.shape[1]
        self.max_coupled_cells = stacked_coupling_filters.shape[1]
        self.n_pixels = self.height * self.width
        self.n_bins_total = stimulus_time_component.shape[0]
        self.n_bins_reconstruction = self.n_bins_total - self.n_bins_filter

        self.dtype = dtype
        self.spiking_loss_fn = spiking_loss_fn

        # fixed temporal component of the stimulus
        self.register_buffer('stim_time_component', torch.tensor(stimulus_time_component, dtype=dtype))

        ##### GLM parameters as torch buffers #############################################
        # shape (n_cells, n_pixels)
        stacked_flat_spat_filters = stacked_spatial_filters.reshape(self.n_cells, -1)
        self.register_buffer('stacked_flat_spat_filters', torch.tensor(stacked_flat_spat_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_timecourse_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_timecourse_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_timecourse_filters', torch.tensor(stacked_timecourse_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_feedback_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_feedback_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_feedback_filters', torch.tensor(stacked_feedback_filters, dtype=dtype))

        # shape (n_cells, max_coupled_cells, n_bins_filter)
        assert stacked_coupling_filters.shape == (self.n_cells, self.max_coupled_cells, self.n_bins_filter), \
            f'stacked_coupling_filters must have shape {(self.n_cells, self.max_coupled_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_coupling_filters', torch.tensor(stacked_coupling_filters, dtype=dtype))

        # shape (n_cells, 1)
        assert stacked_bias.shape == (self.n_cells, 1), f'stacked_bias must have shape {(self.n_cells, 1)}'
        self.register_buffer('stacked_bias', torch.tensor(stacked_bias, dtype=dtype))

        # shape (n_cells, max_coupled_cells), integer LongTensor
        assert coupling_idx_sel.shape == (self.n_cells, self.max_coupled_cells), \
            f'coupling_idx_sel must have shape {(self.n_cells, self.max_coupled_cells)}'
        self.register_buffer('coupled_sel', torch.tensor(coupling_idx_sel, dtype=torch.long))

        # Create buffers for pre-computed quantities (stuff that doesn't for a fixed spike train)
        self.register_buffer('precomputed_feedback_coupling_gensig',
                             torch.zeros((self.batch, self.n_cells, self.n_bins_total - self.n_bins_filter + 1),
                                         dtype=torch.float32))

        self.register_buffer('precomputed_timecourse_contrib',
                             torch.zeros((self.n_cells, self.n_bins_total - self.n_bins_filter + 1)))

    def precompute_gensig_components(self,
                                     all_observed_spikes: torch.Tensor) -> None:
        fc_pc = self._precompute_feedback_coupling_gensig_components(all_observed_spikes)
        t_pc = self._precompute_timecourse_component(self.stim_time_component)

        self.precomputed_feedback_coupling_gensig.data[:] = fc_pc.data[:]
        self.precomputed_timecourse_contrib.data[:] = t_pc.data[:]


    def compute_feedback_exp_arg(self,
                                 all_observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param all_observed_spikes: (batch, n_cells, n_bins_observed),
            one entry for every batch, every cell
        :return: shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        '''

        conv_padded = F.conv1d(all_observed_spikes,
                               self.stacked_feedback_filters[:, None, :],
                               groups=self.n_cells)
        return conv_padded

    def compute_coupling_exp_arg(self,
                                 all_observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param all_observed_spikes: shpae (batch, n_cells, n_bins_observed)
        :return: shape (batchm, n_cells, n_bins_observed - n_bins_filter + 1)
        '''

        batch, n_cells, n_bins_observed = all_observed_spikes.shape

        # we want an output set of spike trains with shape
        # (batch, n_cells, max_coupled_cells, n_bins_observed)

        # we need to pick our data out of all_observed_spikes, which has shape
        # (batch, n_cells, n_bins_observed)
        # using indices contained in self.coupled_sel, which has shape
        # (n_cells, max_coupled_cells), which contains indices from 0 to (n_cells - 1)

        # in order to use gather, the number of dimensions of each need to match
        # (we need 4 total dimensions)

        # shape (batch, n_cells, max_coupled_cells, n_bins_observed), index dimension is dim1 max_coupled_cells
        indices_repeated = self.coupled_sel[:, :, None].expand(batch, -1, -1, n_bins_observed)

        # shape (batch, n_cells, n_cells, n_bins_observed)
        observed_spikes_repeated = all_observed_spikes[:, None, :, :].expand(-1, self.n_cells, -1, -1)

        # shape (batch, n_cells, max_coupled_cells, n_bins_observed)
        selected_spike_trains = torch.gather(observed_spikes_repeated, 2, indices_repeated)

        # now we have to do a 1D convolution with the coupling filters
        # the intended output has shape
        # (n_cells, n_bins_observed - n_bins_filter + 1)

        # the input is in selected_spike_trains and has shape
        # (n_cells, max_coupled_cells, n_bins_observed)

        # the coupling filters are in self.stacked_coupling_filters and have shape
        # (n_cells, n_coupled_cells, n_bins_filter)

        # this looks like it needs to be a grouped 1D convolution with some reshaping,
        # since we convolve along time, need to sum over the coupled cells, but have
        # an extra batch dimension

        # we do a 1D convolution, with n_cells different groups

        # shape (batch, n_cells * max_coupled_cells, n_bins_observed)
        selected_spike_trains_reshape = selected_spike_trains.reshape(batch, -1, n_bins_observed)

        # (batch, n_cells * max_coupled_cells, n_bins_observed) \ast (n_cells, n_coupled_cells, n_bins_filter)
        # -> (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        coupling_conv = F.conv1d(selected_spike_trains_reshape,
                                 self.stacked_coupling_filters,
                                 groups=self.n_cells)

        return coupling_conv

    def _precompute_feedback_coupling_gensig_components(self,
                                                        all_cells_spiketrain: torch.Tensor) -> torch.Tensor:
        '''
        Precompute the feedback and coupling components of the generator signal,
            since given a fixed observedspike train, these components do not depend
            on the stimulus at all. When doing reconstruction, the only thing that
            changes is the spatial component of the stimulus.
        :param all_cells_spiketrain:
        :return: shape (batch, n_cells, n_bins_observed - n_bins_filter + 1); sum of the coupling and feedback
            components to the generator signal
        '''

        with torch.no_grad():
            coupling_component = self.compute_coupling_exp_arg(all_cells_spiketrain)
            feedback_component = self.compute_feedback_exp_arg(all_cells_spiketrain)
            return coupling_component + feedback_component

    def _precompute_timecourse_component(self,
                                         stim_time: torch.Tensor) -> torch.Tensor:
        '''
        Convolves the timecourse of each cell with the time component of the stimulus

        :param stim_time: shape (n_bins_observed, )
        :return: shape (n_cells, n_bins_observed - n_bins_filter + 1);
            convolution of the timecourse of each cell with the time component of the stimulus
        '''

        with torch.no_grad():
            # shape (1, 1, n_bins_observed) \ast (n_cells, 1, n_bins_filter)
            # -> (1, n_cells, n_bins_observed - n_bins_filter + 1)
            conv_extra_dims = F.conv1d(stim_time[None, None, :],
                                       self.stacked_timecourse_filters[:, None, :])
            return conv_extra_dims.squeeze(0)

    def compute_stimulus_spat_component(self,
                                        spat_stim_flat: torch.Tensor) -> torch.Tensor:
        '''
        Applies the spatial filters and biases to the stimuli

        :param spat_stim_flat: shape (batch, n_pixels)
        :return: shape (n_cells, ) ;
            result of applying the spatial filter for each cell onto each stimulus image
        '''

        # shape (batch, n_pixels) @ (n_pixels, n_cells)
        # -> (batch, n_cells) -> (n_cells, )
        spat_filt_applied = (spat_stim_flat @ self.stacked_flat_spat_filters.T)
        return spat_filt_applied

    def gen_sig(self, image_flattened: torch.Tensor) -> torch.Tensor:
        '''
        The generator signal no longer explicitly depends on the observed spike train,
            since for a fixed spike train we can precompute that component

        :param image_flattened: shape (batch, n_pixels)
        :param observed_spikes: shape (batch, n_cells, n_bins_observed)
        :return: shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        '''

        # shape (batch, n_cells)
        gensig_spat_component = self.compute_stimulus_spat_component(image_flattened)

        # shape (batch, n_cells, 1) * (1, n_cells, n_bins_observed - n_bins_filter + 1)
        # -> (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        gensig_spat_time = gensig_spat_component[:, :, None] * self.precomputed_timecourse_contrib[None, :, :] \
            + self.stacked_bias[None, :, :]

        # shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        total_gensig = gensig_spat_time + self.precomputed_feedback_coupling_gensig

        return total_gensig

    def image_loss(self, image_flattened: torch.Tensor, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param image_flattened: shape (batch, n_pixels)
        :param observed_spikes_null_padded:  torch.Tensor, shape (batch, n_cells, n_bins_observed), first row is the
            NULL cell with no spikes observed
        '''

        # shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        generator_signal = self.gen_sig(image_flattened)

        # shape (batch, n_bins_observed - n_bins_filter + 1)
        loss_per_timestep = self.spiking_loss_fn(generator_signal[:, :, :-1],
                                                 observed_spikes[:, :, self.n_bins_filter:])
        return loss_per_timestep

    def calculate_loss(self, image_imshape: torch.Tensor, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param image_imshape: shape (height, width)
        :param observed_spikes:  shape (n_cells, n_bins_observed)
        :return:
        '''

        # shape (batch, n_pixels)
        image_flat = image_imshape.reshape(self.batch, self.n_pixels)

        # shape (batch, n_bins_observed - n_bins_filter + 1)
        loss_per_bin = self.image_loss(image_flat, observed_spikes)

        # shape (batch, )
        return torch.sum(loss_per_bin, dim=1)

    def multi_image_gradient(self,
                             batched_multi_image: torch.Tensor,
                             batched_spikes: torch.Tensor) -> torch.Tensor:
        '''
        Calculates the positive gradient of the multiple losses with respect
            to multiple images in closed form using autograd, for fixed batch of images

        Useful for computing Hessian-vector products in parallel

        :param batched_multi_image:
        :param batched_spikes:
        :return:
        '''

        pass


class BatchKnownSeparable_TrialGLM_ProxProblem(BatchParallelUnconstrainedProblem, BatchParallel_HQS_X_Problem):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 batch: int,
                 stacked_spatial_filters: np.ndarray,
                 stacked_timecourse_filters: np.ndarray,
                 stacked_feedback_filters: np.ndarray,
                 stacked_coupling_filters: np.ndarray,
                 coupling_idx_sel: np.ndarray,
                 stacked_bias: np.ndarray,
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 rho: float,
                 dtype: torch.dtype = torch.float32):

        super().__init__()

        self.batch_size = batch
        self.rho = rho

        self.glm_loss_calc = BatchKnownSeparableTrialGLMLoss_Precompute(
            batch,
            stacked_spatial_filters,
            stacked_timecourse_filters,
            stacked_feedback_filters,
            stacked_coupling_filters,
            coupling_idx_sel,
            stacked_bias,
            stimulus_time_component,
            spiking_loss_fn,
            dtype=dtype
        ) # type: BatchKnownSeparableTrialGLMLoss_Precompute

        self.height, self.width = self.glm_loss_calc.height, self.glm_loss_calc.width


        #### CONSTANTS ###############################################
        self.register_buffer('z_const_tensor', torch.empty((self.batch_size, self.height, self.width), dtype=dtype))

        #### OPTIM VARIABLES #########################################
        # OPT VARIABLE 0: image, shape (batch, height, width)
        self.image = nn.Parameter(torch.empty((self.batch_size, self.height, self.width),
                                              dtype=dtype))
        nn.init.uniform_(self.image, a=-1e-2, b=1e-2)

    def assign_z(self, z: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = z.data

    def reinitialize_variables(self,
                               initialized_z_const: Optional[torch.Tensor] = None) -> None:
        # nn.init.normal_(self.z_const_tensor, mean=0.0, std=1.0)
        if initialized_z_const is None:
            self.z_const_tensor.data[:] = 0.0
        else:
            self.z_const_tensor.data[:] = initialized_z_const.data[:]

        nn.init.normal_(self.image, mean=0.0, std=0.5)

    def get_reconstructed_image(self) -> np.ndarray:
        return self.image.detach().cpu().numpy()

    @property
    def n_problems(self) -> int:
        return self.batch_size

    def precompute_gensig_components(self, observed_spikes: torch.Tensor) -> None:
        '''

        :param observed_spikes: shape (batch, n_cells, n_bins_observed)
        :return:
        '''
        self.glm_loss_calc.precompute_gensig_components(observed_spikes)

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        '''

        :param args:
        :param kwargs:
        :return:
        '''

        # shape (batch, height, width)
        batched_image_imshape = args[self.IMAGE_IDX_ARGS]

        # shape (batch, n_cells, n_bins)
        batched_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]


        # shape ()
        encoding_loss = self.glm_loss_calc.calculate_loss(batched_image_imshape,
                                                          batched_spikes)

        # shape ()
        prox_diff = batched_image_imshape - self.z_const_tensor
        prox_loss = 0.5 * self.rho * torch.sum(prox_diff * prox_diff, dim=(1, 2))

        return encoding_loss + prox_loss

    def assign_proxto(self, prox_to: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = prox_to.data[:]

    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def get_output_image(self) -> torch.Tensor:
        return self.reconstructed_images.detach().clone()

    def encoding_multi_image_gradient(self,
                                      batched_multi_images: torch.Tensor,
                                      batched_spikes: torch.Tensor):
        return self.glm_loss_calc.multi_image_gradient(batched_multi_images,
                                                       batched_spikes)


class BatchParallelPatchGaussian1FPriorGLMReconstruction(BatchParallelUnconstrainedProblem):

    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 batch_size: int,
                 patch_zca_matrix: np.ndarray,
                 gaussian_prior_lambda: float,
                 stacked_spatial_filters: np.ndarray,
                 stacked_timecourse_filters: np.ndarray,
                 stacked_feedback_filters: np.ndarray,
                 stacked_coupling_filters: np.ndarray,
                 coupling_idx_sel: np.ndarray,
                 stacked_bias: np.ndarray,
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32,
                 patch_stride: int = 1):

        super().__init__()

        self.batch = batch_size

        self.gaussian_prior_lambda = gaussian_prior_lambda
        self.batch_prior_callable = ConvPatch1FGaussianPrior(patch_zca_matrix,
                                                             patch_stride=patch_stride,
                                                             dtype=dtype)

        self.encoding_loss_callable = BatchKnownSeparableTrialGLMLoss_Precompute(
            batch_size,
            stacked_spatial_filters,
            stacked_timecourse_filters,
            stacked_feedback_filters,
            stacked_coupling_filters,
            coupling_idx_sel,
            stacked_bias,
            stimulus_time_component,
            spiking_loss_fn,
            dtype=dtype)

        self.height, self.width = self.encoding_loss_callable.height, self.encoding_loss_callable.width

        self.reconstructed_image = nn.Parameter(torch.empty((batch_size, self.height, self.width), dtype=dtype),
                                                requires_grad=True)
        nn.init.uniform_(self.reconstructed_image, a=-0.1, b=0.1)

    def get_reconstructed_image(self) -> np.ndarray:
        return self.reconstructed_image.detach().cpu().numpy()

    @property
    def n_problems(self) -> int:
        return self.batch

    def precompute_gensig_components(self, observed_spikes: torch.Tensor) -> None:
        self.encoding_loss_callable.precompute_gensig_components(observed_spikes)

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:

        # shape (batch, height, width)
        batched_image_imshape = args[self.IMAGE_IDX_ARGS]

        # shape (batch, n_cells)
        batched_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape (batch, )
        encoding_loss = self.encoding_loss_callable.calculate_loss(batched_image_imshape,
                                                                   batched_spikes)

        # shape (batch, )
        gaussian_prior_penalty = 0.5 * self.gaussian_prior_lambda * self.batch_prior_callable(batched_image_imshape)

        # shape (batch, )
        return encoding_loss + gaussian_prior_penalty


class Gaussian1FPriorGLMReconstruction(KnownSeparableTrialGLMLoss_Precompute, SingleUnconstrainedProblem):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 gaussian_prior_reconstruction_matrix: np.ndarray,
                 gaussian_prior_lambda: float,
                 stacked_spatial_filters: np.ndarray,
                 stacked_timecourse_filters: np.ndarray,
                 stacked_feedback_filters: np.ndarray,
                 stacked_coupling_filters: np.ndarray,
                 coupling_idx_sel: np.ndarray,
                 stacked_bias: np.ndarray,
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):
        KnownSeparableTrialGLMLoss_Precompute.__init__(self,
                                                       stacked_spatial_filters,
                                                       stacked_timecourse_filters,
                                                       stacked_feedback_filters,
                                                       stacked_coupling_filters,
                                                       coupling_idx_sel,
                                                       stacked_bias,
                                                       stimulus_time_component,
                                                       spiking_loss_fn,
                                                       dtype=dtype)

        self.gaussian_prior_lambda = gaussian_prior_lambda
        self.gaussian_prior_callable = Full1FGaussianPrior(gaussian_prior_reconstruction_matrix,
                                                           dtype=dtype)

        self.reconstructed_image = nn.Parameter(torch.empty((self.height, self.width), dtype=self.dtype),
                                                requires_grad=True)
        nn.init.uniform_(self.reconstructed_image, a=-0.1, b=0.1)

    def reinitialize_variables(self) -> None:
        nn.init.uniform_(self.reconstructed_image, a=-0.1, b=0.1)

    def get_reconstructed_image(self) -> np.ndarray:
        return self.reconstructed_image.detach().cpu().numpy()

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        # shape (n_cells, n_bins)
        observed_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape (height, width)
        image_imshape = args[self.IMAGE_IDX_ARGS]

        # shape (n_pixels, )
        image_flat = image_imshape.reshape(-1)

        # compute the encoding loss
        # shape (n_bins - n_bins_filter, )
        content_loss_per_bin = self.image_loss(image_flat,
                                               observed_spikes)

        total_content_loss = torch.sum(content_loss_per_bin, dim=0)

        # compute the regularization term
        regularization_term = self.gaussian_prior_callable(image_imshape)

        return total_content_loss + 0.5 * self.gaussian_prior_lambda * regularization_term


def trial_glm_nn_prior_known_separable_sampling_with_precompute(
        trial_glm_model: KnownSeparableTrialGLMLoss_Precompute,
        denoiser_callable: Callable[[torch.Tensor], torch.Tensor],
        image_height_width: Tuple[int, int],
        encoding_spikes: np.ndarray,
        device: torch.device,
        sig_0: float = 1.0,
        sig_L: float = 0.01,
        h0: float = 0.01,
        beta: float = 0.01,
        prior_term_weight: float = 1.0,
        max_iter: int = 1000) -> np.ndarray:
    '''
    Implementation note: This function owns the WIP reconstructed image, and does not
        use the image_flattened attribute of glm_model

    :param denoiser_callable:
    :param glm_model_params:
    :param encoding_spikes: np.ndarray, shape (n_cells, n_bins_observed)
        Note that this is unpadded, i.e. it does not have the null cell included.
        This function is supposed to take care of the padding

    :param stimulus_time_component: np.ndarray, shape (n_bins_observed, )
    :param spiking_loss_function:
    :param device:
    :param sig_0:
    :param sig_L:
    :param h0:
    :param beta:
    :param prior_term_weight:
    :param max_iter:
    :return:
    '''

    height, width = image_height_width
    n_pixels = height * width
    root_pixels = np.sqrt(n_pixels)

    y_iter_reconstruct = torch.randn(height, width, dtype=torch.float32, device=device) * sig_0
    encoding_spikes_tensor = torch.tensor(encoding_spikes, dtype=torch.float32, device=device)

    trial_glm_model.precompute_gensig_components(encoding_spikes_tensor)

    # TODO get rid of later
    print('DEBUG')

    t = 1
    sig_t = sig_0
    while sig_t >= sig_L:
        print(f't={t}, sig_t={sig_t}\r', end='')
        h_t = h0 * t / (1 + h0 * (t - 1))

        with torch.no_grad():
            # shape (1, height, width) -> (height, width)
            residual_imshape = denoiser_callable(y_iter_reconstruct[None, :, :]).squeeze(0)

            encoding_weight = sig_t * sig_t / prior_term_weight

        # shape (height, width)
        glm_loss_gradient = encoding_weight * trial_glm_model.image_gradient(y_iter_reconstruct,
                                                                             encoding_spikes_tensor) * 2  # TODO Get rid of later
        with torch.no_grad():
            # shape (height, width)
            d_t = residual_imshape + glm_loss_gradient

            # shape (n_bins_reconstruct, )
            sig_t = torch.linalg.norm(residual_imshape, dim=(0, 1)).item() / root_pixels

            gamma_t = np.sqrt((1 - beta * h_t) ** 2 - (1 - h_t) ** 2) * sig_t

            z_t = torch.randn_like(y_iter_reconstruct)

            # make sure we only update the not-yet-converged problems
            # logic: if sig_t drops below sig_L even once, we mark that problem
            # as converged forever
            step_update = h_t * d_t + gamma_t * z_t
            y_iter_reconstruct = y_iter_reconstruct + step_update

        if t > max_iter:
            break

        t += 1

    with torch.no_grad():
        residual_imshape = denoiser_callable(y_iter_reconstruct[None, :, :]).squeeze(0)
        final_image_retina_range = (y_iter_reconstruct + residual_imshape).detach().cpu().numpy()

    return final_image_retina_range


class KnownSeparableTrialGLMLoss(nn.Module):
    '''
    Method 1 for computing GLM loss and doing image reconstructions over all of the cells
    
    For this method, we assume that the stimulus is space-time separable (reasonably only for
        Nora's flashed static image trials), and that the time component of the stimulus (either
        a square wave or a step function) is known ahead of time and fixed. This is semi-cheating,
        but it is quite justifiable in comparison to both Nora's published linear reconstruction
        method and my previous binomial model stuff. The justification is that this method completely
        throws away the time component of the reconstruction, and does not generalize to movies.
        
    Convention:
        We assume that the reconstruction algorithm "knows" exactly what the stimulus is prior to the period
            that it is trying to reconstruct (i.e. for flashed trials, we explicitly let the reconstruction
            algorithm know that the stimulus is all grey prior to the flashed image). This is so that the initial
            conditions for the GLM are reasonable, but it is a bit of cheating...
    '''

    def __init__(self,
                 stacked_spatial_filters: np.ndarray,
                 stacked_timecourse_filters: np.ndarray,
                 stacked_feedback_filters: np.ndarray,
                 stacked_coupling_filters: np.ndarray,
                 coupling_idx_sel: np.ndarray,
                 stacked_bias: np.ndarray,
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):
        '''

        :param stacked_spatial_filters: shape (n_cells, height, width); spatial filters, one per cell/GLM
        :param stacked_timecourse_filters: shape (n_cells, n_bins_filter); timecourse filters, one per cell/GLM
        :param stacked_feedback_filters: shape (n_cells, n_bins_filter); feedback filters, one per cell/GLM
        :param stacked_coupling_filters: shape (n_cells, max_coupled_cells, n_bins_filter); coupling_filters,
        :param coupling_idx_sel: shape (n_cells, max_coupled_cells);
        :param stacked_bias: shape (n_cells, 1)
        :param n_bins_reconstruction: int, number of frames that we want to reconstruct; each reconstructed frame
            should correspond to a single timebin

            n_bins_reconstruction should satisfy
                n_bins_reconstruction + n_bins_filter = n_bins_spikes

        :param pre_flash_stimulus: shape (n_bins_filter, n_pixels), the exact stimulus for the time period directly
            preceding the period that we are trying to reconstruct; used to set the initial conditions of the GLM

            For the flashed trials, this should be a matrix of zeros.

        '''
        super().__init__()

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_bins_filter = stacked_timecourse_filters.shape[1]
        self.max_coupled_cells = stacked_coupling_filters.shape[1]
        self.n_pixels = self.height * self.width
        self.n_bins_total = stimulus_time_component.shape[0]
        self.n_bins_reconstruction = self.n_bins_total - self.n_bins_filter

        self.dtype = dtype
        self.spiking_loss_fn = spiking_loss_fn

        # fixed temporal component of the stimulus
        self.register_buffer('stim_time_component', torch.tensor(stimulus_time_component, dtype=dtype))

        ##### GLM parameters as torch buffers #############################################
        # shape (n_cells, n_pixels)
        stacked_flat_spat_filters = stacked_spatial_filters.reshape(self.n_cells, -1)
        self.register_buffer('stacked_flat_spat_filters', torch.tensor(stacked_flat_spat_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_timecourse_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_timecourse_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_timecourse_filters', torch.tensor(stacked_timecourse_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_feedback_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_feedback_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_feedback_filters', torch.tensor(stacked_feedback_filters, dtype=dtype))

        # shape (n_cells, max_coupled_cells, n_bins_filter)
        assert stacked_coupling_filters.shape == (self.n_cells, self.max_coupled_cells, self.n_bins_filter), \
            f'stacked_coupling_filters must have shape {(self.n_cells, self.max_coupled_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_coupling_filters', torch.tensor(stacked_coupling_filters, dtype=dtype))

        # shape (n_cells, 1)
        assert stacked_bias.shape == (self.n_cells, 1), f'stacked_bias must have shape {(self.n_cells, 1)}'
        self.register_buffer('stacked_bias', torch.tensor(stacked_bias, dtype=dtype))

        # shape (n_cells, max_coupled_cells), integer LongTensor
        assert coupling_idx_sel.shape == (self.n_cells, self.max_coupled_cells), \
            f'coupling_idx_sel must have shape {(self.n_cells, self.max_coupled_cells)}'
        self.register_buffer('coupled_sel', torch.tensor(coupling_idx_sel, dtype=torch.long))

        ##### Reconstruction variable ######################################################
        # ONLY USEFUL FOR WHEN SOLVING THE CONVEX OPTIMIZATION PROBLEM FOR AN MLE ESTIMATE
        # (RECONSTRUCTION WITH NO PRIOR)
        # shape (n_bins_reconstruction, n_pixels)
        self.image_flattened = nn.Parameter(torch.empty((self.n_pixels,), dtype=dtype),
                                            requires_grad=True)
        nn.init.uniform_(self.image_flattened, a=-0.1, b=0.1)  # FIXME parameterize this

    def compute_stimulus_time_component(self,
                                        stim_time: torch.Tensor) -> torch.Tensor:
        '''
        Convolves the timecourse of each cell with the time component of the stimulus

        :param stim_time: shape (n_bins_observed, )
        :return: shape (n_cells, n_bins_observed); convolution of the timecourse of each cell with the
            time component of the stimulus
        '''

        # shape (1, 1, n_bins_observed) \ast (n_cells, 1, n_bins_filter)
        # -> (1, n_cells, n_bins_observed - n_bins_filter)
        conv_extra_dims = F.conv1d(stim_time[None, None, :],
                                   self.stacked_timecourse_filters[:, None, :])
        return conv_extra_dims.squeeze(0)

    def compute_stimulus_spat_component(self,
                                        spat_stim_flat: torch.Tensor) -> torch.Tensor:
        '''
        Applies the spatial filters and biases to the stimuli

        :param spat_stim_flat: shape (n_pixels, )
        :return: shape (n_cells, ) ;
            result of applying the spatial filter for each cell onto each stimulus image
        '''

        # shape (1, n_pixels) @ (n_pixels, n_cells)
        # -> (1, n_cells) -> (n_cells, )
        spat_filt_applied = (spat_stim_flat[None, :] @ self.stacked_flat_spat_filters.T).squeeze(0)
        return spat_filt_applied

    def compute_feedback_exp_arg(self,
                                 all_observed_spikes: torch.Tensor) -> torch.Tensor:
        '''
        Computes the feedback component of the generator signal from the real data for every cell
        
        Implementation: 1D conv with groups
        :param all_observed_spikes:  shape (n_cells, n_bins_observed), observed spike trains for 
            all of the cells, for just the image being reconstructed
        :return: shape (n_cells, n_bins_observed - n_bins_filter + 1)
        '''

        # the feedback filters have shape
        # shape (n_cells, n_bins_filter), one for every cell

        # the observed spikes have shape (n_cells, n_bins_observed)

        # we want an output with shape (n_cells, n_bins_observed - n_bins_filter + 1)

        # (1, n_cells, n_bins_observed) \ast (n_cells, 1, n_bins_filter)
        # -> (1, n_cells, n_bins_observed - n_bins_filter + 1)
        conv_padded = F.conv1d(all_observed_spikes[None, :, :],
                               self.stacked_feedback_filters[:, None, :],
                               groups=self.n_cells).squeeze(0)

        return conv_padded

    def compute_coupling_exp_arg(self,
                                 all_observed_spikes: torch.Tensor) -> torch.Tensor:
        '''
        Computes the coupling component of the generator signal from the real data for every cell 

        Implementation: Gather using the specified indices, then a 1D conv

        :param all_observed_spikes: shape (n_cells, n_bins_observed); observed spike trains for
            all of the cells, for just the image being reconstructed 
        :return: shape (n_cells, n_bins_observed - n_bins_filter + 1)
        '''

        _, n_bins_observed = all_observed_spikes.shape

        # we want an output set of spike trains with shape
        # (n_cells, max_coupled_cells, n_bins_observed)

        # we need to pick our data out of all_observed_spikes, which has shape
        # (n_cells, n_bins_observed)
        # using indices contained in self.coupled_sel, which has shape
        # (n_cells, max_coupled_cells), which contains indices from 0 to (n_cells - 1)

        # in order to use gather, the number of dimensions of each need to match
        # (we need 3 total dimensions)

        # shape (n_cells, max_coupled_cells, n_bins_observed), index dimension is dim1 max_coupled_cells
        indices_repeated = self.coupled_sel[:, :, None].expand(-1, -1, n_bins_observed)

        # shape (n_cells, n_cells, n_bins_observed)
        observed_spikes_repeated = all_observed_spikes[None, :, :].expand(self.n_cells, -1, -1)

        # shape (n_cells, max_coupled_cells, n_bins_observed)
        selected_spike_trains = torch.gather(observed_spikes_repeated, 1, indices_repeated)

        # now we have to do a 1D convolution with the coupling filters
        # the intended output has shape
        # (n_cells, n_bins_observed - n_bins_filter + 1)

        # the input is in selected_spike_trains and has shape
        # (n_cells, max_coupled_cells, n_bins_observed)

        # the coupling filters are in self.stacked_coupling_filters and have shape
        # (n_cells, n_coupled_cells, n_bins_filter)

        # this looks like it needs to be a grouped 1D convolution with some reshaping,
        # since we convolve along time, need to sum over the coupled cells, but have
        # an extra batch dimension

        # we do a 1D convolution, with n_cells different groups

        # shape (1, n_cells * max_coupled_cells, n_bins_observed)
        selected_spike_trains_reshape = selected_spike_trains.reshape(1, -1, n_bins_observed)

        # (1, n_cells * max_coupled_cells, n_bins_observed) \ast (n_cells, n_coupled_cells, n_bins_filter)
        # -> (1, n_cells, n_bins_filter) -> (n_cells, n_bins_observed - n_bins_filter + 1)
        coupling_conv = F.conv1d(selected_spike_trains_reshape,
                                 self.stacked_coupling_filters,
                                 groups=self.n_cells).squeeze(0)

        return coupling_conv

    def gen_sig(self, image_flattened: torch.Tensor, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''
        :param image_flattened: shape (n_pixels, )
        :param observed_spikes: shape (n_cells, n_bins_observed)
        :return: shape (n_cells, n_bins_observed - n_bins_filter + 1)
        '''

        # shape (n_cells, n_bins_observed - n_bins_filter + 1)
        gensig_time_component = self.compute_stimulus_time_component(self.stim_time_component)

        # shape (n_cells, )
        gensig_spat_component = self.compute_stimulus_spat_component(image_flattened)

        # shape (n_cells, 1) * (n_cells, n_bins_observed - n_bins_filter + 1)
        # -> (n_cells, n_bins_observed - n_bins_filter + 1)
        gensig_spat_time = gensig_spat_component[:, None] * gensig_time_component + self.stacked_bias

        # shape (n_cells, n_bins_observed - n_bins_filter + 1)
        coupling_component = self.compute_coupling_exp_arg(observed_spikes)

        # shape (n_cells, n_bins_observed - n_bins_filter + 1)
        feedback_component = self.compute_feedback_exp_arg(observed_spikes)

        # shape (n_cells, n_bins_observed - n_bins_filter + 1)
        total_gensig = gensig_spat_time + coupling_component + feedback_component

        return total_gensig

    def image_loss(self, image_flattened: torch.Tensor, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param image_flattened: shape (n_pixels, )
        :param observed_spikes_null_padded:  torch.Tensor, shape (n_cells, n_bins_observed), first row is the
            NULL cell with no spikes observed
        '''

        # shape (n_cells, n_bins_observed - n_bins_filter + 1)
        generator_signal = self.gen_sig(image_flattened, observed_spikes)

        loss_per_timestep = self.spiking_loss_fn(generator_signal[:, :-1],
                                                 observed_spikes[:, self.n_bins_filter:])
        return loss_per_timestep

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        image_flattened = args[self.IMAGE_IDX_ARGS]
        padded_observed_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape (n_bins_reconstructed, )
        loss_per_bin = self.image_loss(image_flattened, padded_observed_spikes)

        total_loss = torch.sum(loss_per_bin, dim=0)

        return total_loss

    def image_gradient(self, image_imshape: torch.Tensor, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param image_imshape: shape (height, width)
        :param observed_spikes: shape (n_cells, n_bins_observed)
        :return: shape (height, width)
        '''
        image_flattened = image_imshape.reshape(self.height * self.width, )
        image_flattened.requires_grad_(True)

        # shape (n_bins_reconstructed, )
        loss_per_bin = self.image_loss(image_flattened, observed_spikes)

        total_loss = torch.sum(loss_per_bin, dim=0)

        gradient_image_flat, = autograd.grad(total_loss, image_flattened)

        gradient_imshape = gradient_image_flat.reshape(self.height, self.width)

        return -gradient_imshape


def trial_glm_nn_prior_known_separable_sampling(
        trial_glm_model: KnownSeparableTrialGLMLoss,
        denoiser_callable: Callable[[torch.Tensor], torch.Tensor],
        image_height_width: Tuple[int, int],
        encoding_spikes: np.ndarray,
        device: torch.device,
        sig_0: float = 1.0,
        sig_L: float = 0.01,
        h0: float = 0.01,
        beta: float = 0.01,
        prior_term_weight: float = 1.0,
        max_iter: int = 1000) -> np.ndarray:
    '''
    Implementation note: This function owns the WIP reconstructed image, and does not
        use the image_flattened attribute of glm_model

    :param denoiser_callable:
    :param glm_model_params:
    :param encoding_spikes: np.ndarray, shape (n_cells, n_bins_observed)
        Note that this is unpadded, i.e. it does not have the null cell included.
        This function is supposed to take care of the padding

    :param stimulus_time_component: np.ndarray, shape (n_bins_observed, )
    :param spiking_loss_function:
    :param device:
    :param sig_0:
    :param sig_L:
    :param h0:
    :param beta:
    :param prior_term_weight:
    :param max_iter:
    :return:
    '''

    height, width = image_height_width
    n_pixels = height * width
    root_pixels = np.sqrt(n_pixels)

    y_iter_reconstruct = torch.randn(height, width, dtype=torch.float32, device=device) * sig_0
    encoding_spikes_tensor = torch.tensor(encoding_spikes, dtype=torch.float32, device=device)

    t = 1
    sig_t = sig_0
    while sig_t >= sig_L:
        print(f't={t}, sig_t={sig_t}\r', end='')
        h_t = h0 * t / (1 + h0 * (t - 1))

        # shape (1, height, width) -> (height, width)
        residual_imshape = denoiser_callable(y_iter_reconstruct[None, :, :]).squeeze(0)

        encoding_weight = sig_t * sig_t / prior_term_weight

        # shape (height, width)
        glm_loss_gradient = encoding_weight * trial_glm_model.image_gradient(y_iter_reconstruct, encoding_spikes_tensor)

        # shape (height, width)
        d_t = residual_imshape + glm_loss_gradient

        # shape (n_bins_reconstruct, )
        sig_t = torch.linalg.norm(residual_imshape, dim=(0, 1)).item() / root_pixels

        gamma_t = np.sqrt((1 - beta * h_t) ** 2 - (1 - h_t) ** 2) * sig_t

        z_t = torch.randn_like(y_iter_reconstruct)

        # make sure we only update the not-yet-converged problems
        # logic: if sig_t drops below sig_L even once, we mark that problem
        # as converged forever
        step_update = h_t * d_t + gamma_t * z_t
        y_iter_reconstruct = y_iter_reconstruct + step_update

        if t > max_iter:
            break

        t += 1

    residual_imshape = denoiser_callable(y_iter_reconstruct[None, :, :]).squeeze(0)
    final_image_retina_range = (y_iter_reconstruct + residual_imshape).detach().cpu().numpy()

    return final_image_retina_range


class TrialGLMLoss(nn.Module):
    '''
    Method 2 for computing the GLM loss / doing image reconstructions

    This method


    
    Uses autograd to compute the loss, as well as the gradient of the loss with respect to the observed spikes
        for the total GLM model (one GLM for every cell)

    Convention:
        We assume that the reconstruction algorithm "knows" exactly what the stimulus is prior to the period
            that it is trying to reconstruct (i.e. for flashed trials, we explicitly let the reconstruction
            algorithm know that the stimulus is all grey prior to the flashed image). This is so that the initial
            conditions for the GLM are reasonable, but it is a bit of cheating...

        We currently do not assume that the stimulus is space-time separable (even though it is for the flashed
            trials), for ease of implementation.
    '''

    def __init__(self,
                 stacked_spatial_filters: np.ndarray,
                 stacked_timecourse_filters: np.ndarray,
                 stacked_feedback_filters: np.ndarray,
                 stacked_coupling_filters: np.ndarray,
                 coupling_idx_sel: np.ndarray,
                 stacked_bias: np.ndarray,
                 n_bins_reconstruction: int,
                 pre_flash_stimulus: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32,
                 verbose: bool = False):
        '''

        :param stacked_spatial_filters: shape (n_cells, height, width); spatial filters, one per cell/GLM
        :param stacked_timecourse_filters: shape (n_cells, n_bins_filter); timecourse filters, one per cell/GLM
        :param stacked_feedback_filters: shape (n_cells, n_bins_filter); feedback filters, one per cell/GLM
        :param stacked_coupling_filters: shape (n_cells, max_coupled_cells, n_bins_filter); coupling_filters,
            may contain unused slots since each cell may be coupled to a different number of other cells
        :param coupling_idx_sel: shape (n_cells, max_coupled_cells); positions marked 0 correspond to unused slots, the
            first valid cell corresponds to index 1
        :param stacked_bias: shape (n_cells, 1)
        :param n_bins_reconstruction: int, number of frames that we want to reconstruct; each reconstructed frame
            should correspond to a single timebin

            n_bins_reconstruction should satisfy
                n_bins_reconstruction + n_bins_filter = n_bins_spikes

        :param pre_flash_stimulus: shape (n_bins_filter, n_pixels), the exact stimulus for the time period directly
            preceding the period that we are trying to reconstruct; used to set the initial conditions of the GLM

            For the flashed trials, this should be a matrix of zeros.

        '''
        super().__init__(verbose=verbose)

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_bins_filter = stacked_timecourse_filters.shape[1]
        self.max_coupled_cells = stacked_coupling_filters.shape[1]
        self.n_pixels = self.height * self.width
        self.n_bins_reconstruction = n_bins_reconstruction

        self.dtype = dtype
        self.spiking_loss_fn = spiking_loss_fn

        # check the correctness of the inputs
        if pre_flash_stimulus.shape[0] != self.n_bins_filter:
            raise ValueError(f'pre_flash_stimulus must have dim0={self.n_bins_filter}')

        stacked_flat_spat_filters = stacked_spatial_filters.reshape(self.n_cells, -1)

        ##### GLM parameters as torch buffers #############################################
        # shape (n_cells, n_pixels)
        self.register_buffer('stacked_flat_spat_filters', torch.tensor(stacked_flat_spat_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_timecourse_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_timecourse_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_timecourse_filters', torch.tensor(stacked_timecourse_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_feedback_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_feedback_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_feedback_filters', torch.tensor(stacked_feedback_filters, dtype=dtype))

        # shape (n_cells, max_coupled_cells, n_bins_filter)
        assert stacked_coupling_filters.shape == (self.n_cells, self.max_coupled_cells, self.n_bins_filter), \
            f'stacked_coupling_filters must have shape {(self.n_cells, self.max_coupled_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_coupling_filters', torch.tensor(stacked_coupling_filters, dtype=dtype))

        # shape (n_cells, 1)
        assert stacked_bias.shape == (self.n_cells, 1), f'stacked_bias must have shape {(self.n_cells, 1)}'
        self.register_buffer('stacked_bias', torch.tensor(stacked_bias, dtype=dtype))

        # shape (n_cells, max_coupled_cells), integer LongTensor
        assert coupling_idx_sel.shape == (self.n_cells, self.max_coupled_cells), \
            f'coupling_idx_sel must have shape {(self.n_cells, self.max_coupled_cells)}'
        self.register_buffer('coupled_sel', torch.tensor(coupling_idx_sel, dtype=torch.long))

        ##### Initial stimulus for setting initial conditions ##############################
        self.register_buffer('init_stimulus', torch.tensor(pre_flash_stimulus, dtype=dtype))

        ##### Reconstruction variable ######################################################
        # ONLY USEFUL FOR WHEN SOLVING THE OPTIMIZATION PROBLEM FOR AN MLE ESTIMATE
        # (NO PRIOR)
        # shape (n_bins_reconstruction, n_pixels)
        self.image_flattened = nn.Parameter(torch.empty((n_bins_reconstruction, self.n_pixels), dtype=dtype),
                                            requires_grad=True)
        nn.init.uniform_(self.image_flattened, a=-0.1, b=0.1)  # FIXME parameterize this

    def gen_sig(self, image_flattened: torch.Tensor, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''
        By default, we reconstruct an image for every valid convolutional shift, excluding the very last entry
            since the GLM has to be causal. This means that rather than reconstructing a single frame, we construct
            a movie.

        :param image_flattened: shape (n_bins_reconstruct = n_bins_observed - n_bins_filtered, n_pixels)
        :param observed_spikes_null_padded:  torch.Tensor, shape (n_cells, n_bins_observed)

        :return: value of the generator signal, for every bin for which we can make a prediction
        '''

        # concatenate the initial portion of the movie with the portion that we are trying to fit
        # shape (n_bins_reconstruct + n_bins_filter, n_pixels)
        concatenated_time_movie = torch.cat([self.init_stimulus, image_flattened], dim=0)

        # apply the spatial filters for every cell, for every frame
        # shape (n_cells, n_pixels) @ (n_pixels, n_bins_total)
        # -> (n_cells, n_bins_total)
        spatial_filters_applied = self.stacked_flat_spat_filters @ concatenated_time_movie.T

        # perform the convolutions with the timecourse signal
        # these are grouped convolutions, since each cell has its own timecourse filter
        # and the timecourse filters from one cell obviously has nothing to do with the timecourse filter
        # for a different cell
        # (1, n_cells, n_bins_total) \ast (n_cells, 1, n_bins_filter)
        # -> (1, n_cells, n_bins_total - n_bins_filter + 1)
        # -> (n_cells, n_bins_total - n_bins_filter + 1)
        timecourse_filters_applied = F.conv1d(spatial_filters_applied[None, :, :],
                                              self.stacked_timecourse_filters[:, None, :],
                                              groups=self.n_cells).squeeze(0)

        # perform the convolutions with the observed spikes
        # the observed spikes should be the full set
        # (1, n_cells, n_bins_total) \ast (n_cells, 1, n_bins_filter)
        # -> (1, n_cells, n_bins_total - n_bins_filter + 1)
        # -> (n_cells, n_bins_total - n_bins_filter + 1)
        feedback_filters_applied = F.conv1d(observed_spikes[None, :, :],
                                            self.stacked_feedback_filters[:, None, :],
                                            groups=self.n_cells).squeeze(0)

        # observed_spikes_null_padded has shape (n_cells + 1, n_bins_observed)
        # coupled_sel has shape (n_cells, max_coupled_cells)
        # view has shape (n_cells, max_coupled_cells, n_bins_observed)
        expanded_coupled_sel = self.coupled_sel[:, :, None].expand(-1, -1, observed_spikes.shape[1])

        observed_spikes_expanded = observed_spikes[:, None, :].expand(-1, expanded_coupled_sel.shape[1], -1)
        # shape (n_cells, max_coupled_cells, n_bins_observed)
        # some entries are all zero, which should be fine
        # FIXME we gotta test this in isolation
        relevant_coupling_spikes = torch.gather(observed_spikes_expanded, 0, expanded_coupled_sel)

        # we need to do a reshape to apply conv1d in a straightforward way
        # shape (n_cells * max_coupled_cells, n_bins_observed)
        relevant_coupling_spikes_flat = relevant_coupling_spikes.reshape(-1, relevant_coupling_spikes.shape[2])

        # shape (n_cells * max_coupled_cells, n_bins_filter)
        coupling_filters_flat = self.stacked_coupling_filters.reshape(-1, self.stacked_coupling_filters.shape[2])

        # (1, n_cells * max_coupled_cells, n_bins_observed) \ast (n_cells * max_coupled_cells, 1, n_bins_filter)
        # -> (1, n_cells * max_coupled_cells, n_bins_observed - n_bins_filter + 1)
        # -> (n_cells * max_coupled_cells, n_bins_observed - n_bins_filter + 1)
        flat_coupling_conv = F.conv1d(relevant_coupling_spikes_flat[None, :, :],
                                      coupling_filters_flat[:, None, :],
                                      groups=(self.n_cells * self.max_coupled_cells)).squeeze(0)

        # -> (n_cells, max_coupled_cells, n_bins_observed - n_bins_filter + 1)
        coupling_conv_all = flat_coupling_conv.reshape(self.n_cells, self.max_coupled_cells, -1)

        # -> (n_cells, n_bins_observed - n_bins_filter + 1)
        coupling_filters_applied = torch.sum(coupling_conv_all, dim=1)

        generator_signal = coupling_filters_applied + feedback_filters_applied \
                           + timecourse_filters_applied + self.stacked_bias

        return generator_signal

    def image_loss(self, image_flattened: torch.Tensor, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param image_flattened: shape (n_bins_reconstruct = n_bins_observed - n_bins_filtered, n_pixels)
        :param observed_spikes_null_padded:  torch.Tensor, shape (n_cells, n_bins_observed), first row is the
            NULL cell with no spikes observed
        '''

        generator_signal = self.gen_sig(image_flattened, observed_spikes)

        loss_per_timestep = self.spiking_loss_fn(generator_signal[:, :-1],
                                                 observed_spikes[:, self.n_bins_filter:])
        return loss_per_timestep

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        image_flattened = args[self.IMAGE_IDX_ARGS]
        padded_observed_spikes = kwargs[self.PADDED_OBSERVED_SPIKES_KWARGS]

        # shape (n_bins_reconstructed, )
        loss_per_bin = self.image_loss(image_flattened, padded_observed_spikes)

        total_loss = torch.sum(loss_per_bin, dim=0)

        return total_loss

    def image_gradient(self, image_imshape: torch.Tensor, observed_spikes: torch.Tensor) \
            -> torch.Tensor:
        image_flattened = image_imshape.reshape(-1, self.height * self.width)
        image_flattened.requires_grad_(True)

        # shape (n_bins_reconstructed, )
        loss_per_bin = self.image_loss(image_flattened, observed_spikes)

        total_loss = torch.sum(loss_per_bin, dim=0)

        gradient_image_flat, = autograd.grad(total_loss, image_flattened)

        gradient_imshape = gradient_image_flat.reshape(-1, self.height, self.width)

        return -gradient_imshape


def trial_glm_nn_movie_style_prior_sampling(
        denoiser_callable: Callable[[torch.Tensor], torch.Tensor],
        glm_model_params: PackedGLMTensors,
        encoding_spikes: np.ndarray,
        prior_to_reconstructed_image: np.ndarray,
        spiking_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device,
        sig_0: float = 1.0,
        sig_L: float = 0.01,
        h0: float = 0.01,
        beta: float = 0.01,
        prior_term_weight: float = 1.0,
        max_iter: int = 1000) -> np.ndarray:
    '''
    Implementation note: This function owns the WIP reconstructed image, and does not
        use the image_flattened attribute of glm_model

    Implementation note: Since we are reconstructing a series of images all at the same time,
        we need to have the option of partial updates in case some of the images converge
        faster than other images

    :param denoiser_callable:
    :param glm_model:
    :param encoding_spikes: np.ndarray, shape (n_cells, n_bins_observed)
        Note that this is unpadded, i.e. it does not have the null cell included.
        This function is supposed to take care of the padding

    :param prior_to_reconstructed_image: np.ndarray, shape (n_bins_filter, height, width)
    :param device:
    :param sig_0:
    :param sig_L:
    :param h0:
    :param beta:
    :param prior_term_weight:
    :param max_iter:
    :return:
    '''

    n_bins_filter, height, width = prior_to_reconstructed_image.shape
    n_pixels = height * width
    root_pixels = np.sqrt(n_pixels)

    n_cells, n_bins_observed = encoding_spikes.shape

    n_bins_reconstruct = n_bins_observed - n_bins_filter

    y_iter_reconstruct = torch.randn(n_bins_reconstruct, height, width,
                                     dtype=torch.float32, device=device) * sig_0

    encoding_spikes_tensor = torch.tensor(encoding_spikes, dtype=torch.float32, device=device)

    trial_glm_model = TrialGLMLoss(
        glm_model_params.spatial_filters,
        glm_model_params.timecourse_filters,
        glm_model_params.feedback_filters,
        glm_model_params.coupling_filters,
        glm_model_params.coupling_indices,
        glm_model_params.bias,
        n_bins_reconstruct,
        prior_to_reconstructed_image.reshape(n_bins_filter, -1),
        spiking_loss_function,
        dtype=torch.float32).to(device)  # type: TrialGLMLoss

    t = 1
    sig_t = torch.ones((n_bins_reconstruct,), dtype=torch.float32, device=device) * sig_0
    not_yet_converged = torch.ones((n_bins_reconstruct,), dtype=torch.bool, device=device)
    while torch.any(not_yet_converged):
        print(f't={t}, sig_t={torch.max(sig_t).item()}\r', end='')
        h_t = h0 * t / (1 + h0 * (t - 1))

        # shape (n_bins_reconstruct, height, width)
        residual_imshape = denoiser_callable(y_iter_reconstruct)

        encoding_weight = sig_t * sig_t / prior_term_weight

        # shape (n_bins_reconstruct, height, width)
        glm_loss_gradient = encoding_weight[:, None, None] * \
                            trial_glm_model.image_gradient(y_iter_reconstruct, encoding_spikes_tensor)

        # shape (n_bins_reconstruct, height, width)
        d_t = residual_imshape + glm_loss_gradient

        # shape (n_bins_reconstruct, )
        sig_t = torch.linalg.norm(residual_imshape, dim=(1, 2)) / root_pixels

        # shape (n_bins_reconstruct, )
        gamma_t = np.sqrt((1 - beta * h_t) ** 2 - (1 - h_t) ** 2) * sig_t

        z_t = torch.randn_like(y_iter_reconstruct)

        # make sure we only update the not-yet-converged problems
        # logic: if sig_t drops below sig_L even once, we mark that problem
        # as converged forever
        step_all = h_t * d_t + gamma_t[:, None, None] * z_t
        step_update = not_yet_converged.float()[:, None, None] * step_all
        y_iter_reconstruct = y_iter_reconstruct + step_update

        not_yet_converged = not_yet_converged | (sig_t <= sig_L)

        if t > max_iter:
            break

        t += 1

    residual_imshape = denoiser_callable(y_iter_reconstruct)
    final_image_retina_range = (y_iter_reconstruct + residual_imshape).detach().cpu().numpy()
    # final_image_retina_range = (y_iter_reconstruct).detach().cpu().numpy()

    return final_image_retina_range


class Gaussian1FPriorReconstruction(SingleUnconstrainedProblem,
                                    KnownSeparableTrialGLMLoss_Precompute):

    def __init__(self,
                 stacked):
        pass

    pass
