from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict, Union, Callable, List, Optional, Any

from convex_solver_base.optim_base import SingleUnconstrainedProblem, SingleProxProblem
from convex_solver_base.prox_optim import ProxSolverParams, single_prox_solve


class AlternatingOptMixin:

    def clone_parameters_model(self, coord_desc_other: 'AlternatingOptMixin') -> None:
        raise NotImplementedError

    def return_parameters_np(self) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError


class TimecourseFitGLM(SingleProxProblem, AlternatingOptMixin):
    '''
    New optimization strategy to maximize computational efficiency

    We keep the original alternating optimization, where we alternate between
        (1) Holding the timecourse fixed, optimize the spatial, feedback, and
            coupling filters
        (2) Holding the spatial filter fixfed, optimize the timecourse, feedback,
            and coupling filters

    However, we make the following modifications to reduce the number of
        matrix multiplications involved (we don't do parallelization, since
        the coupling cells become very complicated and probably won't fit
        comfortably on GPU)

    * For the timecourse optimization step, we first do a single matrix multiplication
        of the stimulus image with the fixed spatial stimulus filter

    '''

    FILT_SPAT_MOVIE_KWARGS = 'spat_filt_movie'
    FILT_TIME_MOVIE_KWARGS = 'time_basis_filt_movie'
    CELL_SPIKE_KWARGS = 'binned_spikes_cell'
    FILT_FEEDBACK_KWARGS = 'filtered_feedback'
    FILT_COUPLING_KWARGS = 'filtered_coupling'

    COUPLING_IDX_ARGS = 0
    FEEDBACK_IDX_ARGS = 1
    TIME_FILTER_IDX_ARGS = 2
    BIAS_IDX_ARGS = 3
    COUPLING_AUX_IDX_ARGS = 4

    def __init__(self,
                 n_basis_stim_time: int,
                 n_basis_feedback: int,
                 n_basis_coupling: int,
                 n_coupled_cells: int,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 group_sparse_reg_lambda: float = 0.0,
                 dtype: torch.dtype = torch.float32,
                 stim_time_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_feedback_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_bias: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_norm: Optional[Union[np.ndarray, torch.Tensor]] = None):

        super().__init__()

        self.n_basis_stim_time = n_basis_stim_time
        self.n_basis_feedback = n_basis_feedback
        self.n_basis_coupling = n_basis_coupling
        self.n_coupled_cells = n_coupled_cells

        self.loss_callable = loss_callable

        self.group_sparse_reg_lambda = group_sparse_reg_lambda

        # OPT VARIABLE 0: coupling_w, shape (n_coupled_cells, n_basis_coupling)
        if init_coupling_w is not None:
            if init_coupling_w.shape != (n_coupled_cells, n_basis_coupling):
                raise ValueError(f"init_coupling_w must have shape {(n_coupled_cells, n_basis_coupling)}")
            if isinstance(init_coupling_w, np.ndarray):
                self.coupling_w = nn.Parameter(torch.tensor(init_coupling_w, dtype=dtype), requires_grad=True)
            else:
                self.coupling_w = nn.Parameter(init_coupling_w.detach().clone(), requires_grad=True)
        else:
            self.coupling_w = nn.Parameter(torch.empty((n_coupled_cells, n_basis_coupling), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.coupling_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 1, feedback_w, shape (1, n_basis_feedback)
        if init_feedback_w is not None:
            if init_feedback_w.shape != (1, n_basis_feedback):
                raise ValueError(f"init_feedback_w must have shape {(1, n_basis_feedback)}")
            if isinstance(init_feedback_w, np.ndarray):
                self.feedback_w = nn.Parameter(torch.tensor(init_feedback_w, dtype=dtype), requires_grad=True)
            else:
                self.feedback_w = nn.Parameter(init_feedback_w.detach().clone(), requires_grad=True)
        else:
            self.feedback_w = nn.Parameter(torch.empty((1, n_basis_feedback), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.feedback_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 2, shape (1, n_basis_stim_time)
        if stim_time_init_guess is not None:
            if stim_time_init_guess.shape != (1, n_basis_stim_time):
                raise ValueError("stim_time_init_guess must be (1, {0})".format(n_basis_stim_time))
            if isinstance(stim_time_init_guess, np.ndarray):
                self.stim_time_w = nn.Parameter(torch.tensor(stim_time_init_guess, dtype=dtype),
                                                requires_grad=True)
            else:
                self.stim_time_w = nn.Parameter(stim_time_init_guess.detach().clone(),
                                                requires_grad=True)
        else:
            self.stim_time_w = nn.Parameter(torch.empty((1, n_basis_stim_time), dtype=dtype),
                                            requires_grad=True)
            nn.init.uniform_(self.stim_time_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 3, bias, shape (1, )
        if init_bias is not None:
            if init_bias.shape != (1,):
                raise ValueError(f"init_bias must have shape {(1,)}")
            if isinstance(init_bias, np.ndarray):
                self.bias = nn.Parameter(torch.tensor(init_bias, dtype=dtype), requires_grad=True)
            else:
                self.bias = nn.Parameter(init_bias.detach().clone(), requires_grad=True)
        else:
            self.bias = nn.Parameter(torch.empty((1,), dtype=dtype), requires_grad=True)
            nn.init.uniform_(self.bias, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 4 (this one is for the coupling filter group sparsity penalty)
        # shape (n_coupled_cells, )
        if init_coupling_norm is not None:
            if init_coupling_norm.shape != (n_coupled_cells,):
                raise ValueError(f"init_coupling_norm must have shape {(n_coupled_cells,)}")
            if isinstance(init_coupling_norm, np.ndarray):
                self.coupling_filter_norm = nn.Parameter(torch.tensor(init_coupling_norm, dtype=dtype),
                                                         requires_grad=True)
            else:
                self.coupling_filter_norm = nn.Parameter(init_coupling_norm.detach().clone(),
                                                         requires_grad=True)
        else:
            self.coupling_filter_norm = nn.Parameter(torch.empty((self.coupling_w.shape[0],), dtype=dtype),
                                                     requires_grad=True)
            nn.init.uniform_(self.coupling_filter_norm, a=-1e-2, b=1e-2)

    def _stimulus_contrib_gensig(self, *args, **kwargs) -> torch.Tensor:

        # shape (1, n_basis_stim_time)
        time_filters = args[self.TIME_FILTER_IDX_ARGS]

        # shape (batch, )
        spatial_filter_applied = kwargs[self.FILT_SPAT_MOVIE_KWARGS]

        # shape (n_basis_stim_time, n_bins - n_bins_filter + 1)
        time_stimulus = kwargs[self.FILT_TIME_MOVIE_KWARGS]

        # shape (1, n_basis_stim_time) @ (n_basis_stim_time, n_bins - n_bins_filter + 1)
        # -> (1, n_bins - n_bins_filter + 1)  -> (n_bins - n_bins_filter + 1, )
        time_filter_applied = (time_filters @ time_stimulus).squeeze(0)

        # shape (batch, 1) @ (1, n_bins - n_bins_filter + 1)
        # -> (batch, n_bins - n_bins_filter + 1)
        spacetime_filter_applied = spatial_filter_applied[:, None] @ time_filter_applied[None, :]

        return spacetime_filter_applied

    def _coupling_feedback_contrib_gensig(self, *args, **kwargs) -> torch.Tensor:
        # shape (n_coupled_cells, n_basis_coupling)
        coupling_filt_w = args[self.COUPLING_IDX_ARGS]

        # shape (1, n_basis_feedback)
        feedback_filt_w = args[self.FEEDBACK_IDX_ARGS]

        # shape (batch, n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1)
        coupling_convolved = kwargs[self.FILT_COUPLING_KWARGS]

        # shape (batch, 1, n_basis_feedback, n_bins_total - n_bins_filter + 1)
        feedback_convolved = kwargs[self.FILT_FEEDBACK_KWARGS]

        # shape (1, n_coupled_cells, 1, n_basis_coupling) @
        #   (batch, n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1)
        # -> (batch, n_coupled_cells, 1, n_bins_total - n_bins_filter + 1)
        # -> (batch, n_coupled_cells, n_bins_total - n_bins_filter + 1)
        coupling_filt_applied = (coupling_filt_w[None, :, None, :] @ coupling_convolved).squeeze(2)

        # shape (1, 1, 1, n_basis_feedback) @
        #   (batch, 1, n_basis_feedback, n_bins_total - n_bins_filter + 1)
        # -> (batch, 1, 1, n_bins_total - n_bins_filter + 1)
        # -> (batch, n_bins_total - n_bins_filter + 1)
        feedback_filt_applied = (feedback_filt_w[None, :, None, :] @ feedback_convolved).squeeze(2).squeeze(1)

        return torch.sum(coupling_filt_applied, dim=1) + feedback_filt_applied

    def gen_sig(self, *args, **kwargs) -> torch.Tensor:
        bias = args[self.BIAS_IDX_ARGS]

        stimulus_contribution = self._stimulus_contrib_gensig(*args, **kwargs)
        feedback_coupling_contribution = self._coupling_feedback_contrib_gensig(*args, **kwargs)

        gen_sig = bias + stimulus_contribution + feedback_coupling_contribution
        return gen_sig

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:

        # shape (batch, n_bins_total - n_bins_filter + 1)
        binned_spikes = kwargs[self.CELL_SPIKE_KWARGS]
        coupling_filter_norm_var = args[self.COUPLING_AUX_IDX_ARGS]

        gen_sig = self.gen_sig(*args, **kwargs)

        spiking_loss = self.loss_callable(gen_sig[:, :-1], binned_spikes[:, 1:])

        if self.group_sparse_reg_lambda != 0.0:
            regularization_penalty = self.group_sparse_reg_lambda * torch.sum(coupling_filter_norm_var)
            return spiking_loss + regularization_penalty
        return spiking_loss

    def prox_project_variables(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        '''
        The only not-smooth part of the penalty that we need to project is the
            neighboring cell coupling filter coefficients, and the neighboring cell
            coupling norm auxiliary variables.

        Everything else we can just pass through

        :param args:
        :param kwargs:
        :return:
        '''

        # shape (n_coupled_cells, n_coupling_basis)
        coupling_filter_w = args[self.COUPLING_IDX_ARGS]

        # shape (n_coupled_cells, )
        coupling_auxvar = args[self.COUPLING_AUX_IDX_ARGS]

        # passthrough variables
        feedback_filter_w = args[self.FEEDBACK_IDX_ARGS]
        timecourse_w = args[self.TIME_FILTER_IDX_ARGS]
        bias_w = args[self.BIAS_IDX_ARGS]

        with torch.no_grad():
            # shape (n_coupled_cells, )
            coupling_norm = torch.linalg.norm(coupling_filter_w, dim=1)

            # shape (n_coupled_cells, ), binary-valued
            less_than = (coupling_norm <= coupling_auxvar)
            less_than_neg = (coupling_norm <= (-coupling_auxvar))

            # shape (n_coupled_cells, )
            scale_mult_numerator = (coupling_norm + coupling_auxvar)
            scale_mult = (scale_mult_numerator / (2 * coupling_norm))
            scale_mult[less_than] = 1.0
            scale_mult[less_than_neg] = 0.0

            # shape (n_coupled_cells, n_coupling_filters) * (n_coupled_cells, 1)
            # -> (n_coupled_cells, n_coupling filters)
            coupling_w_prox_applied = coupling_filter_w * scale_mult[:, None]

            auxvar_prox = scale_mult_numerator / 2.0
            auxvar_prox[less_than] = coupling_auxvar[less_than]
            auxvar_prox[less_than_neg] = 0.0

        return coupling_w_prox_applied, feedback_filter_w, timecourse_w, bias_w, auxvar_prox

    def clone_parameters_model(self, coord_desc_other: 'AlternatingOptMixin') -> None:

        self.coupling_w.data[:] = coord_desc_other.coupling_w.data[:]
        self.feedback_w.data[:] = coord_desc_other.feedback_w.data[:]
        self.bias.data[:] = coord_desc_other.bias.data[:]
        self.coupling_filter_norm.data[:] = coord_desc_other.coupling_filter_norm.data[:]

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        return (
            self.coupling_w.detach().cpu().numpy(),
            self.feedback_w.detach().cpu().numpy(),
            self.stim_time_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy(),
            self.coupling_filter_norm.detach().cpu().numpy()
        )

    def return_parameters_torch(self) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        return (
            self.coupling_w.detach().clone(),
            self.feedback_w.detach().clone(),
            self.stim_time_w.detach().clone(),
            self.bias.detach().clone(),
            self.coupling_filter_norm.detach().clone()
        )


class SpatialFitGLM(SingleProxProblem, AlternatingOptMixin):
    FILT_SPAT_MOVIE_KWARGS = 'raw_stim_frame'
    FILT_TIME_MOVIE_KWARGS = 'time_filt_movie'
    CELL_SPIKE_KWARGS = 'binned_spikes_cell'
    FILT_FEEDBACK_KWARGS = 'filtered_feedback'
    FILT_COUPLING_KWARGS = 'filtered_coupling'

    COUPLING_IDX_ARGS = 0
    FEEDBACK_IDX_ARGS = 1
    SPAT_FILTER_IDX_ARGS = 2
    BIAS_IDX_ARGS = 3
    COUPLING_AUX_IDX_ARGS = 4

    def __init__(self,
                 n_pixels: int,
                 n_basis_feedback: int,
                 n_basis_coupling: int,
                 n_coupled_cells: int,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 spat_filter_sparsity_lambda: float = 0.0,
                 group_sparse_reg_lambda: float = 0.0,
                 dtype: torch.dtype = torch.float32,
                 stim_spat_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_feedback_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_bias: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_norm: Optional[Union[np.ndarray, torch.Tensor]] = None):

        super().__init__()

        self.n_pixels = n_pixels

        self.n_basis_feedback = n_basis_feedback
        self.n_basis_coupling = n_basis_coupling
        self.n_coupled_cells = n_coupled_cells

        self.loss_callable = loss_callable
        self.spat_filter_sparsity_lambda = spat_filter_sparsity_lambda
        self.group_sparse_reg_lambda = group_sparse_reg_lambda

        # OPT VARIABLE 0: coupling_w, shape (n_coupled_cells, n_basis_coupling)
        if init_coupling_w is not None:
            if init_coupling_w.shape != (n_coupled_cells, n_basis_coupling):
                raise ValueError(f"init_coupling_w must have shape {(n_coupled_cells, n_basis_coupling)}")
            if isinstance(init_coupling_w, np.ndarray):
                self.coupling_w = nn.Parameter(torch.tensor(init_coupling_w, dtype=dtype), requires_grad=True)
            else:
                self.coupling_w = nn.Parameter(init_coupling_w.detach().clone(), requires_grad=True)
        else:
            self.coupling_w = nn.Parameter(torch.empty((n_coupled_cells, n_basis_coupling), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.coupling_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 1, feedback_w, shape (1, n_basis_feedback)
        if init_feedback_w is not None:
            if init_feedback_w.shape != (1, n_basis_feedback):
                raise ValueError(f"init_feedback_w must have shape {(1, n_basis_feedback)}")
            if isinstance(init_feedback_w, np.ndarray):
                self.feedback_w = nn.Parameter(torch.tensor(init_feedback_w, dtype=dtype), requires_grad=True)
            else:
                self.feedback_w = nn.Parameter(init_feedback_w.detach().clone(), requires_grad=True)
        else:
            self.feedback_w = nn.Parameter(torch.empty((1, n_basis_feedback), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.feedback_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 2, shape (1, n_basis_stim_spat)
        if stim_spat_init_guess is not None:
            if stim_spat_init_guess.shape != (self.n_pixels,):
                raise ValueError(f"stim_spat_init_guess must be ({self.n_pixels}, )")
            if isinstance(stim_spat_init_guess, np.ndarray):
                self.stim_spat_w = nn.Parameter(torch.tensor(stim_spat_init_guess, dtype=dtype),
                                                requires_grad=True)
            else:
                self.stim_spat_w = nn.Parameter(stim_spat_init_guess.detach().clone(),
                                                requires_grad=True)
        else:
            self.stim_spat_w = nn.Parameter(torch.empty((self.n_pixels,), dtype=dtype),
                                            requires_grad=True)
            nn.init.uniform_(self.stim_spat_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 3, bias, shape (1, )
        if init_bias is not None:
            if init_bias.shape != (1,):
                raise ValueError(f"init_bias must have shape {(1,)}")
            if isinstance(init_bias, np.ndarray):
                self.bias = nn.Parameter(torch.tensor(init_bias, dtype=dtype), requires_grad=True)
            else:
                self.bias = nn.Parameter(init_bias.detach().clone(), requires_grad=True)
        else:
            self.bias = nn.Parameter(torch.empty((1,), dtype=dtype), requires_grad=True)
            nn.init.uniform_(self.bias, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 4 (this one is for the coupling filter group sparsity penalty)
        # shape (n_coupled_cells, )

        if init_coupling_norm is not None:
            if init_coupling_norm.shape != (n_coupled_cells,):
                raise ValueError(f"init_coupling_norm must have shape {(n_coupled_cells,)}")
            if isinstance(init_coupling_norm, np.ndarray):
                self.coupling_filter_norm = nn.Parameter(torch.tensor(init_coupling_norm, dtype=dtype),
                                                         requires_grad=True)
            else:
                self.coupling_filter_norm = nn.Parameter(init_coupling_norm.detach().clone(),
                                                         requires_grad=True)
        else:
            self.coupling_filter_norm = nn.Parameter(torch.empty((self.coupling_w.shape[0],), dtype=dtype),
                                                     requires_grad=True)
            nn.init.uniform_(self.coupling_filter_norm, a=-1e-2, b=1e-2)

    def _stimulus_contrib_gensig(self, *args, **kwargs) -> torch.Tensor:
        # shape (n_pixels, )
        spatial_filters = args[self.SPAT_FILTER_IDX_ARGS]

        # shape (batch, n_pixels)
        spat_stimulus = kwargs[self.FILT_SPAT_MOVIE_KWARGS]

        # shape (n_bins - n_bins_filter + 1, )
        time_filter_applied = kwargs[self.FILT_TIME_MOVIE_KWARGS]

        # shape (batch, n_pixels) @ (n_pixels, 1)
        # -> (batch, 1) -> (batch, )
        spatial_filter_applied = (spat_stimulus @ spatial_filters[:, None]).squeeze(1)

        # shape (batch, 1) @ (1, n_bins - n_bins_filter + 1)
        # -> (batch, n_bins - n_bins_filter + 1)
        spacetime_filter_applied = spatial_filter_applied[:, None] @ time_filter_applied[None, :]

        return spacetime_filter_applied

    def _coupling_feedback_contrib_gensig(self, *args, **kwargs) -> torch.Tensor:
        # shape (n_coupled_cells, n_basis_coupling)
        coupling_filt_w = args[self.COUPLING_IDX_ARGS]

        # shape (1, n_basis_feedback)
        feedback_filt_w = args[self.FEEDBACK_IDX_ARGS]

        # shape (batch, n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1)
        coupling_convolved = kwargs[self.FILT_COUPLING_KWARGS]

        # shape (batch, 1, n_basis_feedback, n_bins_total - n_bins_filter + 1)
        feedback_convolved = kwargs[self.FILT_FEEDBACK_KWARGS]

        # shape (1, n_coupled_cells, 1, n_basis_coupling) @
        #   (batch, n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1)
        # -> (batch, n_coupled_cells, 1, n_bins_total - n_bins_filter + 1)
        # -> (batch, n_coupled_cells, n_bins_total - n_bins_filter + 1)
        coupling_filt_applied = (coupling_filt_w[None, :, None, :] @ coupling_convolved).squeeze(2)

        # shape (1, 1, 1, n_basis_feedback) @
        #   (batch, 1, n_basis_feedback, n_bins_total - n_bins_filter + 1)
        # -> (batch, 1, 1, n_bins_total - n_bins_filter + 1)
        # -> (batch, n_bins_total - n_bins_filter + 1)
        feedback_filt_applied = (feedback_filt_w[None, :, None, :] @ feedback_convolved).squeeze(2).squeeze(1)

        return torch.sum(coupling_filt_applied, dim=1) + feedback_filt_applied

    def gen_sig(self, *args, **kwargs) -> torch.Tensor:
        bias = args[self.BIAS_IDX_ARGS]

        stimulus_contribution = self._stimulus_contrib_gensig(*args, **kwargs)
        feedback_coupling_contribution = self._coupling_feedback_contrib_gensig(*args, **kwargs)

        gen_sig = bias + stimulus_contribution + feedback_coupling_contribution
        return gen_sig

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        binned_spikes = kwargs[self.CELL_SPIKE_KWARGS]
        coupling_filter_norm_var = args[self.COUPLING_AUX_IDX_ARGS]
        gen_sig = self.gen_sig(*args, **kwargs)

        spiking_loss = self.loss_callable(gen_sig[:, :-1], binned_spikes[:, 1:])
        if self.group_sparse_reg_lambda != 0.0:
            regularization_penalty = self.group_sparse_reg_lambda * torch.sum(coupling_filter_norm_var)
            return spiking_loss + regularization_penalty

        return spiking_loss

    def prox_project_variables(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        '''
        The only not-smooth part of the penalty that we need to project is the
            neighboring cell coupling filter coefficients, and the neighboring cell
            coupling norm auxiliary variables.

        Everything else we can just pass through

        :param args:
        :param kwargs:
        :return:
        '''

        # shape (n_coupled_cells, n_coupling_basis)
        coupling_filter_w = args[self.COUPLING_IDX_ARGS]

        # shape (n_coupled_cells, )
        coupling_auxvar = args[self.COUPLING_AUX_IDX_ARGS]

        # shape (n_pixels, )
        spat_filter = args[self.SPAT_FILTER_IDX_ARGS]

        # passthrough variables
        feedback_filter_w = args[self.FEEDBACK_IDX_ARGS]
        bias_w = args[self.BIAS_IDX_ARGS]

        with torch.no_grad():
            # shape (n_coupled_cells, )
            coupling_norm = torch.linalg.norm(coupling_filter_w, dim=1)

            # shape (n_coupled_cells, ), binary-valued
            less_than = (coupling_norm <= coupling_auxvar)
            less_than_neg = (coupling_norm <= (-coupling_auxvar))

            # shape (n_coupled_cells, )
            scale_mult_numerator = (coupling_norm + coupling_auxvar)
            scale_mult = (scale_mult_numerator / (2.0 * coupling_norm))
            scale_mult[less_than] = 1.0
            scale_mult[less_than_neg] = 0.0

            # shape (n_coupled_cells, n_coupling_filters) * (n_coupled_cells, 1)
            # -> (n_coupled_cells, n_coupling filters)
            coupling_w_prox_applied = coupling_filter_w * scale_mult[:, None]

            auxvar_prox = scale_mult_numerator / 2.0
            auxvar_prox[less_than] = coupling_auxvar[less_than]
            auxvar_prox[less_than_neg] = 0.0

            if self.spat_filter_sparsity_lambda != 0.0:
                spat_filter = torch.clamp_min_(spat_filter - self.spat_filter_sparsity_lambda, 0.0) \
                              - torch.clamp_min_(-spat_filter - self.spat_filter_sparsity_lambda, 0.0)

        return coupling_w_prox_applied, feedback_filter_w, spat_filter, bias_w, auxvar_prox

    def clone_parameters_model(self, coord_desc_other: 'AlternatingOptMixin') -> None:

        self.coupling_w.data[:] = coord_desc_other.coupling_w.data[:]
        self.feedback_w.data[:] = coord_desc_other.feedback_w.data[:]
        self.bias.data[:] = coord_desc_other.bias.data[:]
        self.coupling_filter_norm.data[:] = coord_desc_other.coupling_filter_norm.data[:]

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        return (
            self.coupling_w.detach().cpu().numpy(),
            self.feedback_w.detach().cpu().numpy(),
            self.stim_spat_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy(),
            self.coupling_filter_norm.detach().cpu().numpy()
        )

    def return_parameters_torch(self) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        return (
            self.coupling_w.detach().clone(),
            self.feedback_w.detach().clone(),
            self.stim_spat_w.detach().clone(),
            self.bias.detach().clone(),
            self.coupling_filter_norm.detach().clone()
        )


class SpatialScaleGLM(SingleProxProblem):
    FILT_SPAT_MOVIE_KWARGS = 'raw_stim_frame'
    FILT_TIME_MOVIE_KWARGS = 'time_filt_movie'
    CELL_SPIKE_KWARGS = 'binned_spikes_cell'
    FILT_FEEDBACK_KWARGS = 'filtered_feedback'
    FILT_COUPLING_KWARGS = 'filtered_coupling'

    COUPLING_IDX_ARGS = 0
    FEEDBACK_IDX_ARGS = 1
    SPAT_FILTER_SCALE_IDX_ARGS = 2
    BIAS_IDX_ARGS = 3
    COUPLING_AUX_IDX_ARGS = 4

    def __init__(self,
                 n_pixels: int,
                 sta_prior_filter: Union[np.ndarray, torch.Tensor],
                 n_basis_feedback: int,
                 n_basis_coupling: int,
                 n_coupled_cells: int,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 group_sparse_reg_lambda: float = 0.0,
                 dtype: torch.dtype = torch.float32,
                 init_feedback_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_bias: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_norm: Optional[Union[np.ndarray, torch.Tensor]] = None):

        super().__init__()

        self.n_pixels = n_pixels

        self.n_basis_feedback = n_basis_feedback
        self.n_basis_coupling = n_basis_coupling
        self.n_coupled_cells = n_coupled_cells

        self.loss_callable = loss_callable
        self.group_sparse_reg_lambda = group_sparse_reg_lambda

        # Constant prior filter
        if sta_prior_filter.shape != (self.n_pixels,):
            raise ValueError(f"sta_prior_filter must be ({self.n_pixels}, )")
        if isinstance(sta_prior_filter, np.ndarray):
            self.register_buffer('sta_prior', torch.tensor(sta_prior_filter, dtype=dtype))
        else:
            self.register_buffer('sta_prior', sta_prior_filter.detach().clone())

        # OPT VARIABLE 0: coupling_w, shape (n_coupled_cells, n_basis_coupling)
        if init_coupling_w is not None:
            if init_coupling_w.shape != (n_coupled_cells, n_basis_coupling):
                raise ValueError(f"init_coupling_w must have shape {(n_coupled_cells, n_basis_coupling)}")
            if isinstance(init_coupling_w, np.ndarray):
                self.coupling_w = nn.Parameter(torch.tensor(init_coupling_w, dtype=dtype), requires_grad=True)
            else:
                self.coupling_w = nn.Parameter(init_coupling_w.detach().clone(), requires_grad=True)
        else:
            self.coupling_w = nn.Parameter(torch.empty((n_coupled_cells, n_basis_coupling), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.coupling_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 1, feedback_w, shape (1, n_basis_feedback)
        if init_feedback_w is not None:
            if init_feedback_w.shape != (1, n_basis_feedback):
                raise ValueError(f"init_feedback_w must have shape {(1, n_basis_feedback)}")
            if isinstance(init_feedback_w, np.ndarray):
                self.feedback_w = nn.Parameter(torch.tensor(init_feedback_w, dtype=dtype), requires_grad=True)
            else:
                self.feedback_w = nn.Parameter(init_feedback_w.detach().clone(), requires_grad=True)
        else:
            self.feedback_w = nn.Parameter(torch.empty((1, n_basis_feedback), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.feedback_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 2, shape (1, )
        self.stim_spat_w = nn.Parameter(torch.empty((1,), dtype=dtype),
                                        requires_grad=True)
        nn.init.uniform_(self.stim_spat_w, a=0.0, b=1e-2)

        # OPTIMIZATION VARIABLE 3, bias, shape (1, )
        if init_bias is not None:
            if init_bias.shape != (1,):
                raise ValueError(f"init_bias must have shape {(1,)}")
            if isinstance(init_bias, np.ndarray):
                self.bias = nn.Parameter(torch.tensor(init_bias, dtype=dtype), requires_grad=True)
            else:
                self.bias = nn.Parameter(init_bias.detach().clone(), requires_grad=True)
        else:
            self.bias = nn.Parameter(torch.empty((1,), dtype=dtype), requires_grad=True)
            nn.init.uniform_(self.bias, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 4 (this one is for the coupling filter group sparsity penalty)
        # shape (n_coupled_cells, )
        if init_coupling_norm is not None:
            if init_coupling_norm.shape != (n_coupled_cells,):
                raise ValueError(f"init_coupling_norm must have shape {(n_coupled_cells,)}")
            if isinstance(init_coupling_norm, np.ndarray):
                self.coupling_filter_norm = nn.Parameter(torch.tensor(init_coupling_norm, dtype=dtype),
                                                         requires_grad=True)
            else:
                self.coupling_filter_norm = nn.Parameter(init_coupling_norm.detach().clone(),
                                                         requires_grad=True)
        else:
            self.coupling_filter_norm = nn.Parameter(torch.empty((self.coupling_w.shape[0],), dtype=dtype),
                                                     requires_grad=True)
            nn.init.uniform_(self.coupling_filter_norm, a=-1e-2, b=1e-2)

    def _stimulus_contrib_gensig(self, *args, **kwargs) -> torch.Tensor:
        # shape (1, )
        spatial_filter_scale = args[self.SPAT_FILTER_SCALE_IDX_ARGS]

        # shape (batch, n_pixels)
        spat_stimulus = kwargs[self.FILT_SPAT_MOVIE_KWARGS]

        # shape (n_bins - n_bins_filter + 1, )
        time_filter_applied = kwargs[self.FILT_TIME_MOVIE_KWARGS]

        # shape (n_pixels, )
        spatial_filters = self.sta_prior * spatial_filter_scale

        # shape (batch, n_pixels) @ (n_pixels, 1)
        # -> (batch, 1) -> (batch, )
        spatial_filter_applied = (spat_stimulus @ spatial_filters[:, None]).squeeze(1)

        # shape (batch, 1) @ (1, n_bins - n_bins_filter + 1)
        # -> (batch, n_bins - n_bins_filter + 1)
        spacetime_filter_applied = spatial_filter_applied[:, None] @ time_filter_applied[None, :]

        return spacetime_filter_applied

    def _coupling_feedback_contrib_gensig(self, *args, **kwargs) -> torch.Tensor:
        # shape (n_coupled_cells, n_basis_coupling)
        coupling_filt_w = args[self.COUPLING_IDX_ARGS]

        # shape (1, n_basis_feedback)
        feedback_filt_w = args[self.FEEDBACK_IDX_ARGS]

        # shape (batch, n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1)
        coupling_convolved = kwargs[self.FILT_COUPLING_KWARGS]

        # shape (batch, 1, n_basis_feedback, n_bins_total - n_bins_filter + 1)
        feedback_convolved = kwargs[self.FILT_FEEDBACK_KWARGS]

        # shape (1, n_coupled_cells, 1, n_basis_coupling) @
        #   (batch, n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1)
        # -> (batch, n_coupled_cells, 1, n_bins_total - n_bins_filter + 1)
        # -> (batch, n_coupled_cells, n_bins_total - n_bins_filter + 1)
        coupling_filt_applied = (coupling_filt_w[None, :, None, :] @ coupling_convolved).squeeze(2)

        # shape (1, 1, 1, n_basis_feedback) @
        #   (batch, 1, n_basis_feedback, n_bins_total - n_bins_filter + 1)
        # -> (batch, 1, 1, n_bins_total - n_bins_filter + 1)
        # -> (batch, n_bins_total - n_bins_filter + 1)
        feedback_filt_applied = (feedback_filt_w[None, :, None, :] @ feedback_convolved).squeeze(2).squeeze(1)

        return torch.sum(coupling_filt_applied, dim=1) + feedback_filt_applied

    def gen_sig(self, *args, **kwargs) -> torch.Tensor:
        bias = args[self.BIAS_IDX_ARGS]

        stimulus_contribution = self._stimulus_contrib_gensig(*args, **kwargs)
        feedback_coupling_contribution = self._coupling_feedback_contrib_gensig(*args, **kwargs)

        gen_sig = bias + stimulus_contribution + feedback_coupling_contribution
        return gen_sig

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        binned_spikes = kwargs[self.CELL_SPIKE_KWARGS]
        coupling_filter_norm_var = args[self.COUPLING_AUX_IDX_ARGS]

        gen_sig = self.gen_sig(*args, **kwargs)

        spiking_loss = self.loss_callable(gen_sig[:, :-1], binned_spikes[:, 1:])

        if self.group_sparse_reg_lambda != 0.0:
            regularization_penalty = self.group_sparse_reg_lambda * torch.sum(coupling_filter_norm_var)
            return spiking_loss + regularization_penalty

        return spiking_loss

    def prox_project_variables(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        '''
        The only not-smooth part of the penalty that we need to project is the
            neighboring cell coupling filter coefficients, and the neighboring cell
            coupling norm auxiliary variables.

        Everything else we can just pass through

        :param args:
        :param kwargs:
        :return:
        '''

        # shape (n_coupled_cells, n_coupling_basis)
        coupling_filter_w = args[self.COUPLING_IDX_ARGS]

        # shape (n_coupled_cells, )
        coupling_auxvar = args[self.COUPLING_AUX_IDX_ARGS]

        # shape (1, )
        spat_filter_scale = args[self.SPAT_FILTER_SCALE_IDX_ARGS]

        # passthrough variables
        feedback_filter_w = args[self.FEEDBACK_IDX_ARGS]
        bias_w = args[self.BIAS_IDX_ARGS]

        with torch.no_grad():
            # shape (n_coupled_cells, )
            coupling_norm = torch.linalg.norm(coupling_filter_w, dim=1)

            # shape (n_coupled_cells, ), binary-valued
            less_than = (coupling_norm <= coupling_auxvar)
            less_than_neg = (coupling_norm <= (-coupling_auxvar))

            # shape (n_coupled_cells, )
            scale_mult_numerator = (coupling_norm + coupling_auxvar)
            scale_mult = (scale_mult_numerator / (2.0 * coupling_norm))
            scale_mult[less_than] = 1.0
            scale_mult[less_than_neg] = 0.0

            # shape (n_coupled_cells, n_coupling_filters) * (n_coupled_cells, 1)
            # -> (n_coupled_cells, n_coupling filters)
            coupling_w_prox_applied = coupling_filter_w * scale_mult[:, None]

            auxvar_prox = scale_mult_numerator / 2.0
            auxvar_prox[less_than] = coupling_auxvar[less_than]
            auxvar_prox[less_than_neg] = 0.0

        return coupling_w_prox_applied, feedback_filter_w, spat_filter_scale, bias_w, auxvar_prox

    def return_parameters_torch(self) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # shape (1, )
        spatial_filter_scale = self.stim_spat_w

        # shape (n_pixels, )
        spatial_filters = self.sta_prior * spatial_filter_scale

        return (
            self.coupling_w.detach().clone(),
            self.feedback_w.detach().clone(),
            spatial_filters.detach().clone(),
            self.bias.detach().clone(),
            self.coupling_filter_norm.detach().clone()
        )

    def return_parameters_np(self) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # shape (1, )
        spatial_filter_scale = self.stim_spat_w

        # shape (n_pixels, )
        spatial_filters = self.sta_prior * spatial_filter_scale

        return (
            self.coupling_w.detach().cpu().numpy(),
            self.feedback_w.detach().cpu().numpy(),
            spatial_filters.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy(),
            self.coupling_filter_norm.detach().cpu().numpy()
        )


class SpatialFitGLM_WithSTAPrior(SingleProxProblem, AlternatingOptMixin):
    FILT_SPAT_MOVIE_KWARGS = 'raw_stim_frame'
    FILT_TIME_MOVIE_KWARGS = 'time_filt_movie'
    CELL_SPIKE_KWARGS = 'binned_spikes_cell'
    FILT_FEEDBACK_KWARGS = 'filtered_feedback'
    FILT_COUPLING_KWARGS = 'filtered_coupling'

    COUPLING_IDX_ARGS = 0
    FEEDBACK_IDX_ARGS = 1
    SPAT_FILTER_IDX_ARGS = 2
    BIAS_IDX_ARGS = 3
    COUPLING_AUX_IDX_ARGS = 4

    def __init__(self,
                 n_pixels: int,
                 sta_prior_filter: Union[np.ndarray, torch.Tensor],
                 n_basis_feedback: int,
                 n_basis_coupling: int,
                 n_coupled_cells: int,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 prior_filter_lambda: float = 1e-1,
                 group_sparse_reg_lambda: float = 0.0,
                 spat_filter_sparsity_lambda: float = 0.0,
                 dtype: torch.dtype = torch.float32,
                 stim_spat_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_feedback_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_bias: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_norm: Optional[Union[np.ndarray, torch.Tensor]] = None):

        super().__init__()

        self.n_pixels = n_pixels

        self.n_basis_feedback = n_basis_feedback
        self.n_basis_coupling = n_basis_coupling
        self.n_coupled_cells = n_coupled_cells

        self.loss_callable = loss_callable
        self.group_sparse_reg_lambda = group_sparse_reg_lambda
        self.prior_filter_lambda = prior_filter_lambda
        self.spat_filter_sparsity_lambda = spat_filter_sparsity_lambda

        # Constant prior filter
        if sta_prior_filter.shape != (self.n_pixels,):
            raise ValueError(f"sta_prior_filter must be ({self.n_pixels}, )")
        if isinstance(sta_prior_filter, np.ndarray):
            self.register_buffer('sta_prior', torch.tensor(sta_prior_filter, dtype=dtype))
        else:
            self.register_buffer('sta_prior', sta_prior_filter.detach().clone())

        # OPT VARIABLE 0: coupling_w, shape (n_coupled_cells, n_basis_coupling)
        if init_coupling_w is not None:
            if init_coupling_w.shape != (n_coupled_cells, n_basis_coupling):
                raise ValueError(f"init_coupling_w must have shape {(n_coupled_cells, n_basis_coupling)}")
            if isinstance(init_coupling_w, np.ndarray):
                self.coupling_w = nn.Parameter(torch.tensor(init_coupling_w, dtype=dtype), requires_grad=True)
            else:
                self.coupling_w = nn.Parameter(init_coupling_w.detach().clone(), requires_grad=True)
        else:
            self.coupling_w = nn.Parameter(torch.empty((n_coupled_cells, n_basis_coupling), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.coupling_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 1, feedback_w, shape (1, n_basis_feedback)
        if init_feedback_w is not None:
            if init_feedback_w.shape != (1, n_basis_feedback):
                raise ValueError(f"init_feedback_w must have shape {(1, n_basis_feedback)}")
            if isinstance(init_feedback_w, np.ndarray):
                self.feedback_w = nn.Parameter(torch.tensor(init_feedback_w, dtype=dtype), requires_grad=True)
            else:
                self.feedback_w = nn.Parameter(init_feedback_w.detach().clone(), requires_grad=True)
        else:
            self.feedback_w = nn.Parameter(torch.empty((1, n_basis_feedback), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.feedback_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 2, shape (1, n_basis_stim_spat)
        if stim_spat_init_guess is not None:
            if stim_spat_init_guess.shape != (self.n_pixels,):
                raise ValueError(f"stim_spat_init_guess must be ({self.n_pixels}, )")
            if isinstance(stim_spat_init_guess, np.ndarray):
                self.stim_spat_w = nn.Parameter(torch.tensor(stim_spat_init_guess, dtype=dtype),
                                                requires_grad=True)
            else:
                self.stim_spat_w = nn.Parameter(stim_spat_init_guess.detach().clone(),
                                                requires_grad=True)
        else:
            self.stim_spat_w = nn.Parameter(torch.empty((self.n_pixels,), dtype=dtype),
                                            requires_grad=True)
            nn.init.uniform_(self.stim_spat_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 3, bias, shape (1, )
        if init_bias is not None:
            if init_bias.shape != (1,):
                raise ValueError(f"init_bias must have shape {(1,)}")
            if isinstance(init_bias, np.ndarray):
                self.bias = nn.Parameter(torch.tensor(init_bias, dtype=dtype), requires_grad=True)
            else:
                self.bias = nn.Parameter(init_bias.detach().clone(), requires_grad=True)
        else:
            self.bias = nn.Parameter(torch.empty((1,), dtype=dtype), requires_grad=True)
            nn.init.uniform_(self.bias, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 4 (this one is for the coupling filter group sparsity penalty)
        # shape (n_coupled_cells, )
        if init_coupling_norm is not None:
            if init_coupling_norm.shape != (n_coupled_cells,):
                raise ValueError(f"init_coupling_norm must have shape {(n_coupled_cells,)}")
            if isinstance(init_coupling_norm, np.ndarray):
                self.coupling_filter_norm = nn.Parameter(torch.tensor(init_coupling_norm, dtype=dtype),
                                                         requires_grad=True)
            else:
                self.coupling_filter_norm = nn.Parameter(init_coupling_norm.detach().clone(),
                                                         requires_grad=True)
        else:
            self.coupling_filter_norm = nn.Parameter(torch.empty((self.coupling_w.shape[0],), dtype=dtype),
                                                     requires_grad=True)
            nn.init.uniform_(self.coupling_filter_norm, a=-1e-2, b=1e-2)

    def _stimulus_contrib_gensig(self, *args, **kwargs) -> torch.Tensor:
        # shape (n_pixels, )
        spatial_filters = args[self.SPAT_FILTER_IDX_ARGS]

        # shape (batch, n_pixels)
        spat_stimulus = kwargs[self.FILT_SPAT_MOVIE_KWARGS]

        # shape (n_bins - n_bins_filter + 1, )
        time_filter_applied = kwargs[self.FILT_TIME_MOVIE_KWARGS]

        # shape (batch, n_pixels) @ (n_pixels, 1)
        # -> (batch, 1) -> (batch, )
        spatial_filter_applied = (spat_stimulus @ spatial_filters[:, None]).squeeze(1)

        # shape (batch, 1) @ (1, n_bins - n_bins_filter + 1)
        # -> (batch, n_bins - n_bins_filter + 1)
        spacetime_filter_applied = spatial_filter_applied[:, None] @ time_filter_applied[None, :]

        return spacetime_filter_applied

    def _coupling_feedback_contrib_gensig(self, *args, **kwargs) -> torch.Tensor:
        # shape (n_coupled_cells, n_basis_coupling)
        coupling_filt_w = args[self.COUPLING_IDX_ARGS]

        # shape (1, n_basis_feedback)
        feedback_filt_w = args[self.FEEDBACK_IDX_ARGS]

        # shape (batch, n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1)
        coupling_convolved = kwargs[self.FILT_COUPLING_KWARGS]

        # shape (batch, 1, n_basis_feedback, n_bins_total - n_bins_filter + 1)
        feedback_convolved = kwargs[self.FILT_FEEDBACK_KWARGS]

        # shape (1, n_coupled_cells, 1, n_basis_coupling) @
        #   (batch, n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1)
        # -> (batch, n_coupled_cells, 1, n_bins_total - n_bins_filter + 1)
        # -> (batch, n_coupled_cells, n_bins_total - n_bins_filter + 1)
        coupling_filt_applied = (coupling_filt_w[None, :, None, :] @ coupling_convolved).squeeze(2)

        # shape (1, 1, 1, n_basis_feedback) @
        #   (batch, 1, n_basis_feedback, n_bins_total - n_bins_filter + 1)
        # -> (batch, 1, 1, n_bins_total - n_bins_filter + 1)
        # -> (batch, n_bins_total - n_bins_filter + 1)
        feedback_filt_applied = (feedback_filt_w[None, :, None, :] @ feedback_convolved).squeeze(2).squeeze(1)

        return torch.sum(coupling_filt_applied, dim=1) + feedback_filt_applied

    def gen_sig(self, *args, **kwargs) -> torch.Tensor:
        bias = args[self.BIAS_IDX_ARGS]

        stimulus_contribution = self._stimulus_contrib_gensig(*args, **kwargs)
        feedback_coupling_contribution = self._coupling_feedback_contrib_gensig(*args, **kwargs)

        gen_sig = bias + stimulus_contribution + feedback_coupling_contribution
        return gen_sig

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        binned_spikes = kwargs[self.CELL_SPIKE_KWARGS]
        coupling_filter_norm_var = args[self.COUPLING_AUX_IDX_ARGS]

        spatial_filter = args[self.SPAT_FILTER_IDX_ARGS]

        gen_sig = self.gen_sig(*args, **kwargs)

        spiking_loss = self.loss_callable(gen_sig[:, :-1], binned_spikes[:, 1:])

        prior_diff = spatial_filter - self.sta_prior
        prior_norm2 = torch.sum(prior_diff * prior_diff)

        spiking_loss_with_prior_rg = spiking_loss + self.prior_filter_lambda * prior_norm2

        if self.group_sparse_reg_lambda != 0.0:
            regularization_penalty = self.group_sparse_reg_lambda * torch.sum(coupling_filter_norm_var)
            return spiking_loss_with_prior_rg + regularization_penalty

        return spiking_loss_with_prior_rg

    def prox_project_variables(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        '''
        The only not-smooth part of the penalty that we need to project is the
            neighboring cell coupling filter coefficients, and the neighboring cell
            coupling norm auxiliary variables.

        Everything else we can just pass through

        :param args:
        :param kwargs:
        :return:
        '''

        # shape (n_coupled_cells, n_coupling_basis)
        coupling_filter_w = args[self.COUPLING_IDX_ARGS]

        # shape (n_coupled_cells, )
        coupling_auxvar = args[self.COUPLING_AUX_IDX_ARGS]

        # shape (n_pixels, )
        spat_filter = args[self.SPAT_FILTER_IDX_ARGS]

        # passthrough variables
        feedback_filter_w = args[self.FEEDBACK_IDX_ARGS]
        bias_w = args[self.BIAS_IDX_ARGS]

        with torch.no_grad():
            # shape (n_coupled_cells, )
            coupling_norm = torch.linalg.norm(coupling_filter_w, dim=1)

            # shape (n_coupled_cells, ), binary-valued
            less_than = (coupling_norm <= coupling_auxvar)
            less_than_neg = (coupling_norm <= (-coupling_auxvar))

            # shape (n_coupled_cells, )
            scale_mult_numerator = (coupling_norm + coupling_auxvar)
            scale_mult = (scale_mult_numerator / (2.0 * coupling_norm))
            scale_mult[less_than] = 1.0
            scale_mult[less_than_neg] = 0.0

            # shape (n_coupled_cells, n_coupling_filters) * (n_coupled_cells, 1)
            # -> (n_coupled_cells, n_coupling filters)
            coupling_w_prox_applied = coupling_filter_w * scale_mult[:, None]

            auxvar_prox = scale_mult_numerator / 2.0
            auxvar_prox[less_than] = coupling_auxvar[less_than]
            auxvar_prox[less_than_neg] = 0.0

            if self.spat_filter_sparsity_lambda != 0.0:
                spat_filter = torch.clamp_min_(spat_filter - self.spat_filter_sparsity_lambda, 0.0) \
                              - torch.clamp_min_(-spat_filter - self.spat_filter_sparsity_lambda, 0.0)

        return coupling_w_prox_applied, feedback_filter_w, spat_filter, bias_w, auxvar_prox

    def clone_parameters_model(self, coord_desc_other: 'AlternatingOptMixin') -> None:

        self.coupling_w.data[:] = coord_desc_other.coupling_w.data[:]
        self.feedback_w.data[:] = coord_desc_other.feedback_w.data[:]
        self.bias.data[:] = coord_desc_other.bias.data[:]
        self.coupling_filter_norm.data[:] = coord_desc_other.coupling_filter_norm.data[:]

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        return (
            self.coupling_w.detach().cpu().numpy(),
            self.feedback_w.detach().cpu().numpy(),
            self.stim_spat_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy(),
            self.coupling_filter_norm.detach().cpu().numpy()
        )

    def return_parameters_torch(self) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        return (
            self.coupling_w.detach().clone(),
            self.feedback_w.detach().clone(),
            self.stim_spat_w.detach().clone(),
            self.bias.detach().clone(),
            self.coupling_filter_norm.detach().clone()
        )


class SingleCellEncodingLoss(nn.Module):

    def __init__(self,
                 spatial_filter: torch.Tensor,
                 timecourse_filter: torch.Tensor,
                 fixed_feedback_filter: torch.Tensor,
                 fixed_coupling_filters: torch.Tensor,
                 bias: torch.Tensor,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        '''

        :param spatial_filter: shape (n_pixels, )
        :param timecourse_filter:  shape (n_bins_filter, )
        :param fixed_feedback_filter: shape (n_bins_filter, )
        :param fixed_coupling_filters: shape (n_coupled_cells, n_bins_filter)
        :param bias: shape (1, )
        :param loss_callable:
        '''

        super().__init__()

        self.loss_callable = loss_callable

        # shape (n_pixels, )
        self.register_buffer('spat_filt', spatial_filter)

        # shape (n_bins_filter, )
        self.register_buffer('time_filt', timecourse_filter)

        # shape (n_bins_filter, )
        self.register_buffer('feedback_filt', fixed_feedback_filter)

        # shape (n_coupled_cells, n_bins_filter)
        self.register_buffer('couple_filt', fixed_coupling_filters)

        # shape (1, )
        self.register_buffer('bias', bias)

    def forward(self,
                batched_stimuli: torch.Tensor,
                stimulus_time_comp: torch.Tensor,
                batched_fit_cell_spikes: torch.Tensor,
                batched_coupled_cell_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param batched_stimuli: shape (batch, n_pixels)
        :param stimulus_time_comp: shape (n_bins, )
        :param batched_fit_cell_spikes: shape (batch, n_bins)
        :param batched_coupled_cell_spikes: shape (batch, n_coupled_cells, n_bins)
        :return: shape (batch, )
        '''

        n_bins_filter = self.time_filt.shape[0]

        # shape (batch, n_pixels) @ (n_pixels, 1) -> (batch, 1) -> (batch, )
        spatial_inner_prod = (batched_stimuli @ self.spat_filt[:, None]).squeeze(1)

        # shape (n_bins - n_bins_filter + 1, )
        time_domain_applied = F.conv1d(stimulus_time_comp[None, None, :],
                                       self.time_filt[None, None, :]).squeeze(1).squeeze(0)

        # shape (batch, n_bins - n_bins_filter + 1)
        stimulus_gensig_contrib = spatial_inner_prod[:, None] * time_domain_applied[None, :]

        # shape (batch, n_bins - n_bins_filter + 1)
        feedback_applied = F.conv1d(batched_fit_cell_spikes[:, None, :],
                                    self.feedback_filt[None, None, :]).squeeze(1)

        # shape (batch, n_bins - n_bins_filter + 1)
        coupling_applied = F.conv1d(batched_coupled_cell_spikes,
                                    self.couple_filt[None, :, :]).squeeze(1)

        print(spatial_inner_prod.shape, time_domain_applied.shape)
        print(stimulus_gensig_contrib.shape, feedback_applied.shape, coupling_applied.shape)

        total_gensig = stimulus_gensig_contrib + feedback_applied + coupling_applied + self.bias[:, None]

        return self.loss_callable(total_gensig[:, :-1], batched_fit_cell_spikes[:, n_bins_filter:])


def precompute_timecourse_feedback_coupling_basis_convs(
        stimulus_time: torch.Tensor,
        stim_time_filt_basis: torch.Tensor,
        spikes_cell: torch.Tensor,
        feedback_filt_basis: torch.Tensor,
        spikes_coupled_cells: torch.Tensor,
        coupling_filt_basis: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''

    :param stimulus_time: shape (n_bins, ), the time component of the stimulus
    :param stim_time_filt_basis: (n_basis_time_filt, n_bins_filt), stimulus temporal basis set
    :param spikes_cell: (batch, n_bins), spikes for the cell being fit, for every trial
    :param feedback_filt_basis: (n_basis_feedback, n_bins_filt), feedback basis
    :param spikes_coupled_cells: (batch, n_coupled_cells, n_bins),  spikes for all of the
        coupled cells, for every trial
    :param coupling_filt_basis: (n_basis_coupling, n_bins_filt), coupling basis

    :return:
    '''

    with torch.no_grad():
        # shape (1, 1, n_bins)
        # -> (n_basis_time_filt, n_bins - n_bins_filter + 1)
        time_domain_basis_applied = F.conv1d(stimulus_time[None, None, :],
                                             stim_time_filt_basis[:, None, :]).squeeze(0)

        # shape (batch, n_basis_feedback, n_bins - n_bins_filter + 1)
        feedback_basis_applied = F.conv1d(spikes_cell[:, None, :],
                                          feedback_filt_basis[:, None, :])

        batch, n_coupled_cells, _ = spikes_coupled_cells.shape
        temp_flattened_coupled_spikes = spikes_coupled_cells.reshape((batch * n_coupled_cells, -1))

        coupling_basis_applied_temp = F.conv1d(temp_flattened_coupled_spikes[:, None, :],
                                               coupling_filt_basis[:, None, :])

        coupling_basis_applied = coupling_basis_applied_temp.reshape(batch, n_coupled_cells,
                                                                     coupling_filt_basis.shape[0], -1)

        return time_domain_basis_applied, feedback_basis_applied, coupling_basis_applied


def fit_timecourse_only(stimulus_frames: torch.Tensor,
                        binned_spikes_cell: torch.Tensor,
                        fixed_spatial_filter: torch.Tensor,
                        stim_time_basis_convolved: torch.Tensor,
                        spike_feedback_basis_convolved: torch.Tensor,
                        coupling_basis_convolved: torch.Tensor,
                        loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                        solver_params: ProxSolverParams,
                        device: torch.device,
                        l21_group_sparse_lambda: float = 0.0,
                        initial_guess_timecourse: Optional[np.ndarray] = None,
                        verbose: bool = False) \
        -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    '''
    Function for running a single iteration of (timecourse, feedback, coupling) optimization
        where the spatial filter is fixed and the inner product of the spatial stimulus
        with the spatial filter is computed ahead of time

    :param stimulus_frames: raw stimulus frames, flattened, shape (batch, n_pixels)
    :param binned_spikes_cell: binned spikes for the cells being fit,
        shape (batch, n_bins)
    :param fixed_spatial_filter: shape (n_pixels, ); the fixed spatial filter
    :param stim_time_basis_convolved: shape (n_basis_time, n_bins - n_bins_filter + 1)
    :param spike_feedback_basis_convolved: shape (batch, 1, n_basis_feedback, n_bins_total - n_bins_filter + 1)
    :param coupling_basis_convolved: shape (batch, n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1)
    :param loss_callable:
    :param solver_params:
    :param l21_group_sparse_lambda:
    :param initial_guess_timecourse:
    :return:
    '''

    n_basis_stim_time, _ = stim_time_basis_convolved.shape
    n_basis_feedback = spike_feedback_basis_convolved.shape[2]
    n_coupled_cells, n_basis_coupling = coupling_basis_convolved.shape[1], coupling_basis_convolved.shape[2]

    with torch.no_grad():
        # shape (batch, n_pixels) @ (n_pixels, 1) -> (batch, 1) -> (batch, )
        spatial_filter_applied = (stimulus_frames @ fixed_spatial_filter[:, None]).squeeze(1)

    timecourse_opt_module = TimecourseFitGLM(
        n_basis_stim_time,
        n_basis_feedback,
        n_basis_coupling,
        n_coupled_cells,
        loss_callable,
        l21_group_sparse_lambda,
        stim_time_init_guess=initial_guess_timecourse
    ).to(device)

    loss_timecourse = single_prox_solve(timecourse_opt_module,
                                        solver_params,
                                        verbose=verbose,
                                        spat_filt_movie=spatial_filter_applied,
                                        time_basis_filt_movie=stim_time_basis_convolved,
                                        binned_spikes_cell=binned_spikes_cell,
                                        filtered_feedback=spike_feedback_basis_convolved,
                                        filtered_coupling=coupling_basis_convolved)

    return_parameters = timecourse_opt_module.return_parameters_np()

    del timecourse_opt_module

    return loss_timecourse, return_parameters


def fit_scaled_spatial_only(stimulus_frames: torch.Tensor,
                            stimulus_timedomain: torch.Tensor,
                            binned_spikes_cell: torch.Tensor,
                            prior_spatial_filter: Union[torch.Tensor, np.ndarray],
                            fixed_timecourse_filter: Union[torch.Tensor, np.ndarray],
                            spike_feedback_basis_convolved: torch.Tensor,
                            coupling_basis_convolved: torch.Tensor,
                            loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                            solver_params: ProxSolverParams,
                            device: torch.device,
                            l21_group_sparse_lambda: float = 0.0,
                            initial_feedback_w_guess: Optional[np.ndarray] = None,
                            initial_coupling_w_guess: Optional[np.ndarray] = None,
                            initial_bias_guess: Optional[np.ndarray] = None,
                            initial_coupling_norm_guess: Optional[np.ndarray] = None,
                            verbose: bool = False):
    n_trials, n_pixels = stimulus_frames.shape
    n_basis_feedback = spike_feedback_basis_convolved.shape[2]
    n_coupled_cells, n_basis_coupling = coupling_basis_convolved.shape[1], coupling_basis_convolved.shape[2]

    with torch.no_grad():
        # shape (1, 1, n_bins - n_bins_filter + 1)
        # -> (1, n_bins - n_bins_filter + 1)
        # -> (n_bins - n_bins_filter + 1, )
        filtered_stim_time = F.conv1d(stimulus_timedomain[None, None, :],
                                      fixed_timecourse_filter[None, None, :]).squeeze(1).squeeze(0)

    spat_opt_module = SpatialScaleGLM(n_pixels,
                                      prior_spatial_filter,
                                      n_basis_feedback,
                                      n_basis_coupling,
                                      n_coupled_cells,
                                      loss_callable,
                                      group_sparse_reg_lambda=l21_group_sparse_lambda,
                                      init_feedback_w=initial_feedback_w_guess,
                                      init_coupling_w=initial_coupling_w_guess,
                                      init_bias=initial_bias_guess,
                                      init_coupling_norm=initial_coupling_norm_guess).to(device)

    loss_spatial = single_prox_solve(spat_opt_module,
                                     solver_params,
                                     verbose=verbose,
                                     raw_stim_frame=stimulus_frames,
                                     time_filt_movie=filtered_stim_time,
                                     binned_spikes_cell=binned_spikes_cell,
                                     filtered_feedback=spike_feedback_basis_convolved,
                                     filtered_coupling=coupling_basis_convolved)

    return_parameters = spat_opt_module.return_parameters_torch()

    del spat_opt_module

    return loss_spatial, return_parameters


def fit_spatial_only(stimulus_frames: torch.Tensor,
                     stimulus_timedomain: torch.Tensor,
                     binned_spikes_cell: torch.Tensor,
                     prior_spatial_filter: Union[torch.Tensor, np.ndarray],
                     fixed_timecourse_filter: Union[torch.Tensor, np.ndarray],
                     spike_feedback_basis_convolved: torch.Tensor,
                     coupling_basis_convolved: torch.Tensor,
                     loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                     solver_params: ProxSolverParams,
                     device: torch.device,
                     l21_group_sparse_lambda: float = 0.0,
                     l1_spat_sparse_lambda: float = 0.0,
                     l2_prior_similarity: float = 0.0,
                     initial_spat_filt_guess: Optional[np.ndarray] = None,
                     initial_feedback_w_guess: Optional[np.ndarray] = None,
                     initial_coupling_w_guess: Optional[np.ndarray] = None,
                     initial_bias_guess: Optional[np.ndarray] = None,
                     initial_coupling_norm_guess: Optional[np.ndarray] = None,
                     verbose: bool = False):
    '''
    Function for running a single interation of (spatial, feedback, coupling) optimization
        where the timecourse filter is fixed and the convolution of the timecourse filter
        with the separable time component of the stimulus is precomputed.

    :param stimulus_frames: raw stimulus frames, flattened, shape (batch, n_pixels)
    :param stimulus_timedomain: time domain of the separable stimulus, shape (n_bins, )
    :param binned_spikes_cell: binned spikes for the cells being fit,
        shape (batch, n_bins)
    :param prior_spatial_filter: shape (n_pixels, ); the fixed spatial filter
    :param fixed_timecourse_filter: shape (n_bins_filter, )
    :param stim_time_convolved: shape (n_bins - n_bins_filter + 1, )
    :param stim_time_basis_convolved: shape (n_basis_time, n_bins - n_bins_filter + 1
    :param spike_feedback_basis_convolved: shape (batch, 1, n_basis_feedback, n_bins_total - n_bins_filter + 1)
    :param coupling_basis_convolved: shape (batch, n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1
    :param loss_callable:
    :param solver_params:
    :param device:
    :param l21_group_sparse_lambda:
    :param l1_spat_sparse_lambda:
    :param verbose:
    :return:
    '''

    n_trials, n_pixels = stimulus_frames.shape
    n_basis_feedback = spike_feedback_basis_convolved.shape[2]
    n_coupled_cells, n_basis_coupling = coupling_basis_convolved.shape[1], coupling_basis_convolved.shape[2]

    with torch.no_grad():

        # shape (1, 1, n_bins - n_bins_filter + 1)
        # -> (1, n_bins - n_bins_filter + 1)
        # -> (n_bins - n_bins_filter + 1, )
        filtered_stim_time = F.conv1d(stimulus_timedomain[None, None, :],
                                      fixed_timecourse_filter[None, None, :]).squeeze(1).squeeze(0)

    if l2_prior_similarity is not None:
        spat_opt_module = SpatialFitGLM_WithSTAPrior(n_pixels,
                                                     prior_spatial_filter,
                                                     n_basis_feedback,
                                                     n_basis_coupling,
                                                     n_coupled_cells,
                                                     loss_callable,
                                                     l2_prior_similarity,
                                                     l21_group_sparse_lambda,
                                                     l1_spat_sparse_lambda,
                                                     stim_spat_init_guess=initial_spat_filt_guess,
                                                     init_feedback_w=initial_feedback_w_guess,
                                                     init_coupling_w=initial_coupling_w_guess,
                                                     init_bias=initial_bias_guess,
                                                     init_coupling_norm=initial_coupling_norm_guess).to(device)
    else:
        spat_opt_module = SpatialFitGLM(n_pixels,
                                        n_basis_feedback,
                                        n_basis_coupling,
                                        n_coupled_cells,
                                        loss_callable,
                                        l21_group_sparse_lambda,
                                        l1_spat_sparse_lambda,
                                        stim_spat_init_guess=initial_spat_filt_guess,
                                        init_feedback_w=initial_feedback_w_guess,
                                        init_coupling_w=initial_coupling_w_guess,
                                        init_bias=initial_bias_guess,
                                        init_coupling_norm=initial_coupling_norm_guess).to(device)

    loss_spatial = single_prox_solve(spat_opt_module,
                                     solver_params,
                                     verbose=verbose,
                                     raw_stim_frame=stimulus_frames,
                                     time_filt_movie=filtered_stim_time,
                                     binned_spikes_cell=binned_spikes_cell,
                                     filtered_feedback=spike_feedback_basis_convolved,
                                     filtered_coupling=coupling_basis_convolved)

    return_parameters = spat_opt_module.return_parameters_np()

    del spat_opt_module

    return loss_spatial, return_parameters


def alternating_optim(stimulus_frames: torch.Tensor,
                      binned_spikes_cell: torch.Tensor,
                      prior_spatial_filter: torch.Tensor,
                      stim_time_basis: torch.Tensor,
                      stim_time_basis_convolved: torch.Tensor,
                      spike_feedback_basis_convolved: torch.Tensor,
                      coupling_basis_convolved: torch.Tensor,
                      loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                      solver_params: ProxSolverParams,
                      n_iters_outer_opt: int,
                      device: torch.device,
                      l21_group_sparse_lambda: float = 0.0,
                      l1_spat_sparse_lambda: float = 0.0,
                      l2_prior_similarity: float = 0.0,
                      initial_guess_timecourse: Optional[torch.Tensor] = None,
                      initial_guess_coupling: Optional[torch.Tensor] = None,
                      initial_guess_coupling_norm: Optional[torch.Tensor] = None,
                      initial_guess_feedback: Optional[torch.Tensor] = None,
                      initial_guess_bias: Optional[torch.Tensor] = None,
                      inner_opt_verbose: bool = False,
                      outer_opt_verbose: bool = False):
    '''
    Function for performing alternating optimization between
        (timecourse filter, feedback, coupling) optimization and
        (spatial filter, feedback, coupling) optimization
    
    :param stimulus_frames: raw stimulus frames, flattened, shape (batch, n_pixels)
    :param binned_spikes_cell: binned spikes for the cells being fit,
        shape (batch, n_bins)
    :param prior_spatial_filter: shape (n_pixels, ); the fixed spatial filter
    :param stim_time_basis: shape (n_basis_time, n_bins_filter)
    :param stim_time_basis_convolved: shape (n_basis_time, n_bins - n_bins_filter + 1)
    :param spike_feedback_basis_convolved: shape (batch, 1, n_basis_feedback, n_bins_total - n_bins_filter + 1)
    :param coupling_basis_convolved: shape (batch, n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1)
    :param loss_callable: 
    :param solver_params: 
    :param n_iters_outer_opt: 
    :param device: 
    :param l21_group_sparse_lambda: 
    :param l1_spat_sparse_lambda: 
    :param l2_prior_similarity: 
    :param initial_guess_timecourse: 
    :param inner_opt_verbose: 
    :param outer_opt_verbose: 
    :return: 
    '''

    n_trials, n_pixels = stimulus_frames.shape
    n_basis_stim_time, _ = stim_time_basis_convolved.shape
    n_basis_feedback = spike_feedback_basis_convolved.shape[2]
    n_coupled_cells, n_basis_coupling = coupling_basis_convolved.shape[1], coupling_basis_convolved.shape[2]

    # shape (n_pixels, )
    prev_iter_spatial_filter = prior_spatial_filter

    # shape (n_timecourse_basis, )
    timecourse_w = initial_guess_timecourse

    coupling_w, feedback_w = initial_guess_coupling, initial_guess_feedback
    bias, coupling_filt_norm = initial_guess_bias, initial_guess_coupling_norm

    for iter_num in range(n_iters_outer_opt):

        with torch.no_grad():

            # shape (1, n_timecourse_basis) @ (n_timecourse_basis, n_bins_filter)
            # -> (1, n_bins_filter) -> (n_bins_filter, )
            timecourse_filter = (timecourse_w[None, :] @ stim_time_basis).squeeze(0)
            timecourse_norm = torch.linalg.norm(timecourse_filter)

            timecourse_w = timecourse_w / timecourse_norm
            prev_iter_spatial_filter = prev_iter_spatial_filter * timecourse_norm

            # shape (1, n_basis_time) @ (n_basis_time, n_bins - n_bins_filter + 1)
            # -> (1, n_bins - n_bins_filter + 1) -> (n_bins - n_bins_filter + 1, )
            filtered_stim_time = (timecourse_w @ stim_time_basis_convolved).squeeze(0)

        if l2_prior_similarity is not None:
            spat_opt_module = SpatialFitGLM_WithSTAPrior(n_pixels,
                                                         prior_spatial_filter,
                                                         n_basis_feedback,
                                                         n_basis_coupling,
                                                         n_coupled_cells,
                                                         loss_callable,
                                                         l2_prior_similarity,
                                                         l21_group_sparse_lambda,
                                                         l1_spat_sparse_lambda,
                                                         stim_spat_init_guess=prev_iter_spatial_filter,
                                                         init_feedback_w=feedback_w,
                                                         init_coupling_w=coupling_w,
                                                         init_bias=bias,
                                                         init_coupling_norm=coupling_filt_norm).to(device)
        else:
            spat_opt_module = SpatialFitGLM(n_pixels,
                                            n_basis_feedback,
                                            n_basis_coupling,
                                            n_coupled_cells,
                                            loss_callable,
                                            l21_group_sparse_lambda,
                                            l1_spat_sparse_lambda,
                                            stim_spat_init_guess=prev_iter_spatial_filter,
                                            init_feedback_w=feedback_w,
                                            init_coupling_w=coupling_w,
                                            init_bias=bias,
                                            init_coupling_norm=coupling_filt_norm).to(device)

        loss_spatial = single_prox_solve(spat_opt_module,
                                         solver_params,
                                         verbose=inner_opt_verbose,
                                         raw_stim_frame=stimulus_frames,
                                         time_filt_movie=filtered_stim_time,
                                         binned_spikes_cell=binned_spikes_cell,
                                         filtered_feedback=spike_feedback_basis_convolved,
                                         filtered_coupling=coupling_basis_convolved)

        coupling_w, feedback_w, prev_iter_spatial_filter, bias, coupling_filt_norm = spat_opt_module.return_parameters_torch()

        if outer_opt_verbose:
            print(f"Iter {iter_num} spatial opt. loss {loss_spatial}")

        del spat_opt_module

        with torch.no_grad():
            # shape (batch, n_pixels) @ (n_pixels, 1) -> (batch, 1) -> (batch, )
            spatial_filter_applied = (stimulus_frames @ prev_iter_spatial_filter[:, None]).squeeze(1)

        timecourse_opt_module = TimecourseFitGLM(
            n_basis_stim_time,
            n_basis_feedback,
            n_basis_coupling,
            n_coupled_cells,
            loss_callable,
            l21_group_sparse_lambda,
            stim_time_init_guess=timecourse_w,
            init_feedback_w=feedback_w,
            init_coupling_w=coupling_w,
            init_bias=bias,
            init_coupling_norm=coupling_filt_norm
        ).to(device)

        loss_timecourse = single_prox_solve(timecourse_opt_module,
                                            solver_params,
                                            verbose=inner_opt_verbose,
                                            spat_filt_movie=spatial_filter_applied,
                                            time_basis_filt_movie=stim_time_basis_convolved,
                                            binned_spikes_cell=binned_spikes_cell,
                                            filtered_feedback=spike_feedback_basis_convolved,
                                            filtered_coupling=coupling_basis_convolved)

        coupling_w, feedback_w, timecourse_w, bias, coupling_filt_norm = timecourse_opt_module.return_parameters_torch()

        del timecourse_opt_module

        if outer_opt_verbose:
            print(f"Iter {iter_num} timecourse opt. loss {loss_timecourse}")

    return loss_timecourse, (prev_iter_spatial_filter, timecourse_w, coupling_w, feedback_w, bias)
