import numpy as np
import torch
import torch.nn as nn

from convex_solver_base.optim_base import SingleDirectSolveProblem, BatchParallelDirectSolveProblem, \
    SingleProblem, BatchParallelProblem
from convex_solver_base.direct_optim import batch_parallel_direct_solve, single_direct_solve

from abc import ABCMeta, abstractmethod
from typing import Callable, Union, Tuple, Iterator, Optional, List

HQS_ParameterizedSolveFn = Callable[[Union[SingleProblem, BatchParallelProblem], bool], Union[float, torch.Tensor]]


class HQS_X_Problem(metaclass=ABCMeta):
    rho: float

    @abstractmethod
    def assign_z(self, z: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho


class BatchParallel_HQS_X_Problem(metaclass=ABCMeta):
    rho: float

    @abstractmethod
    def assign_z(self, z: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho


class HQS_Z_Problem(metaclass=ABCMeta):
    rho: float
    ax_const_tensor: torch.Tensor
    batch_size: int

    def assign_A_x(self, Ax: torch.Tensor) -> None:
        self.ax_const_tensor.data[:] = Ax

    @abstractmethod
    def get_z(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def compute_prox_penalty(self, *args, **kwargs) -> torch.Tensor:
        z = self.get_z(*args, **kwargs)
        prox_diff = z - self.ax_const_tensor
        return 0.5 * self.rho * torch.sum(prox_diff * prox_diff)

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho

    def set_prior_lambda(self, lambda_val: float) -> None:
        self.prior_lambda = lambda_val


class BatchParallel_HQS_Z_Problem(metaclass=ABCMeta):
    rho: float

    ax_const_tensor: torch.Tensor

    def assign_A_x(self, Ax: torch.Tensor) -> None:
        self.ax_const_tensor.data[:] = Ax

    @abstractmethod
    def get_z(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def compute_prox_penalty(self, *args, **kwargs) -> torch.Tensor:
        z = self.get_z(*args, **kwargs)
        prox_diff = z - self.ax_const_tensor
        return 0.5 * self.rho * torch.sum(prox_diff * prox_diff, dim=(1, 2))

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho

    def set_prior_lambda(self, lambda_val: float) -> None:
        self.prior_lambda = lambda_val


class DirectSolve_HQS_ZGenerator:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self) -> HQS_ParameterizedSolveFn:
        return single_direct_solve


class BatchParallel_DirectSolve_HQS_ZGenerator:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self) -> HQS_ParameterizedSolveFn:
        return batch_parallel_direct_solve


class UnblindDenoiserPrior_HQS_ZProb(SingleDirectSolveProblem,
                                     HQS_Z_Problem):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 unblind_denoiser_module: Callable[[torch.Tensor, Union[float, torch.Tensor]], torch.Tensor],
                 image_shape: Tuple[int, int],
                 hqs_rho: float,
                 prior_lambda: float = 1.0,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.rho = hqs_rho
        self.prior_lambda = prior_lambda

        self.unblind_denoiser_callable = unblind_denoiser_module
        self.height, self.width = image_shape

        self.dtype = dtype

        #### HQS constants #####################################################
        self.register_buffer('ax_const_tensor', torch.zeros((self.height, self.width), dtype=dtype))

        #### OPTIMIZATION VARIABLES ############################################
        self.z_image = nn.Parameter(torch.empty((self.height, self.width), dtype=dtype),
                                    requires_grad=False)
        nn.init.uniform_(self.z_image, a=-0.1, b=0.1)

    def direct_solve(self, **kwargs) -> Tuple[torch.Tensor, ...]:
        with torch.no_grad():
            noise_sigma2 = self.prior_lambda / self.rho
            # noise_sigma2 = 1.0 / (self.rho * self.prior_lambda)
            image_denoiser_applied = self.unblind_denoiser_callable(self.ax_const_tensor[None, :, :],
                                                                    noise_sigma2)

            temp_image = image_denoiser_applied.squeeze(0)

            self.z_image.data[:] = temp_image.data[:]

            return (temp_image,)

    def get_z(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def eval_loss(self, *args, **kwargs) -> torch.Tensor:
        return 0.0

    def reinitialize_variables(self) -> None:
        self.ax_const_tensor.data[:] = 0.0
        nn.init.uniform_(self.z_image, a=-0.1, b=0.1)


class BatchParallel_UnblindDenoiserPrior_HQS_ZProb(BatchParallelDirectSolveProblem,
                                                   BatchParallel_HQS_Z_Problem):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 batch_size: int,
                 unblind_denoiser_module: Callable[[torch.Tensor, Union[float, torch.Tensor]], torch.Tensor],
                 image_shape: Tuple[int, int],
                 hqs_rho: float,
                 prior_lambda: float = 1.0,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.batch_size = batch_size
        self.rho = hqs_rho
        self.prior_lambda = prior_lambda

        self.unblind_denoiser_callable = unblind_denoiser_module
        self.height, self.width = image_shape

        self.dtype = dtype

        #### HQS constants #####################################################
        self.register_buffer('ax_const_tensor', torch.zeros((self.batch_size, self.height, self.width), dtype=dtype))

        #### OPTIMIZATION VARIABLES ############################################
        self.z_image = nn.Parameter(torch.empty((self.batch_size, self.height, self.width), dtype=dtype),
                                    requires_grad=False)
        nn.init.uniform_(self.z_image, a=-0.1, b=0.1)

    def direct_solve(self, **kwargs) -> Tuple[torch.Tensor, ...]:
        with torch.no_grad():
            noise_sigma2 = self.prior_lambda / self.rho
            image_denoiser_applied = self.unblind_denoiser_callable(self.ax_const_tensor,
                                                                    noise_sigma2)

            self.z_image.data[:] = image_denoiser_applied.data[:]

            return (image_denoiser_applied,)

    def get_z(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def eval_loss(self, *args, **kwargs) -> torch.Tensor:
        return 0.0

    def reinitialize_variables(self) -> None:
        self.ax_const_tensor.data[:] = 0.0
        nn.init.uniform_(self.z_image, a=-0.1, b=0.1)


def scheduled_rho_lambda_single_hqs_solve(x_problem: Union[SingleProblem, HQS_X_Problem],
                                          x_solver_it: Iterator[HQS_ParameterizedSolveFn],
                                          z_problem: Union[SingleProblem, HQS_Z_Problem],
                                          z_solver_it: Iterator[HQS_ParameterizedSolveFn],
                                          rho_schedule: Iterator[float],
                                          lambda_schedule: Iterator[float],
                                          max_iter_hqs: int,
                                          verbose: bool = False,
                                          save_intermediates: bool = False,
                                          z_initialization_point: Optional[torch.Tensor] = None,
                                          **kwargs) \
        -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    intermediates_list = None
    if save_intermediates:
        intermediates_list = []

    if z_initialization_point is not None:
        x_problem.assign_z(z_initialization_point)

    for it, x_solve_fn, z_solve_fn, rho_val, prior_lambda_val in zip(range(max_iter_hqs), x_solver_it, z_solver_it,
                                                                     rho_schedule, lambda_schedule):

        x_problem.set_rho(rho_val)
        z_problem.set_rho(rho_val)
        z_problem.set_prior_lambda(prior_lambda_val)

        loss_x_prob = x_solve_fn(x_problem, verbose=verbose, **kwargs)

        with torch.no_grad():
            next_ax = x_problem.compute_A_x(*x_problem.parameters(recurse=False), **kwargs)
            z_problem.assign_A_x(next_ax)

        loss_z_prob = z_solve_fn(z_problem, verbose=verbose, **kwargs)

        with torch.no_grad():
            next_z = z_problem.get_z(*z_problem.parameters(recurse=False), **kwargs)
            x_problem.assign_z(next_z)

        if save_intermediates:
            intermediates_list.append((x_problem.get_reconstructed_image(), next_z.detach().cpu().numpy()))

        if verbose:
            print(f"HQS iter {it}, xloss {loss_x_prob}, zloss {loss_z_prob}")

    return intermediates_list
