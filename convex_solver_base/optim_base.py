from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Callable, Optional, Iterable, Union
import functools
import operator

import torch
import torch.nn as nn
from torch import autograd as autograd



class _SingleSolverMixin(metaclass=ABCMeta):
    parameters: Callable[[Optional[bool]], Iterable[torch.Tensor]]

    def _flatten_variables(self,
                           variable_list: Iterable[torch.Tensor]) -> torch.Tensor:
        acc_list = [x.reshape(-1) for x in variable_list]
        return torch.cat(acc_list, dim=0)

    def _unflatten_variables(self,
                             flat_variables: torch.Tensor) -> List[torch.Tensor]:
        return_sequence, offset = [], 0
        shape_sequence = [x.shape for x in self.parameters(recurse=False)]
        offset = 0
        for tup_seq in shape_sequence:
            sizeof = functools.reduce(lambda x, y: x * y, tup_seq)
            return_sequence.append(flat_variables[offset:offset + sizeof].reshape(tup_seq))
            offset += sizeof

        return return_sequence

    def assign_optimization_vars(self, *cloned_parameters) -> None:
        for param, assign_val in zip(self.parameters(recurse=False),
                                     cloned_parameters):
            param.data[:] = assign_val


class SingleDirectSolveProblem(nn.Module, _SingleSolverMixin, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def direct_solve(self, **kwargs) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @abstractmethod
    def eval_loss(self, *args, **kwargs) -> float:
        raise NotImplementedError


class SingleUnconstrainedProblem(nn.Module, _SingleSolverMixin, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def _packed_eval_smooth_loss(self, packed_variables: torch.Tensor, **kwargs) -> torch.Tensor:
        unpacked_variables = self._unflatten_variables(packed_variables)
        return self._eval_smooth_loss(*unpacked_variables, **kwargs)

    def _packed_loss_and_gradients(self, packed_variables: torch.Tensor, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        packed_variables.requires_grad_(True)
        if packed_variables.grad is not None:
            packed_variables.grad.zero_()

        loss_tensor = self._packed_eval_smooth_loss(packed_variables, **kwargs)
        gradients_output, = autograd.grad(loss_tensor,
                                          packed_variables)

        return loss_tensor, gradients_output

    def _packed_gradients_only(self, packed_variables: torch.Tensor, **kwargs) \
            -> torch.Tensor:
        '''
        Really crappy default implementation: call _packed_loss_and_gradients, use the autograd
            and then throw away the loss

        If there is a straightforward implementation for the manual gradient, it may make
            sense to override the method

        :param packed_variables:
        :param kwargs:
        :return:
        '''

        _, gradients = self._packed_loss_and_gradients(packed_variables, **kwargs)
        return gradients


class SingleProxProblem(nn.Module, _SingleSolverMixin, metaclass=ABCMeta):
    '''
    Solves a single convex minimization problem where the objective function
        is separable into the form

            f(x) = h(x) + g(x)

        where h(x) is a smooth differentiable function and g(x) is not smooth

    Users are responsible for finding an efficient way to do the projection
        required for the g(x) proximal projection step

    Available algorithms:
        1. ISTA (proximal gradient descent)
        2. FISTA (accelerated proximal gradient descent)

    '''

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def _packed_eval_smooth_loss(self, packed_variables: torch.Tensor, **kwargs) -> torch.Tensor:
        unpacked_variables = self._unflatten_variables(packed_variables)
        return self._eval_smooth_loss(*unpacked_variables, **kwargs)

    @abstractmethod
    def prox_project_variables(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def _packed_prox_project_variables(self, packed_variables, **kwargs) -> torch.Tensor:
        unpacked_variables = self._unflatten_variables(packed_variables)
        prox_projected_unpacked = self.prox_project_variables(*unpacked_variables, **kwargs)
        return self._flatten_variables(prox_projected_unpacked)

    def _packed_loss_and_gradients(self, packed_variables: torch.Tensor, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        packed_variables.requires_grad_(True)
        if packed_variables.grad is not None:
            packed_variables.grad.zero_()

        loss_tensor = self._packed_eval_smooth_loss(packed_variables, **kwargs)
        gradients_output, = autograd.grad(loss_tensor,
                                          packed_variables)

        return loss_tensor, gradients_output

    def _packed_gradients_only(self, packed_variables: torch.Tensor, **kwargs) \
            -> torch.Tensor:
        '''
        Really crappy default implementation: call _packed_loss_and_gradients, use the autograd
            and then throw away the loss

        If there is a straightforward implementation for the manual gradient, it may make
            sense to override the method

        :param packed_variables:
        :param kwargs:
        :return:
        '''

        _, gradients = self._packed_loss_and_gradients(packed_variables, **kwargs)
        return gradients


class _BatchSolverMixin(metaclass=ABCMeta):
    n_problems: int
    parameters: Callable[[Optional[bool]], Iterable[torch.Tensor]]

    def _batch_flatten_variables(self,
                                 variable_list: Iterable[torch.Tensor]) \
            -> torch.Tensor:
        return torch.cat([x.reshape(self.n_problems, -1)
                          for x in variable_list], dim=1)

    def _batch_unflatten_variables(self,
                                   flat_variables: torch.Tensor) \
            -> List[torch.Tensor]:
        return_seq, offset = [], 0
        shape_sequence = [x.shape for x in self.parameters(recurse=False)]
        for tup_seq in shape_sequence:
            sizeof = functools.reduce(operator.mul, tup_seq[1:])
            return_seq.append(flat_variables[:, offset:offset + sizeof].reshape(tup_seq))
            offset += sizeof

        return return_seq

    def assign_optimization_vars(self, *cloned_parameters) -> None:
        for param, assign_val in zip(self.parameters(recurse=False), cloned_parameters):
            param.data[:] = assign_val


class BatchParallelDirectSolveProblem(nn.Module, _BatchSolverMixin, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def direct_solve(self, **kwargs) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @abstractmethod
    def eval_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class BatchParallelUnconstrainedProblem(nn.Module, _BatchSolverMixin, metaclass=ABCMeta):
    '''
    Solves a bunch of unconstrained convex minimization problems
        in parallel using line search algorithm

    Can take advantage of autograd if desired
    '''

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def n_problems(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        '''
        Evaluates the loss for each problem
        :param args:
        :param kwargs:
        :return: shape (n_problems, )
        '''
        raise NotImplementedError

    def _packed_eval_smooth_loss(self, packed_variables: torch.Tensor, **kwargs) -> torch.Tensor:
        unpacked_variables = self._batch_unflatten_variables(packed_variables)
        return self._eval_smooth_loss(*unpacked_variables, **kwargs)

    def _packed_gradients_only(self, packed_variables: torch.Tensor, **kwargs) \
            -> torch.Tensor:
        '''
        Really crappy default implementation: call _packed_loss_and_gradients, use the autograd
            and then throw away the loss

        If there is a straightforward implementation for the manual gradient, it may make
            sense to override the method

        :param packed_variables:
        :param kwargs:
        :return:
        '''

        _, gradients = self._packed_loss_and_gradients(packed_variables, **kwargs)
        return gradients

    def _packed_loss_and_gradients(self, packed_variables: torch.Tensor, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        '''

        (This works but might be very slow)

        This is a default implementation using autograd, if it is possible and easier to do
            by manually computing the gradient instead of using autograd, this method
            should be overridden.

        :param packed_variables:
        :param kwargs:
        :return: shape (n_problems, ) and (n_problems, ?)
        '''

        packed_variables.requires_grad_(True)
        if packed_variables.grad is not None:
            packed_variables.grad.zero_()

        loss_tensor = self._packed_eval_smooth_loss(packed_variables, **kwargs)
        summed_loss_tensor = torch.sum(loss_tensor, dim=0)
        gradients_output, = autograd.grad(summed_loss_tensor,
                                          packed_variables)

        return loss_tensor, gradients_output


class BatchParallelProxProblem(nn.Module, _BatchSolverMixin, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def n_problems(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _prox_proj(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def _packed_prox_proj(self, packed_variables, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            unpacked_variables = self._batch_unflatten_variables(packed_variables)
            prox_projected_unpacked = self._prox_proj(*unpacked_variables, **kwargs)
            return self._batch_flatten_variables(prox_projected_unpacked)

    @abstractmethod
    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def _packed_smooth_loss(self, packed_variables: torch.Tensor, **kwargs) -> torch.Tensor:
        '''

        :param packed_variables: packed variables, shape (n_problems, ?)
        :param kwargs:
        :return: shape (n_problems, )
        '''
        unpacked_variables = self._batch_unflatten_variables(packed_variables)
        return self._eval_smooth_loss(*unpacked_variables, **kwargs)

    def _packed_loss_and_gradients(self, packed_variables: torch.Tensor, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        '''

        (This works but might be very slow)

        This is a default implementation using autograd, if it is possible and easier to do
            by manually computing the gradient instead of using autograd, this method
            should be overridden.

        :param packed_variables:
        :param kwargs:
        :return: shape (n_problems, ) and (n_problems, ?)
        '''

        packed_variables.requires_grad_(True)
        if packed_variables.grad is not None:
            packed_variables.grad.zero_()

        loss_tensor = self._packed_smooth_loss(packed_variables, **kwargs)
        gradients_output, = autograd.grad(torch.sum(loss_tensor),
                                          packed_variables)

        return loss_tensor, gradients_output


SingleProblem = Union[SingleDirectSolveProblem, SingleUnconstrainedProblem, SingleProxProblem]


BatchParallelProblem = Union[
    BatchParallelDirectSolveProblem, BatchParallelUnconstrainedProblem, BatchParallelProxProblem]
