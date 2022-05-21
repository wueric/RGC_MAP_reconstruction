import torch
import numpy as np

from typing import Tuple, Union

from convex_solver_base.optim_base import SingleProxProblem, BatchParallelProxProblem


class ProxSolverParams:
    pass


class ProxFixedStepSizeSolverParams(ProxSolverParams):
    def __init__(self,
                 initial_learning_rate: Union[float, torch.Tensor, None],
                 max_iter: int = 250,
                 converge_epsilon: float = 1e-6):
        self.initial_learning_rate = initial_learning_rate
        self.max_iter = max_iter
        self.converge_epsilon = converge_epsilon


class ProxFISTASolverParams(ProxSolverParams):
    def __init__(self,
                 initial_learning_rate: float = 1.0,
                 max_iter: int = 250,
                 converge_epsilon: float = 1e-6,
                 backtracking_beta: float = 0.5):
        self.initial_learning_rate = initial_learning_rate
        self.max_iter = max_iter
        self.converge_epsilon = converge_epsilon
        self.backtracking_beta = backtracking_beta


class ProxGradSolverParams(ProxSolverParams):

    def __init__(self,
                 initial_learning_rate: float = 1.0,
                 max_iter: int = 1000,
                 converge_epsilon: float = 1e-6,
                 backtracking_beta: float = 0.5):
        self.initial_learning_rate = initial_learning_rate
        self.max_iter = max_iter
        self.converge_epsilon = converge_epsilon
        self.backtracking_beta = backtracking_beta


def _single_proj_backtrack_search(single_prox_prob: SingleProxProblem,
                                  s_loss: float,
                                  s_vars_vectorized: torch.Tensor,
                                  gradients_vectorized: torch.Tensor,
                                  step_size: float,
                                  backtracking_beta: float,
                                  **kwargs) \
        -> Tuple[torch.Tensor, float, float]:
    '''
    Backtracking search on a quadratic approximation to the smooth part
        of the loss function

    :param loss_value:
    :param vars_vectorized:
    :param gradients_vectorized:
    :param step_size:
    :param line_search_alpha:
    :param backtracking_beta:
    :param kwargs:
    :return:
    '''

    def eval_quadratic_approximation(next_vars: torch.Tensor,
                                     approx_center: torch.Tensor,
                                     step: float) -> torch.Tensor:
        div_mul = 1.0 / (2.0 * step)

        diff = next_vars - approx_center
        return s_loss + gradients_vectorized.T @ diff + div_mul * (diff.T @ diff)

    with torch.no_grad():
        stepped_vars_vectorized = s_vars_vectorized - gradients_vectorized * step_size
        projected_stepped_vars = single_prox_prob._packed_prox_project_variables(stepped_vars_vectorized)

        quad_approx_val = eval_quadratic_approximation(projected_stepped_vars, s_vars_vectorized, step_size)
        stepped_loss = single_prox_prob._packed_eval_smooth_loss(projected_stepped_vars, **kwargs).item()

        while quad_approx_val.item() < stepped_loss:
            step_size = step_size * backtracking_beta

            stepped_vars_vectorized = s_vars_vectorized - gradients_vectorized * step_size
            projected_stepped_vars = single_prox_prob._packed_prox_project_variables(stepped_vars_vectorized)
            quad_approx_val = eval_quadratic_approximation(projected_stepped_vars, s_vars_vectorized, step_size)
            stepped_loss = single_prox_prob._packed_eval_smooth_loss(projected_stepped_vars, **kwargs).item()

        return projected_stepped_vars, step_size, stepped_loss


def single_prox_grad_solve(single_prox_prob: SingleProxProblem,
                           initial_learning_rate: float,
                           max_iter: int,
                           converge_epsilon: float,
                           backtracking_beta: float,
                           **kwargs) -> float:
    raise NotImplementedError


def single_fista_solve(single_prox_prob: SingleProxProblem,
                       initial_learning_rate: float,
                       max_iter: int,
                       converge_epsilon: float,
                       backtracking_beta: float,
                       verbose: bool = False,
                       **kwargs) -> float:
    '''
    Solves a single smooth / unsmooth separable problem using the FISTA
        accelerated first order gradient method

    For now, we work with a single vector of combined flattened parameters
        when we implement the generic algorithm

    This implementation is based on the pseudocode in Liu et al,
        Multi-Task Feature Learning Via Efficient l2,1-Norm Minimization
    which was based on the "FISTA with backtracking" from the original Beck and Teboulle paper

    The routine has been modified according to L. Vandenberghe EE236C lecture notes on FISTA
        to make it a descent method where the objective function is nonincreasing over the
        number of iterations (no Nesterov ripples), so that there are no surprises
        when the number of iterations is large

    :param initial_learning_rate: float, initial learning rate
    :param max_iter: int, maximum number of FISTA iterations to run
    :param converge_epsilon:
    :param line_search_alpha:
    :param backtracking_beta:
    :param return_loss:
    :param kwargs:
    :return:
    '''

    step_size = initial_learning_rate
    vars_iter = single_prox_prob._flatten_variables(
        single_prox_prob.parameters(recurse=False)
    ).detach().clone()
    vars_iter_min1 = vars_iter.detach().clone()
    t_iter_min1, t_iter = 0.0, 1.0

    # this variable is to keep track of the value of the variables
    # that produces the minimum value of the objective seen so far
    # We always return this variable, so that this implementation of FISTA
    # is a descent method, where the value of the objective function is nonincreasing
    descent_vars_iter = vars_iter.detach().clone()
    descent_loss = np.inf

    for iter in range(max_iter):
        alpha_iter = t_iter_min1 / t_iter
        t_iter_min1 = t_iter
        t_iter = (1 + np.sqrt(1 + 4 * t_iter ** 2)) / 2.0

        s_iter = vars_iter + alpha_iter * (vars_iter - vars_iter_min1)
        s_iter.requires_grad = True

        vars_iter_min1 = vars_iter

        # compute the forward loss and gradients for s_iter
        current_loss, gradient_flattened = single_prox_prob._packed_loss_and_gradients(s_iter, **kwargs)

        # FISTA backtracking search, with projection inside the backtracking search
        vars_iter, step_size, candidate_loss = _single_proj_backtrack_search(
            single_prox_prob,
            current_loss.item(),
            s_iter,
            gradient_flattened,
            step_size,
            backtracking_beta,
            **kwargs
        )

        # We are doing the descent version of FISTA (implementation from L. Vandenberghe UCLA EE236C notes)
        # We reject vars_iter as a solution to the problem
        # if the loss is too high, but use vars_iter to compute the next guess
        # Basically if the next guess sucks, we keep track of the last good guess to return
        # but use the next guess to compute future guesses.
        if candidate_loss < descent_loss:
            descent_loss = candidate_loss
            descent_vars_iter = vars_iter

        if verbose:
            print('iter={0}, loss={1}\r'.format(iter, descent_loss), end='')

        if torch.norm(vars_iter - vars_iter_min1).item() < converge_epsilon:
            break

    with torch.no_grad():
        stepped_variables = single_prox_prob._unflatten_variables(descent_vars_iter)
        single_prox_prob.assign_optimization_vars(*stepped_variables)
        return single_prox_prob._eval_smooth_loss(*stepped_variables, **kwargs).item()


def single_prox_solve(single_prox_prob: SingleProxProblem,
                      solver_params: ProxSolverParams,
                      verbose: bool = False,
                      **kwargs) -> float:
    if isinstance(solver_params, ProxGradSolverParams):
        return single_prox_grad_solve(
            single_prox_solve,
            solver_params.initial_learning_rate,
            solver_params.max_iter,
            solver_params.converge_epsilon,
            solver_params.backtracking_beta,
            verbose=verbose,
            **kwargs
        )

    elif isinstance(solver_params, ProxFISTASolverParams):
        return single_fista_solve(
            single_prox_prob,
            solver_params.initial_learning_rate,
            solver_params.max_iter,
            solver_params.converge_epsilon,
            solver_params.backtracking_beta,
            verbose=verbose,
            **kwargs
        )

    raise TypeError("solver_params must be instance of ProxSolverParams")


def batch_parallel_prox_grad_solve(batch_prox_problem: BatchParallelProxProblem,
                                   init_learning_rate: float,
                                   max_iter: int,
                                   converge_epsilon: float,
                                   backtracking_beta: float,
                                   verbose: bool = False,
                                   **kwargs) -> torch.Tensor:
    # shape (n_problems, ?)
    vars_iter = batch_prox_problem._batch_flatten_variables(
        batch_prox_problem.parameters(recurse=False)
    ).detach().clone()

    # shape (n_problems, )
    step_size = torch.ones((batch_prox_problem.n_problems,), dtype=vars_iter.dtype,
                           device=vars_iter.device) * init_learning_rate

    for iter in range(max_iter):

        # shape (n_problems, ) and shape (n_problems, ?)
        current_loss, gradient_flattened = batch_prox_problem._packed_loss_and_gradients(vars_iter, **kwargs)

        # Backtracking search, with projection inside the search
        vars_iter, step_size, step_loss = _batch_parallel_proj_backtrack_search(
            batch_prox_problem,
            current_loss,
            vars_iter,
            gradient_flattened,
            step_size,
            backtracking_beta,
            **kwargs
        )

        if verbose:
            step_mean = torch.mean(step_loss)
            print(f"iter={iter}, mean loss={step_mean.item()}\r", end="")

        has_converged = torch.norm(gradient_flattened, dim=1) < converge_epsilon
        if torch.all(has_converged):
            break

    with torch.no_grad():
        stepped_variables = batch_prox_problem._batch_unflatten_variables(vars_iter)
        batch_prox_problem.assign_optimization_vars(*stepped_variables)
        return batch_prox_problem._eval_smooth_loss(*stepped_variables, **kwargs)


def _batch_parallel_proj_backtrack_search(batch_prox_problem: BatchParallelProxProblem,
                                          s_loss: torch.Tensor,
                                          s_vars_vectorized: torch.Tensor,
                                          gradients_vectorized: torch.Tensor,
                                          step_size: torch.Tensor,
                                          backtracking_beta: float,
                                          **kwargs) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Backtracking search on a quadratic approximation to the smooth part
        of the loss function

    :param s_loss: shape (n_problems, )
    :param s_vars_vectorized: shape (n_problems, ?)
    :param gradients_vectorized: shape (n_problems, ?)
    :param step_size: shape (n_problems, )
    :param backtracking_beta:
    :param kwargs:
    :return:
    '''

    def eval_quadratic_approximation(next_vars: torch.Tensor,
                                     approx_center: torch.Tensor,
                                     step: torch.Tensor) -> torch.Tensor:
        '''

        :param next_vars: shape (n_problems, ?)
        :param approx_center: shape (n_problems, ?)
        :param step: shape (n_problems, )
        :return: quadratic approximation to the loss function,
            shape (n_problems, )
        '''

        with torch.no_grad():
            # shape (n_problems, )
            div_mul = 1.0 / (2.0 * step)

            # shape (n_problems, ?)
            diff = next_vars - approx_center

            # shape (n_problems, ?) -> (n_problems, )
            gradient_term = torch.sum(gradients_vectorized * diff, dim=1)

            # shape (n_problems, ?) -> (n_problems, )
            diff_term = torch.sum(diff * diff, dim=1)

            return s_loss + gradient_term + div_mul * diff_term

    with torch.no_grad():
        # shape (n_problems, ?)
        stepped_vars_vectorized = s_vars_vectorized - gradients_vectorized * step_size[:, None]

        # shape (n_problems, ?)
        projected_stepped_vars = batch_prox_problem._packed_prox_proj(stepped_vars_vectorized)

        # shape (n_problems, )
        quad_approx_val = eval_quadratic_approximation(projected_stepped_vars, s_vars_vectorized, step_size)

        # shape (n_problems, )
        stepped_loss = batch_prox_problem._packed_smooth_loss(projected_stepped_vars, **kwargs)

        approx_smaller_loss = quad_approx_val < stepped_loss

        while torch.any(approx_smaller_loss):
            # shape (n_problems, )
            update_multiplier = approx_smaller_loss.float() * backtracking_beta \
                                + (~approx_smaller_loss).float() * 1.0
            # shape (n_problems, )
            step_size = step_size * update_multiplier

            # shape (n_problems, ?)
            stepped_vars_vectorized = s_vars_vectorized - gradients_vectorized * step_size[:, None]

            # shape (n_problems, ?)
            projected_stepped_vars = batch_prox_problem._packed_prox_proj(stepped_vars_vectorized)

            # shape (n_problems, )
            quad_approx_val = eval_quadratic_approximation(projected_stepped_vars, s_vars_vectorized, step_size)

            # shape (n_problems, )
            stepped_loss = batch_prox_problem._packed_smooth_loss(projected_stepped_vars, **kwargs)

            approx_smaller_loss = quad_approx_val < stepped_loss

        return projected_stepped_vars, step_size, stepped_loss


def batch_parallel_fista_prox_solve(batch_prox_problem: BatchParallelProxProblem,
                                    initial_learning_rate: float,
                                    max_iter: int,
                                    converge_epsilon: float,
                                    backtracking_beta: float,
                                    verbose: bool = False,
                                    **kwargs) -> torch.Tensor:
    '''
    Solves a bunch of smooth / unsmooth separable problem using the FISTA
        accelerated first order gradient method in parallel

    This implementation is based on the pseudocode in Liu et al,
        Multi-Task Feature Learning Via Efficient l2,1-Norm Minimization
    which was based on the "FISTA with backtracking" from the original Beck and Teboulle paper

    The routine has been modified according to L. Vandenberghe EE236C lecture notes on FISTA
        to make it a descent method where the objective function is nonincreasing over the
        number of iterations (no Nesterov ripples), so that there are no surprises
        when the number of iterations is large

    :param initial_learning_rate: float, initial learning rate
    :param max_iter: int, maximum number of FISTA iterations to run
    :param converge_epsilon:
    :param line_search_alpha:
    :param backtracking_beta:
    :param return_loss:
    :param kwargs:
    :return:
    '''

    # shape (n_problems, ?)
    vars_iter = batch_prox_problem._batch_flatten_variables(
        batch_prox_problem.parameters(recurse=False)
    ).detach().clone()

    vars_iter_min1 = vars_iter.detach().clone()

    # shape (n_problems, )
    step_size = torch.ones((batch_prox_problem.n_problems,), dtype=vars_iter.dtype,
                           device=vars_iter.device) * initial_learning_rate

    t_iter_min1, t_iter = 0.0, 1.0

    # this variable is to keep track of the value of the variables
    # that produces the minimum value of the objective seen so far
    # We always return this variable, so that this implementation of FISTA
    # is a descent method, where the value of the objective function is nonincreasing
    descent_vars_iter = vars_iter.detach().clone()
    descent_loss = torch.empty((batch_prox_problem.n_problems, ), dtype=vars_iter.dtype,
                               device=vars_iter.device)
    descent_loss.data[:] = torch.inf

    for iter in range(max_iter):
        alpha_iter = t_iter_min1 / t_iter
        t_iter_min1 = t_iter
        t_iter = (1 + np.sqrt(1 + 4 * t_iter ** 2)) / 2.0

        with torch.no_grad():
            s_iter = vars_iter + alpha_iter * (vars_iter - vars_iter_min1)
        s_iter.requires_grad = True

        vars_iter_min1 = vars_iter

        # compute the forward loss and gradients for s_iter

        # shape (n_problems, ) and (n_problems, ?)
        current_loss, gradient_flattened = batch_prox_problem._packed_loss_and_gradients(s_iter, **kwargs)

        # FISTA backtracking search, with projection inside the backtracking search
        vars_iter, step_size, candidate_loss = _batch_parallel_proj_backtrack_search(
            batch_prox_problem,
            current_loss,
            s_iter,
            gradient_flattened,
            step_size,
            backtracking_beta,
            **kwargs
        )

        # We are doing the descent version of FISTA (implementation from L. Vandenberghe UCLA EE236C notes)
        # We reject vars_iter as a solution to the problem
        # if the loss is too high, but use vars_iter to compute the next guess
        # Basically if the next guess sucks, we keep track of the last good guess to return
        # but use the next guess to compute future guesses.
        with torch.no_grad():
            is_improvement = (candidate_loss < descent_loss)
            descent_loss.data[is_improvement] = candidate_loss.data[is_improvement]
            descent_vars_iter.data[is_improvement, :] = vars_iter.data[is_improvement, :]

        if verbose:
            print(f"iter={iter}, mean loss={torch.mean(descent_loss).item()}\r", end='')

        # quit early if all of the problems converged
        has_converged = torch.norm(vars_iter - vars_iter_min1, dim=1) < converge_epsilon
        if torch.all(has_converged):
            break

    with torch.no_grad():
        stepped_variables = batch_prox_problem._batch_unflatten_variables(descent_vars_iter)
        batch_prox_problem.assign_optimization_vars(*stepped_variables)
        return batch_prox_problem._eval_smooth_loss(*stepped_variables, **kwargs)


def batch_parallel_prox_solve(batch_prox_problem: BatchParallelProxProblem,
                              solver_params: ProxSolverParams,
                              verbose: bool = False,
                              **kwargs) -> torch.Tensor:
    if isinstance(solver_params, ProxGradSolverParams):
        return batch_parallel_prox_grad_solve(
            batch_prox_problem,
            solver_params.initial_learning_rate,
            solver_params.max_iter,
            solver_params.converge_epsilon,
            solver_params.backtracking_beta,
            verbose=verbose,
            **kwargs
        )

    elif isinstance(solver_params, ProxFISTASolverParams):
        return batch_parallel_fista_prox_solve(
            batch_prox_problem,
            solver_params.initial_learning_rate,
            solver_params.max_iter,
            solver_params.converge_epsilon,
            solver_params.backtracking_beta,
            verbose=verbose,
            **kwargs
        )

    raise TypeError("solver_params must be instance of ProxSolverParams")
