import torch
from torch.optim import LBFGS
import numpy as np

from typing import Tuple, List

from convex_solver_base.optim_base import SingleUnconstrainedProblem, BatchParallelUnconstrainedProblem


class UnconstrainedSolverParams:
    pass


class GradientSolverParams(UnconstrainedSolverParams):

    def __init__(self,
                 init_learning_rate: float = 1.0,
                 line_search_alpha: float = 0.25,
                 backtracking_beta: float = 0.5,
                 max_iter: int = 1000,
                 converge_step_cutoff: float = 1e-6):
        self.init_learning_rate = init_learning_rate
        self.line_search_alpha = line_search_alpha
        self.backtracking_beta = backtracking_beta
        self.max_iter = max_iter
        self.converge_step_cutoff = converge_step_cutoff


class FistaSolverParams(UnconstrainedSolverParams):

    def __init__(self,
                 initial_learning_rate: float = 1.0,
                 max_iter: int = 250,
                 converge_epsilon: float = 1e-6,
                 backtracking_beta: float = 0.5):
        self.initial_learning_rate = initial_learning_rate
        self.max_iter = max_iter
        self.converge_epsilon = converge_epsilon
        self.backtracking_beta = backtracking_beta


def _single_armijo_backtracking_search(single_unc_problem: SingleUnconstrainedProblem,
                                       loss_value: float,
                                       vars_packed: torch.Tensor,
                                       gradient_packed: torch.tensor,
                                       step_direction: torch.Tensor,
                                       step_size: float,
                                       line_search_alpha: float,
                                       backtracking_beta: float,
                                       **kwargs) \
        -> Tuple[torch.Tensor, float, float]:
    with torch.no_grad():
        loss_step = single_unc_problem._packed_eval_smooth_loss(vars_packed + step_direction * step_size,
                                                                **kwargs).item()
        rhs = loss_value + step_size * line_search_alpha * (gradient_packed.T @ step_direction).item()
        while loss_step > rhs:
            step_size = step_size * backtracking_beta
            loss_step = single_unc_problem._packed_eval_smooth_loss(vars_packed + step_direction * step_size,
                                                                    **kwargs).item()
            rhs = loss_value + step_size * line_search_alpha * (gradient_packed.T @ step_direction).item()

        grad_l2 = (gradient_packed.T @ gradient_packed).item()

        return vars_packed + step_direction * step_size, step_size, grad_l2


def _single_backtracking_search(single_unc_problem: SingleUnconstrainedProblem,
                                loss_value: float,
                                vars_vectorized: torch.Tensor,
                                gradients_vectorized: torch.Tensor,
                                step_size: float,
                                line_search_alpha: float,
                                backtracking_beta: float,
                                **kwargs) \
        -> Tuple[torch.Tensor, float, float]:
    with torch.no_grad():
        grad_l2 = (gradients_vectorized.T @ gradients_vectorized).item()
        stepped_vars = vars_vectorized - gradients_vectorized * step_size

        rhs_ineq = loss_value - line_search_alpha * step_size * grad_l2
        lhs_ineq = single_unc_problem._packed_eval_smooth_loss(stepped_vars, **kwargs).item()

        while lhs_ineq > rhs_ineq:
            step_size = backtracking_beta * step_size
            rhs_ineq = loss_value - line_search_alpha * step_size * grad_l2

            stepped_vars = vars_vectorized - gradients_vectorized * step_size
            lhs_ineq = single_unc_problem._packed_eval_smooth_loss(stepped_vars, **kwargs).item()

        return stepped_vars, step_size, grad_l2


def single_grad_backtrack_solve(single_unc_problem: SingleUnconstrainedProblem,
                                init_learning_rate: float,
                                line_search_alpha: float,
                                backtracking_beta: float,
                                max_iter: int,
                                converge_step_cutoff: float,
                                verbose: bool = False,
                                **kwargs) -> float:
    '''
    Minimizes a bunch of convex problems in parallel using backtracking line search

    :param init_learning_rate: Initial learning rate for line search
    :param line_search_alpha: Alpha, in interval [0.0, 0.5).
    :param backtracking_beta: Beta, in interval [0.0, 1). Multiply the learning rate by this value
        every time the step size needs to be reduced
    :param max_iter: Maximum number of iterations to run the algorithm
    :param converge_step_cutoff: Terminate if the step size is less than this value
    :param kwargs:
    :return:
    '''

    step_size = init_learning_rate
    stepped_vars_vectorized = single_unc_problem._flatten_variables(
        single_unc_problem.parameters(recurse=False)
    ).detach().clone()
    for iter_count in range(max_iter):

        # calculate the forward loss and gradient
        current_loss, gradient_flattened = single_unc_problem._packed_loss_and_gradients(stepped_vars_vectorized,
                                                                                         **kwargs)

        '''
        stepped_vars_vectorized, step_size, grad_l2 = _single_backtracking_search(
            single_unc_problem,
            current_loss.item(),
            stepped_vars_vectorized,
            gradient_flattened,
            step_size,
            line_search_alpha,
            backtracking_beta,
            **kwargs
        )
        '''

        stepped_vars_vectorized, step_size, grad_l2 = _single_armijo_backtracking_search(
            single_unc_problem,
            current_loss.item(),
            stepped_vars_vectorized,
            gradient_flattened,
            -gradient_flattened,
            step_size,
            line_search_alpha,
            backtracking_beta,
            **kwargs
        )

        if verbose:
            print('iter={0}, loss={1}, grad_l2={2}\r'.format(iter_count, round(current_loss.item(), 5),
                                                             round(grad_l2, 9)), end='')

        ###### evaluate termination criterion on gt ###########################
        if grad_l2 < converge_step_cutoff:
            break

    with torch.no_grad():
        stepped_variables = single_unc_problem._unflatten_variables(stepped_vars_vectorized)
        single_unc_problem.assign_optimization_vars(*stepped_variables)
        return single_unc_problem._eval_smooth_loss(*stepped_variables, **kwargs).item()


def _single_fista_backtrack_search(single_unc_problem: SingleUnconstrainedProblem,
                                   loss_termination_value: float,
                                   s_vector: torch.Tensor,
                                   gradients_vectorized: torch.Tensor,
                                   step_size: float,
                                   backtracking_beta: float,
                                   **kwargs) \
        -> Tuple[torch.Tensor, float, float]:
    def eval_quadratic_approximation(next_vars: torch.Tensor,
                                     approx_point: torch.Tensor,
                                     step) -> torch.Tensor:
        diff = next_vars - approx_point
        return loss_termination_value + gradients_vectorized.T @ diff + diff.T @ diff / (2.0 * step)

    with torch.no_grad():
        stepped_vars_vectorized = s_vector - gradients_vectorized * step_size
        quad_approx_val = eval_quadratic_approximation(stepped_vars_vectorized, s_vector, step_size)
        stepped_loss = single_unc_problem._packed_eval_smooth_loss(stepped_vars_vectorized, **kwargs)

        while quad_approx_val.item() < stepped_loss.item():
            step_size = step_size * backtracking_beta

            stepped_vars_vectorized = s_vector - gradients_vectorized * step_size
            quad_approx_val = eval_quadratic_approximation(stepped_vars_vectorized, s_vector, step_size)
            stepped_loss = single_unc_problem._packed_eval_smooth_loss(stepped_vars_vectorized, **kwargs)

        return stepped_vars_vectorized, step_size, stepped_loss


def single_fista_solve(single_unc_problem: SingleUnconstrainedProblem,
                       initial_learning_rate: float,
                       max_iter: int,
                       converge_epsilon: float,
                       backtracking_beta: float,
                       verbose: bool = False,
                       **kwargs) -> float:
    step_size = initial_learning_rate
    vars_iter = single_unc_problem._flatten_variables(
        single_unc_problem.parameters(recurse=False)
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

        # update the loop variables
        alpha_iter = t_iter_min1 / t_iter
        t_iter_min1 = t_iter
        t_iter = (1 + np.sqrt(1 + 4 * t_iter ** 2)) / 2.0

        s_iter = vars_iter + alpha_iter * (vars_iter - vars_iter_min1)
        s_iter.requires_grad = True

        vars_iter_min1 = vars_iter

        # compute forward loss and gradients for s_iter
        current_loss, gradient_flattened = single_unc_problem._packed_loss_and_gradients(s_iter, **kwargs)

        # FISTA backtracking search
        # Since we assume that the objective function is smooth
        # we don't need a projection step
        vars_iter, step_size, candidate_loss = _single_fista_backtrack_search(
            single_unc_problem,
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
            print('iter={0}, loss={1}\r'.format(iter, descent_loss.item()), end='')

        # the early termination condition is a statement about the norm
        # of the change in the variables
        if torch.norm(vars_iter - vars_iter_min1).item() < converge_epsilon:
            break

    with torch.no_grad():
        stepped_variables = single_unc_problem._unflatten_variables(descent_vars_iter)
        single_unc_problem.assign_optimization_vars(*stepped_variables)
        return single_unc_problem._eval_smooth_loss(*stepped_variables, **kwargs).item()


def single_l_bfgs_solve(single_unc_problem: SingleUnconstrainedProblem,
                        lbfgs_n_iters: int = 100,
                        learning_rate: float = 1.0,
                        max_iter_per_iter: int = 20,
                        tolerance_grad: float = 1e-5,
                        tolerance_change: float = 1e-9,
                        history_size: int = 100,
                        verbose: bool = False,
                        **kwargs) -> float:
    lbfgs_optimizer = LBFGS(single_unc_problem.parameters(recurse=False),
                            lr=learning_rate,
                            max_iter=max_iter_per_iter,
                            tolerance_grad=tolerance_grad,
                            tolerance_change=tolerance_change,
                            history_size=history_size,
                            line_search_fn='strong_wolfe')

    for i in range(lbfgs_n_iters):
        lbfgs_optimizer.zero_grad()
        objective_val = single_unc_problem(**kwargs)
        objective_val.backward()
        lbfgs_optimizer.step(lambda: single_unc_problem(**kwargs))

        if verbose:
            print(f"LBFGS iter={i}, loss={objective_val.item()}\r", end='')

    return single_unc_problem(**kwargs).item()


def single_unconstrained_solve(single_unc_problem: SingleUnconstrainedProblem,
                               solver_params: UnconstrainedSolverParams,
                               verbose: bool = False,
                               **kwargs) -> float:
    if not isinstance(solver_params, UnconstrainedSolverParams):
        raise TypeError("solver_params must be UnconstrainedSolverParams")

    if isinstance(solver_params, GradientSolverParams):
        return single_grad_backtrack_solve(
            single_unc_problem,
            solver_params.init_learning_rate,
            solver_params.line_search_alpha,
            solver_params.backtracking_beta,
            solver_params.max_iter,
            solver_params.converge_step_cutoff,
            verbose=verbose,
            **kwargs
        )

    elif isinstance(solver_params, FistaSolverParams):
        return single_fista_solve(
            single_unc_problem,
            solver_params.initial_learning_rate,
            solver_params.max_iter,
            solver_params.converge_epsilon,
            solver_params.backtracking_beta,
            verbose=verbose,
            **kwargs
        )

    else:
        raise RuntimeError("Solver method not implemented")


def _batch_parallel_backtrack_search(batch_unc_problem: BatchParallelUnconstrainedProblem,
                                     loss_value: torch.Tensor,
                                     vars_vectorized: torch.Tensor,
                                     gradients_vectorized: torch.Tensor,
                                     step_size: torch.Tensor,
                                     line_search_alpha: float,
                                     backtracking_beta: float,
                                     **kwargs) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Inner helper function for generic backtracking line search for unconstrained
        minimization of a convex function

    :param loss_value: shape (n_problems, )
    :param vars_vectorized: shape (n_problems, n_vars)
    :param gradients_vectorized: shape (n_problems, n_vars)
    :param step_size: shape (n_problems, )
    :param line_search_alpha:
    :param backtracking_beta:
    :param kwargs:
    :return:
    '''

    with torch.no_grad():
        # shape (n_problems, )
        grad_l2 = torch.sum(gradients_vectorized * gradients_vectorized, dim=1)

        # shape (n_problems, n_vars)
        stepped_vars = vars_vectorized - gradients_vectorized * step_size[:, None]

        # shape (n_problems, )
        rhs_ineq = loss_value - line_search_alpha * step_size * grad_l2
        lhs_ineq = batch_unc_problem._packed_eval_smooth_loss(stepped_vars, **kwargs)

        ineq_eval = lhs_ineq > rhs_ineq
        while torch.any(ineq_eval):
            update_multiplier = ineq_eval.float() * backtracking_beta + (~ineq_eval).float()
            step_size = step_size * update_multiplier

            # shape (n_problems, n_vars)
            stepped_vars = vars_vectorized - gradients_vectorized * step_size[:, None]

            # shape (n_problems, )
            rhs_ineq = loss_value - line_search_alpha * step_size * grad_l2
            lhs_ineq = batch_unc_problem._packed_eval_smooth_loss(stepped_vars, **kwargs)
            ineq_eval = lhs_ineq > rhs_ineq

        return stepped_vars, step_size, grad_l2


def batch_parallel_grad_backtrack_solve(batch_unc_problem: BatchParallelUnconstrainedProblem,
                                        init_learning_rate: float,
                                        line_search_alpha: float,
                                        backtracking_beta: float,
                                        max_iter: int,
                                        converge_step_cutoff: float,
                                        verbose: bool = False,
                                        **kwargs) -> torch.Tensor:
    '''
    Minimizes a bunch of convex problems in parallel using backtracking line search

    :param init_learning_rate:
    :param line_search_alpha:
    :param backtracking_beta:
    :param max_iter:
    :param converge_step_cutoff:
    :param kwargs:
    :return:
    '''

    # shape (n_problems, n_vars)
    vars_iter = batch_unc_problem._batch_flatten_variables(
        batch_unc_problem.parameters(recurse=False)
    ).detach().clone()

    step_size = torch.ones((batch_unc_problem.n_problems,), dtype=vars_iter.dtype,
                           device=vars_iter.device) * init_learning_rate
    for iter_count in range(max_iter):

        current_loss, gradient_flattened = batch_unc_problem._packed_loss_and_gradients(vars_iter, **kwargs)

        vars_iter, step_size, grad_l2 = _batch_parallel_backtrack_search(
            batch_unc_problem,
            current_loss,
            vars_iter,
            gradient_flattened,
            step_size,
            line_search_alpha,
            backtracking_beta,
            **kwargs
        )

        if verbose:
            with torch.no_grad():
                current_mean = torch.mean(current_loss)
                print(f"iter={iter}, mean loss={current_mean.item()}\r", end="")

        has_converged = grad_l2 < converge_step_cutoff
        if torch.all(has_converged):
            break

    with torch.no_grad():
        stepped_variables = batch_unc_problem._batch_unflatten_variables(vars_iter)
        batch_unc_problem.assign_optimization_vars(*stepped_variables)
        return batch_unc_problem._eval_smooth_loss(*stepped_variables, **kwargs)


def _batch_parallel_fista_backtrack_search(batch_unc_problem: BatchParallelUnconstrainedProblem,
                                           loss_termination_value: torch.Tensor,
                                           s_vector: torch.Tensor,
                                           gradients_vectorized: torch.Tensor,
                                           step_size: torch.Tensor,
                                           backtracking_beta: float,
                                           **kwargs) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def eval_quadratic_approximation(next_vars: torch.Tensor,
                                     approx_point: torch.Tensor,
                                     step: torch.Tensor) -> torch.Tensor:
        # shape (n_problems, n_vars)
        diff = next_vars - approx_point

        # shape (n_problems, )
        grad_diff_ip = torch.sum(gradients_vectorized * diff, dim=1)
        diff_ip = torch.sum(diff * diff, dim=1)

        return loss_termination_value + grad_diff_ip + (diff_ip / (2.0 * step))

    with torch.no_grad():
        # shape (n_problems, n_vars)
        stepped_vars_vectorized = s_vector - gradients_vectorized * step_size[:, None]

        # shape (n_problems, )
        quad_approx_val = eval_quadratic_approximation(stepped_vars_vectorized,
                                                       s_vector,
                                                       step_size)
        stepped_loss = batch_unc_problem._packed_eval_smooth_loss(stepped_vars_vectorized, **kwargs)

        ineq_eval = quad_approx_val < stepped_loss
        while torch.any(ineq_eval):
            update_multiplier = ineq_eval.float() * backtracking_beta + (~ineq_eval).float()
            step_size = step_size * update_multiplier

            # shape (n_problems, n_vars)
            stepped_vars_vectorized = s_vector - gradients_vectorized * step_size[:, None]

            # shape (n_problems, )
            quad_approx_val = eval_quadratic_approximation(stepped_vars_vectorized,
                                                           s_vector,
                                                           step_size)
            stepped_loss = batch_unc_problem._packed_eval_smooth_loss(stepped_vars_vectorized, **kwargs)

            ineq_eval = quad_approx_val < stepped_loss

        return stepped_vars_vectorized, step_size, stepped_loss


def batch_parallel_fista_solve(batch_unc_problem: BatchParallelUnconstrainedProblem,
                               initial_learning_rate: float,
                               max_iter: int,
                               converge_epsilon: float,
                               backtracking_beta: float,
                               verbose: bool = False,
                               **kwargs) -> torch.Tensor:
    # shpae (n_problems, n_vars)
    vars_iter = batch_unc_problem._batch_flatten_variables(
        batch_unc_problem.parameters(recurse=False)
    ).detach().clone()
    vars_iter_min1 = vars_iter.detach().clone()

    # shape (n_problems, )
    step_size = torch.ones((batch_unc_problem.n_problems,), dtype=vars_iter.dtype,
                           device=vars_iter.device) * initial_learning_rate

    t_iter_min1, t_iter = 0.0, 1.0

    # this variable is to keep track of the value of the variables
    # that produces the minimum value of the objective seen so far
    # We always return this variable, so that this implementation of FISTA
    # is a descent method, where the value of the objective function is nonincreasing
    descent_vars_iter = vars_iter.detach().clone()
    descent_loss = torch.empty((batch_unc_problem.n_problems, ), dtype=vars_iter.dtype,
                               device=vars_iter.device)
    descent_loss.data[:] = torch.inf


    for iter in range(max_iter):

        alpha_iter = t_iter_min1 / t_iter
        t_iter_min1 = t_iter
        t_iter = (1 + np.sqrt(1 + 4 * t_iter ** 2)) / 2.0

        with torch.no_grad():
            s_iter = vars_iter + alpha_iter * (vars_iter - vars_iter_min1)

        vars_iter_min1 = vars_iter
        current_loss, gradient_flattened = batch_unc_problem._packed_loss_and_gradients(s_iter, **kwargs)

        # FISTA backtracking search
        vars_iter, step_size, candidate_loss = _batch_parallel_fista_backtrack_search(
            batch_unc_problem,
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
            with torch.no_grad():
                current_mean = torch.mean(descent_loss)
                print(f"iter={iter}, mean loss={current_mean.item()}\r", end="")

        termination_norm = torch.norm(vars_iter - vars_iter_min1, dim=1)
        if torch.all(termination_norm < converge_epsilon):
            break

    stepped_variables = batch_unc_problem._batch_unflatten_variables(descent_vars_iter)
    batch_unc_problem.assign_optimization_vars(*stepped_variables)
    with torch.no_grad():
        return batch_unc_problem._eval_smooth_loss(*stepped_variables, **kwargs)


def batch_parallel_unconstrained_solve(batch_unc_problem: BatchParallelUnconstrainedProblem,
                                       solver_params: UnconstrainedSolverParams,
                                       verbose: bool = False,
                                       **kwargs) -> torch.Tensor:
    if not isinstance(solver_params, UnconstrainedSolverParams):
        raise TypeError("solver_params must be UnconstrainedSolverParams")

    if isinstance(solver_params, GradientSolverParams):
        return batch_parallel_grad_backtrack_solve(
            batch_unc_problem,
            solver_params.init_learning_rate,
            solver_params.line_search_alpha,
            solver_params.backtracking_beta,
            solver_params.max_iter,
            verbose=verbose,
            **kwargs
        )

    elif isinstance(solver_params, FistaSolverParams):
        return batch_parallel_fista_solve(
            batch_unc_problem,
            solver_params.initial_learning_rate,
            solver_params.max_iter,
            solver_params.converge_epsilon,
            solver_params.backtracking_beta,
            verbose=verbose,
            **kwargs
        )

    else:
        raise RuntimeError("Solver method not implemented")
