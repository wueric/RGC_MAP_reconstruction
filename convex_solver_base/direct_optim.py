import torch
import torch.nn

from convex_solver_base.optim_base import SingleDirectSolveProblem, BatchParallelDirectSolveProblem


def single_direct_solve(problem: SingleDirectSolveProblem,
                        verbose: bool = False,
                        **kwargs) -> float:

    with torch.no_grad():
        solved_vars = problem.direct_solve(**kwargs)
        problem.assign_optimization_vars(*solved_vars)
        return problem.eval_loss(*solved_vars, **kwargs)


def batch_parallel_direct_solve(problem: BatchParallelDirectSolveProblem,
                                verbose: bool = False,
                                **kwargs) -> torch.Tensor:

    with torch.no_grad():
        solved_vars = problem.direct_solve(**kwargs)
        problem.assign_optimization_vars(*solved_vars)
        return problem.eval_loss(*solved_vars, **kwargs)


