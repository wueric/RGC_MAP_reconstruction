import numpy as np

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


def backshift_n_samples(bump_basis: np.ndarray, n_samples: int) -> np.ndarray:
    '''
    To avoid self-feedback (i.e. peeking at the observed spikes
        from the current time bin), we need to shift the bump
        functions back by 1 time bin to guarantee that the weights for
        the current time bin (the first entry) in the bump basis vectors
        is exactly zero
    :param bump_basis:
    :return:
    '''

    bump_basis_shifted = np.zeros_like(bump_basis)
    bump_basis_shifted[:, n_samples:] = bump_basis[:, :-n_samples]
    return bump_basis_shifted