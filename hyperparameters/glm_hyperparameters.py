from dataclasses import dataclass

import numpy as np

from typing import Dict, Union

from glm_basis_functions.time_basis_functions import make_cosine_bump_family, backshift_n_samples


@dataclass
class GLMModelHyperparameters:
    spatial_basis: Union[None, np.ndarray]  # shape (n_pixels, n_basis_spat_stim) if specified
    timecourse_basis: np.ndarray  # shape (n_basis_timecourse, n_bins_filter)
    feedback_basis: np.ndarray  # shape (n_basis_feedback, n_bins_filter)
    coupling_basis: np.ndarray  # shape (n_basis_coupling, n_bins_filter)

    neighboring_cell_dist: Dict[str, float]

    l21_reg_const: float = 0.0
    l1_spat_sparse_reg_const: float = 0.0
    l2_prior_reg_const: float = 0.0
    l2_prior_filt_scale: float = 1.0
    n_iter_inner: int = 750
    n_iter_outer: int = 2


def make_fullres_hyperparameters_2018_08_07_5(temporal_binsize: int) \
        -> Dict[str, GLMModelHyperparameters]:
    '''

    :return:
    '''

    ret_dict = {}  # type: Dict[str, GLMModelHyperparameters]

    interval_step_time = temporal_binsize * 0.05

    ### MAKE THE BASIS FUNCTIONS FOR PARASOLS ##########
    ####################################################
    A_parasol_timecourse = 5.5
    C_parasol_timecourse = 1.0
    N_BASIS_parasol_timecourse = 18
    TIMESTEPS_parasol_timecourse = np.r_[0:250:interval_step_time]

    bump_basis_timecourse_parasol = make_cosine_bump_family(A_parasol_timecourse, C_parasol_timecourse,
                                                            N_BASIS_parasol_timecourse,
                                                            TIMESTEPS_parasol_timecourse)
    bump_basis_timecourse_parasol = bump_basis_timecourse_parasol[8:]
    bump_basis_timecourse_parasol_bs = backshift_n_samples(bump_basis_timecourse_parasol,
                                                           int(500 / temporal_binsize))
    bump_basis_timecourse_parasol_rev = np.ascontiguousarray(bump_basis_timecourse_parasol_bs[:, ::-1])

    ###################################################
    A_parasol_feedback = 5.5
    C_parasol_feedback = 1.0
    N_BASIS_parasol_feedback = 18
    TIMESTEPS_parasol_feedback = np.r_[0:250:interval_step_time]

    bump_basis_feedback_parasol = make_cosine_bump_family(A_parasol_feedback, C_parasol_feedback,
                                                          N_BASIS_parasol_feedback, TIMESTEPS_parasol_feedback)
    bump_basis_feedback_parsol_rev = np.ascontiguousarray(bump_basis_feedback_parasol[:, ::-1])

    ####################################################
    A_parasol_coupling = 3.2
    C_parasol_coupling = 1.0
    N_BASIS_parasol_coupling = 10
    TIMESTEPS_parasol_coupling = np.r_[0:250:interval_step_time]

    bump_basis_coupling_parasol = make_cosine_bump_family(A_parasol_coupling, C_parasol_coupling,
                                                          N_BASIS_parasol_coupling, TIMESTEPS_parasol_coupling)
    bump_basis_coupling_parasol_rev = np.ascontiguousarray(bump_basis_coupling_parasol[:, ::-1])

    #####################################################
    parasol_neighboring_cell_distance = {'ON parasol': 8,
                                         'OFF parasol': 8,
                                         'ON midget': 5,
                                         'OFF midget': 5}

    on_parasol_hyperparams = GLMModelHyperparameters(None,
                                                     bump_basis_timecourse_parasol_rev,
                                                     bump_basis_feedback_parsol_rev,
                                                     bump_basis_coupling_parasol_rev,
                                                     parasol_neighboring_cell_distance,
                                                     l21_reg_const=1e-4,
                                                     l1_spat_sparse_reg_const=1e-7,
                                                     l2_prior_reg_const=1.0, # 2.0
                                                     l2_prior_filt_scale=5e-3,
                                                     n_iter_inner=500,
                                                     n_iter_outer=2)
    ret_dict['ON parasol'] = on_parasol_hyperparams

    off_parasol_hyperparams = GLMModelHyperparameters(None,
                                                      bump_basis_timecourse_parasol_rev,
                                                      bump_basis_feedback_parsol_rev,
                                                      bump_basis_coupling_parasol_rev,
                                                      parasol_neighboring_cell_distance,
                                                      l21_reg_const=1e-4,
                                                      l1_spat_sparse_reg_const=1e-7, # 5e-7
                                                      l2_prior_reg_const=5e-1,
                                                      l2_prior_filt_scale=-5e-3,
                                                      n_iter_inner=500,
                                                      n_iter_outer=2)

    ret_dict['OFF parasol'] = off_parasol_hyperparams

    ### MIDGET HYPERPARAMETERS #######################################################
    midget_neighboring_cell_distance = {'ON parasol': 8,
                                        'OFF parasol': 8,
                                        'ON midget': 5,
                                        'OFF midget': 5}

    on_midget_hyperparams = GLMModelHyperparameters(None,
                                                    bump_basis_timecourse_parasol_rev,
                                                    bump_basis_feedback_parsol_rev,
                                                    bump_basis_coupling_parasol_rev,
                                                    midget_neighboring_cell_distance,
                                                    l21_reg_const=1e-4,
                                                    l1_spat_sparse_reg_const=1e-7, # 1e-6
                                                    l2_prior_reg_const=1e-2,
                                                    l2_prior_filt_scale=5e-3,
                                                    n_iter_inner=500,
                                                    n_iter_outer=2)
    ret_dict['ON midget'] = on_midget_hyperparams

    off_midget_hyperparams = GLMModelHyperparameters(None,
                                                     bump_basis_timecourse_parasol_rev,
                                                     bump_basis_feedback_parsol_rev,
                                                     bump_basis_coupling_parasol_rev,
                                                     midget_neighboring_cell_distance,
                                                     l21_reg_const=1e-4,
                                                     l1_spat_sparse_reg_const=1e-7, # 2e-7
                                                     l2_prior_reg_const=1e-6,
                                                     l2_prior_filt_scale=-5e-3,
                                                     n_iter_inner=500,
                                                     n_iter_outer=2)

    ret_dict['OFF midget'] = off_midget_hyperparams

    return ret_dict
