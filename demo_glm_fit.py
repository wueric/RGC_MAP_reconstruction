'''
NOTE:

This script provides a demonstration of how the GLM
models in the published paper were fit. This script
is intended ONLY as a demonstration, as there is not
enough training data supplied to produce high-quality
GLM model fits.

There is not much value in using the fits that result
from this script, as they will be massively overfit
to the training data and will not generalize

We use the same hyperparameters here as were used to
fit the models in the paper.
'''

import argparse
import pickle
from typing import Dict, List, Tuple, Union, Optional

import tqdm

import numpy as np
import torch

from data_util.cell_interactions import InteractionGraph
from data_util.matched_cells_struct import OrderedMatchedCellsStruct
from encoding_models.separable_trial_glm import alternating_optim, \
    precompute_timecourse_feedback_coupling_basis_convs, fit_scaled_spatial_only
from convex_solver_base.prox_optim import ProxFISTASolverParams

from data_util.load_data import load_timecourse_1ms_by_type, load_stacked_dataset, load_typed_dataset, \
    load_cell_interaction_graph, load_cell_ordering, load_cropped_stas_by_type, make_glm_stim_time_component
from data_util.load_data import get_fit_cell_binned_spikes_glm, get_coupled_cells_binned_spikes_glm

from hyperparameters.glm_hyperparameters import make_fullres_hyperparameters_2018_08_07_5, \
    GLMModelHyperparameters

from reconstruction_alg.glm_inverse_alg import FittedGLM, FittedGLMFamily, batch_bernoulli_spiking_neg_ll_loss

INFO_STR = '''Fits a GLM for the natural scenes flashed trial data for the cell type specified in the arguments.
(We fit by cell type because fitting the GLMs is very intensive, and we would like to minimize the amount of work
required to fit and refit the models).
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(INFO_STR)
    parser.add_argument('cell_type', type=str,
                        help="cell type to fit; must be ['ON parasol', 'OFF parasol', 'ON midget', 'OFF midget']")
    parser.add_argument('save_path', type=str, help='path to save model fits')
    parser.add_argument('-gpu', '--gpu', action='store_true', default=False, help='use GPU')
    args = parser.parse_args()

    use_gpu = args.gpu
    device = torch.device('cuda') if use_gpu else torch.device('cpu')

    cell_type = args.cell_type

    ################################################################
    # get the data
    # IMPORTANT: THIS IS A TINY SUBSET OF THE ACTUAL DATASET THAT WAS
    # USED TO FIT THE GLM; THESE FITS ARE NOT GOING TO BE VERY GOOD BY
    # DEFINITION
    train_frames, train_spikes_dict = load_typed_dataset()

    ################################################################
    # load the STA prior
    all_prior_filters = load_cropped_stas_by_type()
    relevant_cropped_stas = all_prior_filters[cell_type]

    ################################################################
    # Load the cell types and matching
    cells_ordered = load_cell_ordering() # type: OrderedMatchedCellsStruct
    relevant_cell_ref_ids = cells_ordered.get_reference_cell_order(args.cell_type)
    ct_order = cells_ordered.get_cell_types()

    #################################################################
    # Load the interactions graph
    pairwise_interactions = load_cell_interaction_graph() # type: InteractionGraph

    #################################################################
    # get the hyperparameters
    hyperparameters_by_cell_type = make_fullres_hyperparameters_2018_08_07_5(20)
    hyperparameters_for_cell_type = hyperparameters_by_cell_type[args.cell_type]  # type: GLMModelHyperparameters

    ################################################################
    # load the timecourse for the relevant cell type
    all_timecourses = load_timecourse_1ms_by_type()
    avg_timecourse = all_timecourses[cell_type]

    # shape (n_basis_timecourse, n_bins_filter)
    timecourse_basis = hyperparameters_for_cell_type.timecourse_basis

    n_bins_filter = timecourse_basis.shape[1]

    # normalize the average timecourse so that it has L2 1.0
    avg_timecourse = avg_timecourse / np.linalg.norm(avg_timecourse)
    avg_timecourse = avg_timecourse[-n_bins_filter:]

    # now solve the least squares problem for the best initial guess
    # for the timecourse in terms of the timecourse basis vectors

    initial_timevector_guess = np.linalg.solve(timecourse_basis @ timecourse_basis.T,
                                               timecourse_basis @ avg_timecourse)

    # shape (n_bins_filter, n_basis_timecourse) @ (n_basis_timecourse, ) -> (n_bins_filter, )
    timecourse_solved = timecourse_basis.T @ initial_timevector_guess
    timecourse_solved_torch = torch.tensor(timecourse_solved, device=device, dtype=torch.float32)
    timecourse_inital_guess_torch = torch.tensor(initial_timevector_guess, dtype=torch.float32, device=device)

    ################################################################
    # Make the stimulus separable time component
    stimulus_separable_time = make_glm_stim_time_component()
    stimulus_separable_time_torch = torch.tensor(stimulus_separable_time,
                                                 dtype=torch.float32, device=device)

    ################################################################
    # move the basis vectors to GPU
    feedback_basis = hyperparameters_for_cell_type.feedback_basis
    coupling_basis = hyperparameters_for_cell_type.coupling_basis

    timecourse_basis_torch = torch.tensor(timecourse_basis, dtype=torch.float32, device=device)
    feedback_basis_torch = torch.tensor(feedback_basis, dtype=torch.float32, device=device)
    coupling_basis_torch = torch.tensor(coupling_basis, dtype=torch.float32, device=device)

    ################################################################
    # move the separable stimulus (complete no crop) to GPU
    train_frames_torch = torch.tensor(train_frames, dtype=torch.float32, device=device)

    # shape (n_trials, n_pixels)
    train_frames_flat_torch = train_frames_torch.reshape(train_frames_torch.shape[0], -1)

    ################################################################
    # fit the model for each cell, one at a time
    fitted_models_dict = {}  # type: Dict[int, FittedGLM]

    model_fit_pbar = tqdm.tqdm(total=len(relevant_cell_ref_ids))
    for idx, (cell_id, prior_cropped) in enumerate(zip(relevant_cell_ref_ids, relevant_cropped_stas)):
        print(cell_id)

        fit_cell_binned_train = get_fit_cell_binned_spikes_glm(train_spikes_dict,
                                                               cell_type, cell_id,
                                                               cells_ordered)

        # figure out what neighboring cells we need to include
        all_coupled_cell_ids_ordered = []  # type: List[int]
        typed_relevant_subset = {}  # type: Dict[str, List[int]]
        for coupled_cell_type in ct_order:
            max_coupled_distance_typed = hyperparameters_for_cell_type.neighboring_cell_dist[coupled_cell_type]
            interaction_edges = pairwise_interactions.query_cell_interaction_edges(cell_id, coupled_cell_type)
            coupled_cell_ids = [x.dest_cell_id for x in interaction_edges if
                                x.additional_attributes['distance'] < max_coupled_distance_typed]

            all_coupled_cell_ids_ordered.extend(coupled_cell_ids)
            typed_relevant_subset[coupled_cell_type] = coupled_cell_ids

        # bin the spikes for the coupled cells
        # (does not include the cell being fit)
        coupled_binned_spikes_train = get_coupled_cells_binned_spikes_glm(train_spikes_dict,
                                                                          typed_relevant_subset,
                                                                          cells_ordered)

        # move the data to GPU, do the pre-convolutions with the basis vectors
        # on GPU to avoid fighting for CPU, and then clean up the raw data to save
        # GPU space
        # shape (n_pixels, )
        prior_filter_flat_torch = torch.tensor(prior_cropped.reshape(-1, ), dtype=torch.float32, device=device)

        # shape (n_train_trials, 1, n_bins)
        # -> (n_train_trials, n_bins)
        fit_cell_spikes_train_torch = torch.tensor(fit_cell_binned_train.squeeze(1), dtype=torch.float32, device=device)

        # shape (n_train_trials, n_coupled_cells, n_bins)
        coupled_spikes_train_torch = torch.tensor(coupled_binned_spikes_train, dtype=torch.float32, device=device)

        time_filt, feedback_filt, couple_filt = precompute_timecourse_feedback_coupling_basis_convs(
            stimulus_separable_time_torch,
            timecourse_basis_torch,
            fit_cell_spikes_train_torch,
            feedback_basis_torch,
            coupled_spikes_train_torch,
            coupling_basis_torch
        )

        # remove raw data that we no longer need from GPU
        del coupled_spikes_train_torch

        fitted_params_scaled = fit_scaled_spatial_only(
            train_frames_flat_torch,
            stimulus_separable_time_torch,
            fit_cell_spikes_train_torch[:, n_bins_filter - 1:],
            hyperparameters_for_cell_type.l2_prior_filt_scale * prior_filter_flat_torch,
            timecourse_solved_torch,
            feedback_filt[:, None, :, :],
            couple_filt,
            batch_bernoulli_spiking_neg_ll_loss,
            ProxFISTASolverParams(
                initial_learning_rate=1.0,
                max_iter=500,
                converge_epsilon=1e-6,
                backtracking_beta=0.5),
            device,
            l21_group_sparse_lambda=hyperparameters_for_cell_type.l21_reg_const,
            verbose=True)

        loss_scaled, (coupling_w_scale, feedback_w_scale, spatial_filt_scaled_scale, bias_scale,
               coupling_norm_scale) = fitted_params_scaled

        # now solve the problem with alternating optimization
        loss, fitted_params_alt = alternating_optim(
            train_frames_flat_torch,
            fit_cell_spikes_train_torch[:, n_bins_filter - 1:],
            spatial_filt_scaled_scale,
            timecourse_basis_torch,
            time_filt,
            feedback_filt[:, None, :, :],
            couple_filt,
            batch_bernoulli_spiking_neg_ll_loss,
            ProxFISTASolverParams(
                initial_learning_rate=1.0,
                max_iter=hyperparameters_for_cell_type.n_iter_inner,
                converge_epsilon=1e-6,
                backtracking_beta=0.5),
            hyperparameters_for_cell_type.n_iter_outer,
            device,
            l21_group_sparse_lambda=hyperparameters_for_cell_type.l21_reg_const,
            l1_spat_sparse_lambda=hyperparameters_for_cell_type.l1_spat_sparse_reg_const,
            l2_prior_similarity=hyperparameters_for_cell_type.l2_prior_reg_const,
            initial_guess_timecourse=timecourse_inital_guess_torch[None, :],
            initial_guess_coupling=coupling_w_scale,
            initial_guess_coupling_norm=coupling_norm_scale,
            initial_guess_feedback=feedback_w_scale,
            initial_guess_bias=bias_scale,
            inner_opt_verbose=True,
            outer_opt_verbose=True
        )

        # extract the parameters
        spat_filt, stim_time_w, coupling_w, feedback_w, bias = fitted_params_alt

        model_summary_parameters = FittedGLM(cell_id, spat_filt, bias, stim_time_w, feedback_w,
                                             (coupling_w, np.array(all_coupled_cell_ids_ordered)), {}, loss, None)

        fitted_models_dict[cell_id] = model_summary_parameters

        model_fit_pbar.update(1)

        del time_filt, feedback_filt, couple_filt, fit_cell_spikes_train_torch
        del prior_filter_flat_torch

        torch.cuda.empty_cache()

    model_fit_pbar.close()

    fitted_model_family = FittedGLMFamily(
        fitted_models_dict,
        None,
        timecourse_basis,
        feedback_basis,
        coupling_basis
    )

    with open(args.save_path, 'wb') as pfile:
        pickle.dump(fitted_model_family, pfile)
