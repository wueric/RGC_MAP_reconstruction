import pickle
from typing import Dict, List, Tuple, Union, Optional, Callable, Sequence, Any
from dataclasses import dataclass
import argparse

import tqdm

import numpy as np
import torch

from data_util.matched_cells_struct import OrderedMatchedCellsStruct
from data_util.load_data import make_glm_stim_time_component, compute_stimulus_onset_spikes, \
    load_stacked_dataset, load_cell_ordering, load_fitted_glm_families

from reconstruction_alg.glm_inverse_alg import PackedGLMTensors, \
    convert_glm_family_to_np, make_full_res_packed_glm_tensors

from linear_models.linear_decoding_models import ClosedFormLinearModel
import denoisers.denoiser_wrappers as denoiser_wrappers

from reconstruction_alg.hqs_alg import HQS_XGenerator, BatchParallel_HQS_XGenerator, \
    BatchParallel_DirectSolve_HQS_ZGenerator, DirectSolve_HQS_ZGenerator, BatchParallel_UnblindDenoiserPrior_HQS_ZProb, \
    UnblindDenoiserPrior_HQS_ZProb
from reconstruction_alg.hqs_alg import scheduled_rho_fixed_lambda_single_hqs_solve
from reconstruction_alg.glm_inverse_alg import FittedGLMFamily, batch_per_bin_bernoulli_neg_log_likelihood, \
    BatchKnownSeparable_TrialGLM_ProxProblem
from hyperparameters.hyperparameters import glm_hqs_hyperparameters_2018_08_07_5, HQSHyperparameters, \
    make_hqs_schedule

from collections import namedtuple


def generate_onebatch_reconstruction(
        x_prob: BatchKnownSeparable_TrialGLM_ProxProblem,
        z_prob: BatchParallel_UnblindDenoiserPrior_HQS_ZProb,
        image_shape: Tuple[int, int],
        single_batch_spikes: torch.Tensor,
        reconstruction_hyperparams: HQSHyperparameters,
        initialize_noise_level: float = 1e-3,
        initialize_linear_model: Optional[ClosedFormLinearModel] = None) \
        -> torch.Tensor:

    height, width = image_shape

    #################################################################################
    # get the hyperparameters
    schedule_rho = make_hqs_schedule(reconstruction_hyperparams)
    prior_weight = reconstruction_hyperparams.prior_weight
    max_iter = reconstruction_hyperparams.max_iter

    unblind_denoise_hqs_x_prob_solver_iter = BatchParallel_HQS_XGenerator(first_niter=300,
                                                                          subsequent_niter=300)
    unblind_denoise_hqs_z_prob_solver_iter = BatchParallel_DirectSolve_HQS_ZGenerator()

    # if we want to use the linear reconstruction as an intermediate, do a linear reconstruction
    initialize_z_tensor = None
    if initialize_linear_model is not None:

        summed_spikes_for_linear = compute_stimulus_onset_spikes(single_batch_spikes)
        initialize_z_tensor = initialize_linear_model(summed_spikes_for_linear)
    else:
        initialize_z_tensor = torch.randn((single_batch_spikes.shape[0], height, width),
                                          dtype=torch.float32) * initialize_noise_level

    x_prob.reinitialize_variables(initialized_z_const=initialize_z_tensor)
    x_prob.precompute_gensig_components(single_batch_spikes)
    z_prob.reinitialize_variables()

    _ = scheduled_rho_fixed_lambda_single_hqs_solve(
        x_prob,
        iter(unblind_denoise_hqs_x_prob_solver_iter),
        z_prob,
        iter(unblind_denoise_hqs_z_prob_solver_iter),
        iter(schedule_rho),
        prior_weight,
        max_iter,
        verbose=False,
        save_intermediates=False,
        observed_spikes=single_batch_spikes
    )

    denoise_hqs_reconstructed_image = x_prob.get_reconstructed_image()
    return denoise_hqs_reconstructed_image


def batch_parallel_generate_open_loop_reconstructions(
        example_spikes: np.ndarray,
        packed_glm_tensors: PackedGLMTensors,
        glm_time_component: np.ndarray,
        reconstruction_hyperparams: HQSHyperparameters,
        max_batch_size: int,
        device: torch.device,
        initialize_noise_level: float = 1e-3,
        initialize_linear_model: Optional[ClosedFormLinearModel] = None) \
        -> np.ndarray:
    '''

    :param example_spikes:
    :param packed_glm_tensors:
    :param glm_time_component:
    :param device:
    :return:
    '''

    n_examples, n_cells, n_bins_observed = example_spikes.shape
    _, height, width = packed_glm_tensors.spatial_filters.shape

    example_spikes_torch = torch.tensor(example_spikes, dtype=torch.float32, device=device)

    ################################################################################
    # first load the unblind denoiser
    unblind_denoiser_model = denoiser_wrappers.load_zhang_drunet_unblind_denoiser(device)
    unblind_denoiser_callable = denoiser_wrappers.make_unblind_apply_zhang_dpir_denoiser(
        unblind_denoiser_model,
        (-1.0, 1.0), (0.0, 255))

    #################################################################################
    # get the hyperparameters
    schedule_rho = make_hqs_schedule(reconstruction_hyperparams)
    prior_weight = reconstruction_hyperparams.prior_weight
    max_iter = reconstruction_hyperparams.max_iter

    ################################################################################
    # make the models for the first N-1 iterations (we need to make new models for the
    # final iteration because the size of that batch could be weird)
    unblind_denoise_hqs_x_prob = BatchKnownSeparable_TrialGLM_ProxProblem(
        max_batch_size,
        packed_glm_tensors.spatial_filters,
        packed_glm_tensors.timecourse_filters,
        packed_glm_tensors.feedback_filters,
        packed_glm_tensors.coupling_filters,
        packed_glm_tensors.coupling_indices,
        packed_glm_tensors.bias,
        glm_time_component,
        batch_per_bin_bernoulli_neg_log_likelihood,
        schedule_rho[0]
    ).to(device)

    unblind_denoise_hqs_z_prob = BatchParallel_UnblindDenoiserPrior_HQS_ZProb(
        max_batch_size,
        unblind_denoiser_callable,
        (height, width),
        schedule_rho[0],
        prior_lambda=reconstruction_hyperparams.prior_weight
    ).to(device)

    # run the first N-1 iterations
    output_image_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
    pbar = tqdm.tqdm(total=n_examples)
    for low in range(0, n_examples - max_batch_size + 1, max_batch_size):
        high = low + max_batch_size
        glm_trial_spikes_torch = example_spikes_torch[low:high, ...]

        batch_reconstructions = generate_onebatch_reconstruction(
            unblind_denoise_hqs_x_prob,
            unblind_denoise_hqs_z_prob,
            (height, width),
            glm_trial_spikes_torch,
            reconstruction_hyperparams,
            initialize_noise_level=initialize_noise_level,
            initialize_linear_model=initialize_linear_model
        ).detach().cpu().numpy()

        output_image_buffer_np[low:high, :, :] = batch_reconstructions

        pbar.update(max_batch_size)

    del unblind_denoise_hqs_x_prob, unblind_denoise_hqs_z_prob

    # run the final iteration
    low = (n_examples // max_batch_size) * max_batch_size
    high = n_examples
    eff_batch_size = high - low

    unblind_denoise_hqs_x_prob = BatchKnownSeparable_TrialGLM_ProxProblem(
        eff_batch_size,
        packed_glm_tensors.spatial_filters,
        packed_glm_tensors.timecourse_filters,
        packed_glm_tensors.feedback_filters,
        packed_glm_tensors.coupling_filters,
        packed_glm_tensors.coupling_indices,
        packed_glm_tensors.bias,
        glm_time_component,
        batch_per_bin_bernoulli_neg_log_likelihood,
        schedule_rho[0]
    ).to(device)

    unblind_denoise_hqs_z_prob = BatchParallel_UnblindDenoiserPrior_HQS_ZProb(
        eff_batch_size,
        unblind_denoiser_callable,
        (height, width),
        schedule_rho[0],
        prior_lambda=prior_weight
    ).to(device)

    glm_trial_spikes_torch = example_spikes_torch[low:high, ...]

    batch_reconstructions = generate_onebatch_reconstruction(
        unblind_denoise_hqs_x_prob,
        unblind_denoise_hqs_z_prob,
        (height, width),
        glm_trial_spikes_torch,
        reconstruction_hyperparams,
        initialize_noise_level=initialize_noise_level,
        initialize_linear_model=initialize_linear_model
    ).detach().cpu().numpy()

    output_image_buffer_np[low:high, :, :] = batch_reconstructions

    pbar.update(eff_batch_size)
    pbar.close()

    del unblind_denoise_hqs_x_prob, unblind_denoise_hqs_z_prob, example_spikes_torch, unblind_denoiser_model

    return output_image_buffer_np


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Mass-generate reconstructions for each method")
    parser.add_argument('mode', type=str, help='must be one of [LNP, GLM]')
    parser.add_argument('output_path', type=str, help='save path for reconstructions')
    parser.add_argument('-b', '--batch', type=int, default=16, help='batch size for reconstruction')
    parser.add_argument('-n', '--noise_init', type=float, default=1e-3,
                        help='noise standard deviation, on interval [-1, 1]')
    parser.add_argument('-l', '--linear_init', action='store_true', default=False, help='use linear initialization')
    parser.add_argument('-gpu', '--gpu', action='store_true', default=False, help='use GPU')
    args = parser.parse_args()

    encoding_model_str = args.mode
    use_glm = (encoding_model_str == 'GLM')

    use_gpu= args.gpu
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    cells_ordered = load_cell_ordering()
    ct_order = cells_ordered.get_cell_types()
    cell_ids_as_ordered_list = []
    for ct in ct_order:
        cell_ids_as_ordered_list.extend(cells_ordered.get_reference_cell_order(ct))

    # images in the demo dataset are already cropped, and we don't do downsampling
    # images are also already rescaled to [-1, 1]
    ground_truth_images, spikes_binned = load_stacked_dataset()
    glm_stim_time_component = make_glm_stim_time_component()

    # load the GLM model and hyperparameters
    if use_glm:
        fitted_glm_families = load_fitted_glm_families()
        fitted_glm_families = {key: convert_glm_family_to_np(val, spat_shape=(height, width))
                               for key, val in fitted_glm_families.items()}
        packed_glm_tensors = make_full_res_packed_glm_tensors(
            cells_ordered,
            fitted_glm_families)

        hyperparameters = glm_hqs_hyperparameters_2018_08_07_5()

        target_reconstructions = batch_parallel_generate_open_loop_reconstructions(
            response_vector,
            packed_glm_tensors,
            glm_stim_time_component,
            hyperparameters,
            16,
            device,
            initialize_noise_level=args.noise_init,
            initialize_linear_model=linear_model_param)

        pass
    # load the LNP model and hyperparameters
    else:
        pass


    with open(args.save_path, 'wb') as pfile:
        save_data = {
            'ground_truth': test_frames if not args.heldout else heldout_frames,
            method_lookup: target_reconstructions
        }

        pickle.dump(save_data, pfile)

    print('done')
