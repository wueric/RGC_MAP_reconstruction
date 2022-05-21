import pickle
from typing import Dict, List, Tuple, Union, Optional, Callable, Sequence, Any
from dataclasses import dataclass

import numpy as np
import torch

from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct

from reconstruction_alg.glm_inverse_alg import PackedGLMTensors, \
    convert_glm_family_to_np, make_full_res_packed_glm_tensors

from linear_models.linear_decoding_models import ClosedFormLinearModel
import denoisers.denoiser_wrappers as denoiser_wrappers

from denoise_inverse_alg.binom_inverse_alg import Binom_ADMM_XGenerator, \
    DirectSolve_ADMM_ZGenerator, BatchParallel_ADMM_XGenerator, BatchParallel_DirectSolve_ADMM_ZGenerator
from denoise_inverse_alg.binom_inverse_alg import PriorBinom_HQS_XProb, \
    UnblindDenoiserPrior_HQS_ZProb, BatchParallel_UnblindDenoiserPrior_HQS_ZProb
from denoise_inverse_alg.glm_inverse_alg import KnownSeparableTrialGLMLoss, \
    per_bin_bernoulli_neg_log_likelihood, KnownSeparableTrialGLM_Precompute_HQS_XProb, \
    BatchKnownSeparable_TrialGLM_ProxProblem, \
    batch_per_bin_bernoulli_neg_log_likelihood

from optim.admm_optim import scheduled_rho_single_hqs_solve, scheduled_rho_lambda_single_hqs_solve

import argparse
import tqdm

from optimization_encoder.trial_glm import FittedGLMFamily

from collections import namedtuple

GridSearchParams = namedtuple('GridSearchParams', ['lambda_start', 'lambda_end', 'prior_weight', 'max_iter'])


def generate_linear_reconstructions(train_frames: np.ndarray,
                                    train_spikes: np.ndarray,
                                    example_spikes: np.ndarray,
                                    device: torch.device) -> np.ndarray:
    '''
    Trains the linear reconstruction model (learns reconstruction filters)
        and then generates sample linear reconstructions on the test data

    :param train_frames: shape (batch, height, width)
    :param train_spikes: shape (batch, n_cells)
    :param example_spikes: shape (n_test, n_cells)
    :return: np.ndarray, reconstructions from the test partition
        of the dataset, shape (n_test, height, width)
    '''

    train_batch, height, width = train_frames.shape
    _, n_cells = train_spikes.shape

    # shape (batch, height, width)
    train_frames_torch = torch.tensor(train_frames, dtype=torch.float32, device=device)

    # shape (batch, n_cells)
    train_spikes_torch = torch.tensor(train_spikes, dtype=torch.float32, device=device)

    # shape (n_test, n_cells)
    example_spikes_torch = torch.tensor(example_spikes, dtype=torch.float32, device=device)

    linear_decoder = ClosedFormLinearModel(n_cells, height, width).to(device)

    with torch.no_grad():
        linear_decoder.solve(train_spikes_torch, train_frames_torch)

        example_reconstructions = linear_decoder(example_spikes_torch).detach().cpu().numpy()

    del train_frames_torch, train_spikes_torch, example_spikes_torch, linear_decoder

    return example_reconstructions


def make_glm_stim_time_component(config_settings: Dict[str, Any]) \
        -> np.ndarray:
    n_bins_before = config_settings[dcp.TimebinningSection.NBINS_BEFORE_TRANS]
    n_bins_after = config_settings[dcp.TimebinningSection.NBINS_AFTER_TRANS]
    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]

    ################################################################
    # Make the stimulus separable time component based on the config
    stimulus_separable_time = np.zeros((n_bins_before + n_bins_after,),
                                       dtype=np.float32)

    n_bins_high = 2000 // samples_per_bin  # FIXME make this a constant
    stimulus_separable_time[n_bins_before:n_bins_before + n_bins_high] = 1.0
    return stimulus_separable_time


def load_fitted_glm_families(type_to_path_dict: Dict[str, str]) \
        -> Dict[str, FittedGLMFamily]:
    output_dict = {}  # type: Dict[str, FittedGLMFamily]
    for key, path in type_to_path_dict.items():
        with open(path, 'rb') as pfile:
            output_dict[key] = pickle.load(pfile)

    return output_dict


@dataclass
class ScheduleVal:
    rho_start: float
    rho_end: float

    lambda_start: float
    lambda_end: float

    n_iter: float


def make_schedules(sched_list: List[ScheduleVal]) -> Tuple[np.ndarray, np.ndarray]:
    to_cat_rho = []
    to_cat_lambda = []
    for sched in sched_list:
        to_cat_rho.append(np.logspace(np.log10(sched.rho_start), np.log10(sched.rho_end), sched.n_iter))
        to_cat_lambda.append(np.logspace(np.log10(sched.lambda_start), np.log10(sched.lambda_end), sched.n_iter))

    return np.concatenate(to_cat_rho), np.concatenate(to_cat_lambda)


FITTED_GLM_MODEL_PATH_DICT = {
    '/Volumes/Lab/Users/ericwu/yass-reconstruction/2018-08-07-5/data000':
        {
            'glm_openloop': {
                'paths': {
                    'ON parasol': 'fitted_glm_models/2018_08_07_5_on_parasol_full_res_glm_v4.p',
                    'OFF parasol': 'fitted_glm_models/2018_08_07_5_off_parasol_full_res_glm_v4.p',
                    'ON midget': 'fitted_glm_models/2018_08_07_5_on_midget_full_res_glm_v4.p',
                    'OFF midget': 'fitted_glm_models/2018_08_07_5_off_midget_full_res_glm_v4.p',
                },
                'hyperparams': GridSearchParams(lambda_start=0.021544346900318832, lambda_end=56.23413251903491,
                                                prior_weight=0.1, max_iter=25)

            },
            'glm_linear_init': {
                'paths': {
                    'ON parasol': 'fitted_glm_models/2018_08_07_5_on_parasol_full_res_glm_v4.p',
                    'OFF parasol': 'fitted_glm_models/2018_08_07_5_off_parasol_full_res_glm_v4.p',
                    'ON midget': 'fitted_glm_models/2018_08_07_5_on_midget_full_res_glm_v4.p',
                    'OFF midget': 'fitted_glm_models/2018_08_07_5_off_midget_full_res_glm_v4.p',
                },
                'hyperparams': GridSearchParams(lambda_start=0.21544346900318834, lambda_end=100.0,
                                                prior_weight=0.1, max_iter=25)

            },
            'd100': {
                'paths': {
                    'ON parasol': 'fitted_glm_models/limited_data_2018_08_07_5/d100/on_parasol_glm.p',
                    'OFF parasol': 'fitted_glm_models/limited_data_2018_08_07_5/d100/off_parasol_glm.p',
                    'ON midget': 'fitted_glm_models/limited_data_2018_08_07_5/d100/on_midget_glm.p',
                    'OFF midget': 'fitted_glm_models/limited_data_2018_08_07_5/d100/off_midget_glm.p',
                },
                'hyperparams': GridSearchParams(lambda_start=0.1, lambda_end=100.0, prior_weight=0.15, max_iter=25),
            },
            'd500': {
                'paths': {
                    'ON parasol': 'fitted_glm_models/limited_data_2018_08_07_5/d500/on_parasol_glm.p',
                    'OFF parasol': 'fitted_glm_models/limited_data_2018_08_07_5/d500/off_parasol_glm.p',
                    'ON midget': 'fitted_glm_models/limited_data_2018_08_07_5/d500/on_midget_glm.p',
                    'OFF midget': 'fitted_glm_models/limited_data_2018_08_07_5/d500/off_midget_glm.p',
                },
                'hyperparams': GridSearchParams(lambda_start=0.1, lambda_end=100.0, prior_weight=0.15, max_iter=25),
            },
            'd1000': {
                'paths': {
                    'ON parasol': 'fitted_glm_models/limited_data_2018_08_07_5/d1000/on_parasol_glm.p',
                    'OFF parasol': 'fitted_glm_models/limited_data_2018_08_07_5/d1000/off_parasol_glm.p',
                    'ON midget': 'fitted_glm_models/limited_data_2018_08_07_5/d1000/on_midget_glm.p',
                    'OFF midget': 'fitted_glm_models/limited_data_2018_08_07_5/d1000/off_midget_glm.p',
                },
                'hyperparams': GridSearchParams(lambda_start=0.1, lambda_end=100.0, prior_weight=0.15, max_iter=25),
            },
            'd2000': {
                'paths': {
                    'ON parasol': 'fitted_glm_models/limited_data_2018_08_07_5/d2000/on_parasol_glm.p',
                    'OFF parasol': 'fitted_glm_models/limited_data_2018_08_07_5/d2000/off_parasol_glm.p',
                    'ON midget': 'fitted_glm_models/limited_data_2018_08_07_5/d2000/on_midget_glm.p',
                    'OFF midget': 'fitted_glm_models/limited_data_2018_08_07_5/d2000/off_midget_glm.p',
                },
                'hyperparams': GridSearchParams(lambda_start=0.1, lambda_end=100.0, prior_weight=0.15, max_iter=25),
            },
            'd5000': {
                'paths': {
                    'ON parasol': 'fitted_glm_models/limited_data_2018_08_07_5/d5000/on_parasol_glm.p',
                    'OFF parasol': 'fitted_glm_models/limited_data_2018_08_07_5/d5000/off_parasol_glm.p',
                    'ON midget': 'fitted_glm_models/limited_data_2018_08_07_5/d5000/on_midget_glm.p',
                    'OFF midget': 'fitted_glm_models/limited_data_2018_08_07_5/d5000/off_midget_glm.p',
                },
                'hyperparams': GridSearchParams(lambda_start=0.1, lambda_end=100.0, prior_weight=0.15, max_iter=25),
            },
        },
    '/Volumes/Lab/Users/ericwu/yass-reconstruction/2018-03-01-0/merge_data010/data010': {
        'glm_openloop': {
            'paths': {
                'ON parasol': 'fitted_glm_models/2018_03_01_0_on_parasol_full_res_glm_v3_hyperparam_tuned.p',
                'OFF parasol': 'fitted_glm_models/2018_03_01_0_off_parasol_full_res_glm_v3_hyperparam_tuned.p',
                'ON midget': 'fitted_glm_models/2018_03_01_0_on_midget_full_res_glm_v3_hyperparam_tuned.p',
                'OFF midget': 'fitted_glm_models/2018_03_01_0_off_midget_full_res_glm_v3_hyperparam_tuned.p',
            },
            'hyperparams' : GridSearchParams(lambda_start=0.01, lambda_end=100.0, prior_weight=0.11, max_iter=25)
        }
    }
}

LinearModelBinningRange = namedtuple('LinearModelBinningRange', ['start_cut', 'end_cut'])


def batch_parallel_generate_open_loop_reconstructions(
        example_spikes: np.ndarray,
        packed_glm_tensors: PackedGLMTensors,
        glm_time_component: np.ndarray,
        reconstruction_hyperparams: GridSearchParams,
        max_batch_size: int,
        device: torch.device,
        initialize_noise_level: float = 1e-3,
        initialize_linear_model: Optional[
            Tuple[ClosedFormLinearModel, LinearModelBinningRange]] = None) \
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
    lambda_start, lambda_end = reconstruction_hyperparams.lambda_start, reconstruction_hyperparams.lambda_end
    prior_weight = reconstruction_hyperparams.prior_weight
    max_iter = reconstruction_hyperparams.max_iter
    schedule_params = [
        ScheduleVal(lambda_start, lambda_end, prior_weight, prior_weight, max_iter)
    ]

    schedule_rho, schedule_lambda = make_schedules(schedule_params)

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
        prior_lambda=schedule_lambda[0]
    ).to(device)

    # run the first N-1 iterations
    output_image_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
    pbar = tqdm.tqdm(total=n_examples)
    for low in range(0, n_examples - max_batch_size, max_batch_size):
        high = low + max_batch_size

        glm_trial_spikes_torch = example_spikes_torch[low:high, ...]

        unblind_denoise_hqs_x_prob_solver_iter = BatchParallel_ADMM_XGenerator(first_niter=300,
                                                                               subsequent_niter=300)
        unblind_denoise_hqs_z_prob_solver_iter = BatchParallel_DirectSolve_ADMM_ZGenerator()

        # if we want to use the linear reconstruction as an intermediate, do a linear reconstruction
        initialize_z_tensor = None
        if initialize_linear_model is not None:
            linear_model, linear_bin_range = initialize_linear_model
            linear_low, linear_high = linear_bin_range.start_cut, glm_trial_spikes_torch.shape[
                1] - linear_bin_range.end_cut
            with torch.no_grad():
                summed_spike_for_linear = torch.sum(glm_trial_spikes_torch[:, :, linear_low:linear_high],
                                                    dim=2)
                initialize_z_tensor = linear_model(summed_spike_for_linear)
        else:
            initialize_z_tensor = torch.randn((glm_trial_spikes_torch.shape[0], height, width),
                                              dtype=torch.float32) * initialize_noise_level

        unblind_denoise_hqs_x_prob.reinitialize_variables(initialized_z_const=initialize_z_tensor)
        unblind_denoise_hqs_x_prob.precompute_gensig_components(glm_trial_spikes_torch)
        unblind_denoise_hqs_z_prob.reinitialize_variables()

        _ = scheduled_rho_lambda_single_hqs_solve(
            unblind_denoise_hqs_x_prob,
            iter(unblind_denoise_hqs_x_prob_solver_iter),
            unblind_denoise_hqs_z_prob,
            iter(unblind_denoise_hqs_z_prob_solver_iter),
            iter(schedule_rho),
            iter(schedule_lambda),
            schedule_rho.shape[0],
            verbose=False,
            save_intermediates=False,
            observed_spikes=glm_trial_spikes_torch
        )

        denoise_hqs_reconstructed_image = unblind_denoise_hqs_x_prob.get_reconstructed_image()
        output_image_buffer_np[low:high, :, :] = denoise_hqs_reconstructed_image

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
        prior_lambda=schedule_lambda[0]
    ).to(device)

    glm_trial_spikes_torch = example_spikes_torch[low:high, ...]

    unblind_denoise_hqs_x_prob_solver_iter = BatchParallel_ADMM_XGenerator(first_niter=300,
                                                                           subsequent_niter=300)
    unblind_denoise_hqs_z_prob_solver_iter = BatchParallel_DirectSolve_ADMM_ZGenerator()

    # if we want to use the linear reconstruction as an intermediate, do a linear reconstruction
    initialize_z_tensor = None
    if initialize_linear_model is not None:
        linear_model, linear_bin_range = initialize_linear_model
        linear_low, linear_high = linear_bin_range.start_cut, glm_trial_spikes_torch.shape[
            1] - linear_bin_range.end_cut
        with torch.no_grad():
            summed_spike_for_linear = torch.sum(glm_trial_spikes_torch[:, :, linear_low:linear_high],
                                                dim=2)
            initialize_z_tensor = linear_model(summed_spike_for_linear)

    unblind_denoise_hqs_x_prob.reinitialize_variables(initialized_z_const=initialize_z_tensor)
    unblind_denoise_hqs_x_prob.precompute_gensig_components(glm_trial_spikes_torch)
    unblind_denoise_hqs_z_prob.reinitialize_variables()

    _ = scheduled_rho_lambda_single_hqs_solve(
        unblind_denoise_hqs_x_prob,
        iter(unblind_denoise_hqs_x_prob_solver_iter),
        unblind_denoise_hqs_z_prob,
        iter(unblind_denoise_hqs_z_prob_solver_iter),
        iter(schedule_rho),
        iter(schedule_lambda),
        schedule_rho.shape[0],
        verbose=False,
        save_intermediates=False,
        observed_spikes=glm_trial_spikes_torch
    )

    denoise_hqs_reconstructed_image = unblind_denoise_hqs_x_prob.get_reconstructed_image()
    output_image_buffer_np[low:high, :, :] = denoise_hqs_reconstructed_image

    pbar.update(eff_batch_size)

    pbar.close()

    del unblind_denoise_hqs_x_prob, unblind_denoise_hqs_z_prob, example_spikes_torch, unblind_denoiser_model

    return output_image_buffer_np


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Mass-generate reconstructions for each method")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('method', type=str, help='For now, either "linear", or "glm_openloop"')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-n', '--noise_init', type=float, default=1e-3)
    parser.add_argument('-l', '--linear_init', type=str, default=None,
                        help='optional, path to linear model if using linear model to initialize HQS')
    parser.add_argument('-bb', '--before_lin_bins', type=int, default=0, help='start binning for linear')
    parser.add_argument('-aa', '--after_lin_bins', type=int, default=0, help='end binning for linear')
    parser.add_argument('-hh', '--heldout', action='store_true', default=False, help='generate heldout images')

    args = parser.parse_args()

    method_lookup = args.method

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

    reference_piece_path = config_settings['ReferenceDataset'].path

    ################################################################
    # Load the cell types and matching
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()

    cell_ids_as_ordered_list = []
    for ct in ct_order:
        cell_ids_as_ordered_list.extend(cells_ordered.get_reference_cell_order(ct))

    ################################################################
    # Load some of the model fit parameters
    crop_width_low, crop_width_high = config_settings[dcp.SettingsSection.CROP_Y_LOW], config_settings[
        dcp.SettingsSection.CROP_Y_HIGH]
    crop_height_low, crop_height_high = config_settings[dcp.SettingsSection.CROP_X_LOW], config_settings[
        dcp.SettingsSection.CROP_X_HIGH]
    nscenes_downsample_factor = config_settings[dcp.SettingsSection.NSCENES_DOWNSAMPLE_FACTOR]

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_rescale_lambda = du.make_image_transform_lambda(image_rescale_low, image_rescale_high, np.float32)

    #################################################################
    # Load the raw data
    n_bins_before = config_settings[dcp.TimebinningSection.NBINS_BEFORE_TRANS]
    n_bins_after = config_settings[dcp.TimebinningSection.NBINS_AFTER_TRANS]
    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]

    # Load the natural scenes Vision datasets and determine what the
    # train and test partitions are
    nscenes_dataset_info_list = config_settings['NScenesDatasets']

    create_test_dataset = (dcp.TestDatasetSection.BINS_DSET_KEY in config_settings)
    create_heldout_dataset = (dcp.HeldoutDatasetSection.BINS_DSET_KEY in config_settings)

    test_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]
    heldout_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]

    if create_test_dataset:
        test_dataset_movie_blocks = config_settings[dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR]
    if create_heldout_dataset:
        heldout_dataset_movie_blocks = config_settings[dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR]

    nscenes_dset_list = ddu.load_nscenes_dataset_and_timebin_blocks(
        nscenes_dataset_info_list,
        samples_per_bin,
        n_bins_before,
        n_bins_after,
        test_dataset_movie_blocks,
        heldout_dataset_movie_blocks
    )

    ###############################################################
    # Load and optionally downsample/crop the stimulus frames
    train_frames, test_frames, heldout_frames = ddu.partition_downsample_frames(
        nscenes_dset_list,
        downsample_factor=nscenes_downsample_factor,
        crop_w_low=crop_width_low,
        crop_w_high=crop_width_high,
        lambda_scale_image=image_rescale_lambda
    )

    n_train_frames, n_test_frames, n_heldout_frames = train_frames.shape[0], test_frames.shape[0], heldout_frames.shape[
        0]

    _, height, width = test_frames.shape

    # we only do reconstructions from single-bin data here
    # use a separate script to do GLM reconstructions once we get those going
    if args.heldout:
        response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
            cells_ordered,
            cell_ids_as_ordered_list,
            nscenes_dset_list,
            ddu.PartitionType.HELDOUT_PARTITION
        )
    else:
        response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
            cells_ordered,
            cell_ids_as_ordered_list,
            nscenes_dset_list,
            ddu.PartitionType.TEST_PARTITION
        )

    print(response_vector.shape)

    glm_stim_time_component = make_glm_stim_time_component(config_settings)

    path_hyperparam_dict = FITTED_GLM_MODEL_PATH_DICT[reference_piece_path][method_lookup]

    fitted_glm_paths = path_hyperparam_dict['paths']
    fitted_glm_families = load_fitted_glm_families(fitted_glm_paths)
    fitted_glm_families = {key: convert_glm_family_to_np(val, spat_shape=(height, width))
                           for key, val in fitted_glm_families.items()}
    packed_glm_tensors = make_full_res_packed_glm_tensors(
        cells_ordered,
        fitted_glm_families)

    hyperparameters = path_hyperparam_dict['hyperparams']

    # Set up linear model if we choose to use it for initialization
    linear_model_param = None
    if args.linear_init is not None:
        linear_decoder = torch.load(args.linear_init, map_location=device)
        linear_bin_cutoffs = LinearModelBinningRange(args.before_lin_bins, args.after_lin_bins)
        linear_model_param = (linear_decoder, linear_bin_cutoffs)

    target_reconstructions = batch_parallel_generate_open_loop_reconstructions(
        response_vector,
        packed_glm_tensors,
        glm_stim_time_component,
        hyperparameters,
        16,
        device,
        initialize_noise_level=args.noise_init,
        initialize_linear_model=linear_model_param)

    with open(args.save_path, 'wb') as pfile:
        save_data = {
            'ground_truth': test_frames if not args.heldout else heldout_frames,
            method_lookup: target_reconstructions
        }

        pickle.dump(save_data, pfile)

    print('done')
