import pickle

import numpy as np
import torch

from data_util.matched_cells_struct import OrderedMatchedCellsStruct
from data_util.load_data import make_glm_stim_time_component, compute_stimulus_onset_spikes, \
    load_stacked_dataset, load_cell_ordering
from data_util.load_models import load_fitted_glm_families

from simple_priors.gaussian_prior import make_zca_gaussian_prior_matrix

from convex_solver_base.unconstrained_optim import batch_parallel_unconstrained_solve
from convex_solver_base.unconstrained_optim import FistaSolverParams, GradientSolverParams
from reconstruction_alg.glm_inverse_alg import BatchParallelPatchGaussian1FPriorGLMReconstruction, \
    batch_per_bin_bernoulli_neg_log_likelihood, FittedGLM, FittedGLMFamily, PackedGLMTensors, \
    make_full_res_packed_glm_tensors

from hyperparameters.hyperparameters import glm_1F_exact_MAP_hyperparameters_2018_08_07_5

import argparse
import tqdm


def batch_parallel_generate_1F_reconstructions(
        example_spikes: np.ndarray,
        packed_glm_tensors: PackedGLMTensors,
        glm_time_component: np.ndarray,
        prior_zca_matrix_imshape: np.ndarray,
        prior_lambda_weight: float,
        max_batch_size: int,
        device: torch.device,
        patch_stride: int = 2) -> np.ndarray:

    '''

    :param example_spikes:
    :param packed_glm_tensors:
    :param glm_time_component:
    :param prior_zca_matrix_imshape: shape (patch_height, patch_width, patch_height, patch_width)
    :param prior_lambda_weight:
    :param max_batch_size:
    :param device:
    :return:
    '''

    n_examples, n_cells, n_bins_observed = example_spikes.shape
    _, height, width = packed_glm_tensors.spatial_filters.shape

    example_spikes_torch = torch.tensor(example_spikes, dtype=torch.float32, device=device)

    batch_gaussian_problem = BatchParallelPatchGaussian1FPriorGLMReconstruction(
        max_batch_size,
        prior_zca_matrix_imshape,
        prior_lambda_weight,
        packed_glm_tensors.spatial_filters,
        packed_glm_tensors.timecourse_filters,
        packed_glm_tensors.feedback_filters,
        packed_glm_tensors.coupling_filters,
        packed_glm_tensors.coupling_indices,
        packed_glm_tensors.bias,
        glm_time_component,
        batch_per_bin_bernoulli_neg_log_likelihood,
        patch_stride=patch_stride
    ).to(device)

    # run the first N-1 iterations
    output_image_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
    pbar = tqdm.tqdm(total=n_examples)
    for low in range(0, n_examples - max_batch_size, max_batch_size):
        high = low + max_batch_size

        glm_trial_spikes_torch = example_spikes_torch[low:high, ...]
        batch_gaussian_problem.precompute_gensig_components(glm_trial_spikes_torch)

        _ = batch_parallel_unconstrained_solve(
            batch_gaussian_problem,
            FistaSolverParams(
                initial_learning_rate=1.0,
                max_iter=250,
                converge_epsilon=1e-6,
                backtracking_beta=0.5
            ),
            verbose=False,
            observed_spikes=glm_trial_spikes_torch,
        )

        reconstructed_image = batch_gaussian_problem.get_reconstructed_image()
        output_image_buffer_np[low:high, :, :] = reconstructed_image

        pbar.update(max_batch_size)

    del batch_gaussian_problem

    # run the final iteration
    low = ((n_examples - 1) // max_batch_size) * max_batch_size
    high = n_examples

    eff_batch_size = high - low

    batch_gaussian_problem = BatchParallelPatchGaussian1FPriorGLMReconstruction(
        eff_batch_size,
        prior_zca_matrix_imshape,
        prior_lambda_weight,
        packed_glm_tensors.spatial_filters,
        packed_glm_tensors.timecourse_filters,
        packed_glm_tensors.feedback_filters,
        packed_glm_tensors.coupling_filters,
        packed_glm_tensors.coupling_indices,
        packed_glm_tensors.bias,
        glm_time_component,
        batch_per_bin_bernoulli_neg_log_likelihood,
        patch_stride=patch_stride
    ).to(device)

    glm_trial_spikes_torch = example_spikes_torch[low:high, ...]
    batch_gaussian_problem.precompute_gensig_components(glm_trial_spikes_torch)

    _ = batch_parallel_unconstrained_solve(
        batch_gaussian_problem,
        FistaSolverParams(
            initial_learning_rate=1.0,
            max_iter=250,
            converge_epsilon=1e-6,
            backtracking_beta=0.5
        ),
        verbose=False,
        observed_spikes=glm_trial_spikes_torch,
    )

    reconstructed_image = batch_gaussian_problem.get_reconstructed_image()
    output_image_buffer_np[low:high, :, :] = reconstructed_image

    pbar.update(eff_batch_size)
    pbar.close()

    del batch_gaussian_problem
    return output_image_buffer_np


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Mass-generate reconstructions for MAP-GLM-dCNN method")
    parser.add_argument('output_path', type=str, help='save path for reconstructions')
    parser.add_argument('-b', '--batch', type=int, default=16, help='batch size for reconstruction')
    parser.add_argument('-l', '--linear_init', action='store_true', default=False, help='use linear initialization')
    parser.add_argument('-gpu', '--gpu', action='store_true', default=False, help='use GPU')
    args = parser.parse_args()

    use_gpu = args.gpu
    device = torch.device('cuda') if use_gpu else torch.device('cpu')

    ################################################################
    # Load the cell types and matching
    cells_ordered = load_cell_ordering()  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()
    cell_ids_as_ordered_list = []
    for ct in ct_order:
        cell_ids_as_ordered_list.extend(cells_ordered.get_reference_cell_order(ct))

    #################################################################
    # images in the demo dataset are already cropped, and we don't do downsampling
    # images are also already rescaled to [-1, 1]
    ground_truth_images, spikes_binned = load_stacked_dataset()
    glm_stim_time_component = make_glm_stim_time_component()

    # load the GLM model and hyperparameters
    fitted_glm_families = load_fitted_glm_families()
    packed_glm_tensors = make_full_res_packed_glm_tensors(
        cells_ordered,
        fitted_glm_families)

    hyperparameters = glm_1F_exact_MAP_hyperparameters_2018_08_07_5()

    patch_height, patch_width = hyperparameters.patch_height, hyperparameters.patch_width
    gaussian_zca_mat = make_zca_gaussian_prior_matrix((patch_height, patch_width),
                                                      dc_multiple=1.0)
    gaussian_zca_mat_imshape = (gaussian_zca_mat.reshape((patch_height, patch_width, patch_height, patch_width)))

    target_reconstructions = batch_parallel_generate_1F_reconstructions(
        spikes_binned,
        packed_glm_tensors,
        glm_stim_time_component,
        gaussian_zca_mat_imshape,
        hyperparameters.prior_weight,
        4,
        device)

    with open(args.output_path, 'wb') as pfile:
        save_data = {
            'ground_truth': ground_truth_images,
            'MAP_GLM_1F': target_reconstructions
        }

        pickle.dump(save_data, pfile)

    print('done')
