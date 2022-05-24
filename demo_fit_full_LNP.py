import argparse
import pickle
from typing import Dict, List, Tuple, Union, Optional
import math

import tqdm

import numpy as np
import torch

from encoding_models.poisson_encoder import ScaledPoissonFittedParams, SharedStimMulicellPoisson
from convex_solver_base.prox_optim import ProxSolverParams, ProxFISTASolverParams, batch_parallel_prox_solve

from data_util.matched_cells_struct import OrderedMatchedCellsStruct
from data_util.load_data import load_typed_dataset, load_cell_ordering, load_cropped_stas_by_type, \
    compute_stimulus_onset_spikes, get_fit_cell_binned_spikes_lnp
from hyperparameters.lnp_hyperparameters import LNP_HYPERPARAMETERS


def parallel_no_window_fit_full_poisson(all_images: np.ndarray,
                                        cell_response_vector: np.ndarray,
                                        scaled_fit_prior_list: List[ScaledPoissonFittedParams],
                                        fitting_parameters: Dict[str, float],
                                        solver_params: ProxSolverParams,
                                        device: torch.device,
                                        batch_size=16) -> List[ScaledPoissonFittedParams]:
    '''
    Fits the Poisson model for every cell using gradient descent, in parallel

    Takes care of zero padding in order to make sure the matrix sizes work out
        (by default some of the edge cells will have smaller dimensions; this
        takes care of that)
    :param training_dataset:
    :param bounding_box_dict:
    :param sta_dict:
    :param device:
    :param downsample_factor:
    :param crop_width_low:
    :param crop_height_low:
    :param learning_rate:
    :param max_iter:
    :param log_freq:
    :return:
    '''

    n_trials, frame_height, frame_width = all_images.shape
    _, n_cells = cell_response_vector.shape

    fitted_model_params = []

    filters_for_cell = np.zeros((n_cells, frame_height, frame_width), dtype=np.float32)
    for idx, prior_fit in enumerate(scaled_fit_prior_list):
        filters_for_cell[idx, :, :] = prior_fit.filter

    full_stimuli = torch.tensor(all_images.reshape(n_trials, -1), dtype=torch.float32, device=device)

    pbar = tqdm.tqdm(total=int(math.ceil(n_cells / batch_size)))
    for low in range(0, n_cells, batch_size):
        high = min(low + batch_size, n_cells)

        curr_batch_size = high - low

        relevant_filters = filters_for_cell[low:high, ...].reshape((curr_batch_size, -1))
        relevant_spikes = cell_response_vector[:, low:high].T

        relevant_spikes_torch = torch.tensor(relevant_spikes, dtype=torch.float32, device=device)

        # make the model
        multicell_poisson_model = SharedStimMulicellPoisson(
            relevant_filters,
            fitting_parameters['l1'],
            fitting_parameters['l2'],
            initial_filter_values=relevant_filters + np.random.randn(*relevant_filters.shape) * 0.01,
        ).to(device)  # type: SharedStimMulicellPoisson

        # now train
        losses = batch_parallel_prox_solve(multicell_poisson_model,
                                           solver_params,
                                           verbose=True,
                                           shared_stimulus=full_stimuli,
                                           multicell_spikes=relevant_spikes_torch)

        # then extract all of the fitted parameters
        trained_filters, trained_biases = multicell_poisson_model.get_filters_and_biases()
        for idx in range(curr_batch_size):
            fitted_model_params.append(ScaledPoissonFittedParams(
                trained_filters[idx, ...].reshape((frame_height, frame_width)),
                trained_biases[idx],
                losses[idx].item(),
            ))

        pbar.update(1)

        del multicell_poisson_model
        del relevant_spikes_torch

    pbar.close()

    del full_stimuli

    return fitted_model_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fit scaled single-bin LNP models to flashed trial data')
    parser.add_argument('scale_fit_path', type=str, help='path to previously-computed scaled fits')
    parser.add_argument('save_path', type=str, help='path to save model fits')
    parser.add_argument('-gpu', '--gpu', action='store_true', default=False, help='use GPU')
    args = parser.parse_args()

    use_gpu = args.gpu
    device = torch.device('cuda') if use_gpu else torch.device('cpu')

    #### Load the previous scaled fits #############################
    scale_fit_path = args.scale_fit_path
    with open(scale_fit_path, 'rb') as pfile:
        previous_fit_prior = pickle.load(pfile)

    ################################################################
    # Load the cell types and matching
    cells_ordered = load_cell_ordering()  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()

    ################################################################
    # get the data
    # IMPORTANT: THIS IS A TINY SUBSET OF THE ACTUAL DATASET THAT WAS
    # USED TO FIT THE GLM; THESE FITS ARE NOT GOING TO BE VERY GOOD BY
    # DEFINITION
    train_frames, train_spikes_dict = load_typed_dataset()
    train_spikes_lnp_dict = {cell_type: compute_stimulus_onset_spikes(glm_spikes)
                             for cell_type, glm_spikes in train_spikes_dict.items()}

    typed_model_dict = {}  # type: Dict[str, List[ScaledPoissonFittedParams]]
    for cell_type in ct_order:

        print("Fitting {0}".format(cell_type))

        reference_id_list = cells_ordered.get_reference_cell_order(cell_type)
        cell_response_vector = get_fit_cell_binned_spikes_lnp(train_spikes_lnp_dict,
                                                              cell_type,
                                                              reference_id_list,
                                                              cells_ordered)

        fitting_params_dict = LNP_HYPERPARAMETERS[cell_type]
        previous_fit_prior_list = previous_fit_prior[cell_type]

        if fitting_params_dict['method'] == 'full':
            all_fits = parallel_no_window_fit_full_poisson(
                train_frames,
                cell_response_vector,
                previous_fit_prior_list,
                fitting_params_dict,
                ProxFISTASolverParams(initial_learning_rate=1.0,
                                      max_iter=fitting_params_dict['n_iter'],
                                      backtracking_beta=0.5,
                                      converge_epsilon=fitting_params_dict['eps']),
                device,
                batch_size=2
            )

            typed_model_dict[cell_type] = all_fits

    with open(args.save_path, 'wb') as pfile:
        pickle.dump(typed_model_dict, pfile)
