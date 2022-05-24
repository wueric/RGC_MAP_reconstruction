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
import math
import tqdm

import numpy as np
import torch

from encoding_models.poisson_encoder import ScaledPoissonFittedParams, SharedStimMulticellScaleOnlyPoisson

from convex_solver_base.unconstrained_optim import FistaSolverParams, UnconstrainedSolverParams, \
    batch_parallel_unconstrained_solve

from data_util.matched_cells_struct import OrderedMatchedCellsStruct
from data_util.load_data import load_typed_dataset, load_cell_ordering, load_cropped_stas_by_type, \
    compute_stimulus_onset_spikes, get_fit_cell_binned_spikes_lnp


def parallel_no_window_fit_scaled_poissons(all_images: np.ndarray,
                                           cell_response_vector: np.ndarray,
                                           filters_for_cell: np.ndarray,
                                           solver_params: UnconstrainedSolverParams,
                                           device: torch.device,
                                           batch_size=16) -> List[ScaledPoissonFittedParams]:

    n_trials, frame_height, frame_width = all_images.shape
    n_cells = len(sta_list)

    # we have to do this in batches otherwise we run out of GPU space for the midgets
    # also might speed the whole thing up

    full_stimuli = torch.tensor(all_images.reshape(n_trials, -1), dtype=torch.float32, device=device)

    fitted_model_list = []  # type: List[ScaledPoissonFittedParams]

    pbar = tqdm.tqdm(total=int(math.ceil(n_cells / batch_size)))
    for low in range(0, n_cells, batch_size):
        high = min(low + batch_size, n_cells)
        curr_batch_size = high - low

        relevant_filters = filters_for_cell[low:high, ...].reshape((curr_batch_size, -1))
        relevant_spikes = cell_response_vector[:, low:high].T

        relevant_spikes_torch = torch.tensor(relevant_spikes, dtype=torch.float32, device=device)

        # make the model
        multicell_poisson_model = SharedStimMulticellScaleOnlyPoisson(
            relevant_filters,
        ).to(device)  # type: SharedStimMulticellScaleOnlyPoisson

        losses = batch_parallel_unconstrained_solve(
            multicell_poisson_model,
            solver_params,
            verbose=True,
            shared_stimulus=full_stimuli,
            multicell_spikes=relevant_spikes_torch
        ).detach().cpu().numpy()

        # then extract all of the fitted parameters
        trained_filters, trained_biases = multicell_poisson_model.get_filters_and_biases()
        for idx in range(curr_batch_size):
            fitted_model_list.append(ScaledPoissonFittedParams(
                trained_filters[idx, ...].reshape((frame_height, frame_width)),
                trained_biases[idx],
                losses[idx],
            ))

        pbar.update(1)

        del relevant_filters
        del multicell_poisson_model

    del full_stimuli

    pbar.close()

    return fitted_model_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fit scaled single-bin LNP models to flashed trial data')
    parser.add_argument('save_path', type=str, help='path to save model fits')
    parser.add_argument('-gpu', '--gpu', action='store_true', default=False, help='use GPU')
    args = parser.parse_args()

    use_gpu = args.gpu
    device = torch.device('cuda') if use_gpu else torch.device('cpu')

    ################################################################
    # get the data
    # IMPORTANT: THIS IS A TINY SUBSET OF THE ACTUAL DATASET THAT WAS
    # USED TO FIT THE GLM; THESE FITS ARE NOT GOING TO BE VERY GOOD BY
    # DEFINITION
    train_frames, train_spikes_dict = load_typed_dataset()
    train_spikes_lnp_dict = {cell_type: compute_stimulus_onset_spikes(glm_spikes)
                             for cell_type, glm_spikes in train_spikes_dict.items()}

    ################################################################
    # load the STA prior
    all_prior_filters = load_cropped_stas_by_type()

    ################################################################
    # Load the cell types and matching
    cells_ordered = load_cell_ordering()  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()

    #### Train the scaled Poisson for every cell #################################
    typed_model_dict = {}  # type: Dict[str, List[ScaledPoissonFittedParams]]
    for cell_type in ct_order:
        print("Fitting {0}".format(cell_type))

        reference_id_list = cells_ordered.get_reference_cell_order(cell_type)
        cell_response_vector = get_fit_cell_binned_spikes_lnp(train_spikes_lnp_dict,
                                                              cell_type,
                                                              reference_id_list,
                                                              cells_ordered)

        sta_list = all_prior_filters[cell_type]

        all_fits = parallel_no_window_fit_scaled_poissons(
            train_frames,
            cell_response_vector,
            sta_list,
            FistaSolverParams(initial_learning_rate=1.0,
                              max_iter=100,
                              backtracking_beta=0.5,
                              converge_epsilon=1e-6),
            device,
            batch_size=32  # in reality, we use 2 for the full dataset
        )

        typed_model_dict[cell_type] = all_fits

    with open(args.save_path, 'wb') as pfile:
        pickle.dump(typed_model_dict, pfile)
