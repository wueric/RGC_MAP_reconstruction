import h5py
import numpy as np
import torch

from typing import Tuple, Dict, Union
import pickle

from data_util.matched_cells_struct import OrderedMatchedCellsStruct
from data_util.cell_interactions import InteractionGraph

from reconstruction_alg.glm_inverse_alg import FittedGLMFamily


def load_typed_dataset() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

    with h5py.File('resources/rawdata/typed_cells.hdf5', 'r') as h5_file:

        stimuli = np.array(h5_file['stimuli'], dtype=np.float32)
        binned_spikes_by_type = {
            'ON parasol': np.array(h5_file['ON parasol'], dtype=np.float32),
            'OFF parasol': np.array(h5_file['OFF parasol'], dtype=np.float32),
            'ON midget': np.array(h5_file['ON midget'], dtype=np.float32),
            'OFF midget': np.array(h5_file['OFF midget'], dtype=np.float32),
        }

    return stimuli, binned_spikes_by_type


def load_stacked_dataset() -> Tuple[np.ndarray, np.ndarray]:

    with h5py.File('resources/rawdata/all_cells_stacked.hdf5', 'r') as h5_file:

        stimuli = np.array(h5_file['stimuli'], dtype=np.float32)
        spikes = np.array(h5_file['all_spikes'], dtype=np.float32)

    return stimuli, spikes


def load_cell_ordering() -> OrderedMatchedCellsStruct:

    with open('resources/rgcdata/2018_08_07_5_responses_ordered.p', 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)
    return cells_ordered


def load_cell_interaction_graph() -> InteractionGraph:

    with open('resources/rgcdata/2018_08_07_5_interactions.p', 'rb') as picklefile:
        pairwise_interactions = pickle.load(picklefile)
    return pairwise_interactions


def make_glm_stim_time_component() -> np.ndarray:

    n_bins_before = 250
    n_bins_after = 151
    n_bins_high = 100

    stimulus_time_component = np.zeros((n_bins_before + n_bins_after, ),
                                       dtype=np.float32)
    stimulus_time_component[n_bins_before:n_bins_before+n_bins_high] = 1.0

    return stimulus_time_component


def compute_stimulus_onset_spikes(glm_binned_spikes: Union[np.ndarray, torch.Tensor]) \
        -> Union[np.ndarray, torch.Tensor]:

    onset_bin = 250
    offset_bin = 400

    if isinstance(glm_binned_spikes, np.ndarray):
        return np.sum(glm_binned_spikes[:, onset_bin:offset_bin], axis=1)
    else:
        return torch.sum(glm_binned_spikes[:, onset_bin:offset_bin], dim=1)


def load_fitted_glm_families() \
        -> Dict[str, FittedGLMFamily]:

    glm_model_paths = {
        'ON parasol' : 'resources/encoding_model_weights/2018_08_07_5_on_parasol_full_res_glm_v4.p',
        'OFF parasol': 'resources/encoding_model_weights/2018_08_07_5_off_parasol_full_res_glm_v4.p',
        'ON midget': 'resources/encoding_model_weights/2018_08_07_5_on_midget_full_res_glm_v4.p',
        'OFF midget': 'resources/encoding_model_weights/2018_08_07_5_off_midget_full_res_glm_v4.p',
    }

    output_dict = {}  # type: Dict[str, FittedGLMFamily]
    for key, path in glm_model_paths.items():
        with open(path, 'rb') as pfile:
            output_dict[key] = pickle.load(pfile)

    return output_dict
