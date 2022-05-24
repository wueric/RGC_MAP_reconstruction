import h5py
import numpy as np
import torch

from typing import Tuple, Dict, Union, List
import pickle

from data_util.matched_cells_struct import OrderedMatchedCellsStruct
from data_util.cell_interactions import InteractionGraph


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "lib.data_utils.matched_cells_struct":
            renamed_module = "data_util.matched_cells_struct"
        elif name == 'FittedGLMFamily' or name == 'FittedGLM' and module == "optimization_encoder.trial_glm":
            renamed_module="reconstruction_alg.glm_inverse_alg"
        elif name == 'ScaledPoissonFittedParams' and module == 'optimization_encoder.poisson_encoder':
            renamed_module='encoding_models.poisson_encoder'
        elif module == "lib.data_utils.interaction_graph" or module == "lib.data_utils.interaction_hashable":
            renamed_module = "data_util.cell_interactions"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


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


def get_fit_cell_binned_spikes_lnp(spikes_dict: Dict[str, np.ndarray],
                                   cell_type: str,
                                   relevant_id_list: List[int],
                                   cell_ordering: OrderedMatchedCellsStruct) -> np.ndarray:

    relevant_idx = cell_ordering.get_idx_for_same_type_cell_id_list(cell_type, relevant_id_list)
    return spikes_dict[cell_type][:, relevant_idx]


def get_fit_cell_binned_spikes_glm(spikes_dict: Dict[str, np.ndarray],
                                   cell_type: str,
                                   cell_id: int,
                                   cell_ordering: OrderedMatchedCellsStruct) -> np.ndarray:
    '''

    :param spikes_dict:
    :param cell_type:
    :param cell_id:
    :param cell_ordering:
    :return: shape (n_trials, n_bins)
    '''

    relevant_idx = cell_ordering.get_idx_for_same_type_cell_id_list(cell_type, [cell_id])[0]
    return spikes_dict[cell_type][:, relevant_idx, :]


def get_coupled_cells_binned_spikes_glm(spikes_dict: Dict[str, np.ndarray],
                                        coupled_cells_ordered_by_type: Dict[str, List[int]],
                                        cell_ordering: OrderedMatchedCellsStruct,
                                        return_stacked_array: bool = True) -> Union[Dict[str, np.ndarray], np.ndarray]:

    binned_spikes_by_type = {} # type: Dict[str, np.ndarray]
    for cell_type, coupled_cell_ids in coupled_cells_ordered_by_type.items():
        relevant_indices = cell_ordering.get_idx_for_same_type_cell_id_list(cell_type, coupled_cell_ids)

        binned_spikes_by_type[cell_type] = spikes_dict[cell_type][:, relevant_indices, :]

    if return_stacked_array:
        ct_order = cell_ordering.get_cell_types()
        return np.concatenate([binned_spikes_by_type[ct] for ct in ct_order], axis=1)

    return binned_spikes_by_type


def load_typed_kim_dataset() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    with h5py.File('resources/rawdata/kim_typed_cells.hdf5', 'r') as h5_file:

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


def load_stacked_kim_dataset() -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File('resources/rawdata/kim_all_cells_stacked.hdf5', 'r') as h5_file:

        stimuli = np.array(h5_file['stimuli'], dtype=np.float32)
        spikes = np.array(h5_file['all_spikes'], dtype=np.float32)

    return stimuli, spikes



def load_cell_ordering() -> OrderedMatchedCellsStruct:

    with open('resources/rgcdata/2018_08_07_5_responses_ordered.p', 'rb') as ordered_cells_file:
        cells_ordered = renamed_load(ordered_cells_file)
    return cells_ordered


def load_cell_interaction_graph() -> InteractionGraph:

    with open('resources/rgcdata/2018_08_07_5_interactions.p', 'rb') as picklefile:
        pairwise_interactions = renamed_load(picklefile)
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
        return np.sum(glm_binned_spikes[:, :, onset_bin:offset_bin], axis=2)
    else:
        return torch.sum(glm_binned_spikes[:, :, onset_bin:offset_bin], dim=2)


def load_timecourse_1ms_by_type() -> Dict[str, np.ndarray]:
    with open('resources/rgcdata/2018_08_07_5_timecourse.p', 'rb') as pfile:
        return pickle.load(pfile)


def load_cropped_stas_by_type() -> Dict[str, np.ndarray]:
    with open('resources/rgcdata/2018_08_07_5_cropped_rfs.p', 'rb') as pfile:
        return pickle.load(pfile)
