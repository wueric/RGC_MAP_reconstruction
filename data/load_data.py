import h5py
import numpy as np

from typing import Tuple, Dict

def load_typed_dataset() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

    with h5py.File('data/rawdata/typed_cells.hdf5', 'r') as h5_file:

        stimuli = np.array(h5_file['stimuli'], dtype=np.float32)
        binned_spikes_by_type = {
            'ON parasol': np.array(h5_file['ON parasol'], dtype=np.float32),
            'OFF parasol': np.array(h5_file['OFF parasol'], dtype=np.float32),
            'ON midget': np.array(h5_file['ON midget'], dtype=np.float32),
            'OFF midget': np.array(h5_file['OFF midget'], dtype=np.float32),
        }

    return stimuli, binned_spikes_by_type


def load_stacked_dataset() -> Tuple[np.ndarray, np.ndarray]:

    with h5py.File('data/rawdata/all_cells_stacked.hdf5', 'r') as h5_file:

        stimuli = np.array(h5_file['stimuli'], dtype=np.float32)
        spikes = np.array(h5_file['all_spikes'], dtype=np.float32)

    return stimuli, spikes
