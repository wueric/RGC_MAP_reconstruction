import pickle
from typing import Dict, List, Tuple, Union, Optional, Callable, Sequence, Any

import torch
import numpy as np

import argparse

from autoencoder.autoencoder import Neurips2017_Decoder, Neurips2017_Encoder

from data_util.matched_cells_struct import OrderedMatchedCellsStruct
from data_util.load_data import load_cell_ordering, load_stacked_dataset, compute_stimulus_onset_spikes
from data_util.load_models import load_linear_reconstruction_model, load_lcae_encoder_and_decoder


def generate_autoencoder_images(linear_reconstructor,
                                encoder: Neurips2017_Encoder,
                                decoder: Neurips2017_Decoder,
                                ground_truth_images: np.ndarray,
                                observed_spikes: np.ndarray,
                                device: torch.device,
                                batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    n_tot_images, height, width = ground_truth_images.shape

    generated_image_buffer = np.zeros((n_tot_images, height, width), dtype=np.float32)
    generated_linear_image_buffer = np.zeros((n_tot_images, height, width), dtype=np.float32)
    for low in range(0, n_tot_images, batch_size):
        high = min(low + batch_size, n_tot_images)

        with torch.no_grad():
            # shape (batch, n_cells)
            batched_spikes_torch = torch.tensor(observed_spikes[low:high, ...],
                                                dtype=torch.float32, device=device)

            # shape (batch, height, width)
            linear_reconstructed = linear_reconstructor(batched_spikes_torch)

            # shape (batch, ?, ??)
            encoded_images = encoder(linear_reconstructed)

            # shape (batch, height, width)
            decoded_images_np = decoder(encoded_images).detach().cpu().numpy()

            generated_image_buffer[low:high, ...] = decoded_images_np
            generated_linear_image_buffer[low:high, ...] = linear_reconstructed.detach().cpu().numpy()

    return generated_image_buffer, generated_linear_image_buffer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'Generate reconstructions for L-CAE method')
    parser.add_argument('output_path', type=str, help='save path for reconstructions')
    parser.add_argument('-b', '--batch', type=int, default=16, help='batch size for reconstruction')
    parser.add_argument('-gpu', '--gpu', action='store_true', help='use GPU')
    args = parser.parse_args()

    use_gpu = args.gpu
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    cells_ordered = load_cell_ordering()  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()
    cell_ids_as_ordered_list = []
    for ct in ct_order:
        cell_ids_as_ordered_list.extend(cells_ordered.get_reference_cell_order(ct))

    ground_truth_images, spikes_binned = load_stacked_dataset()
    onset_spikes_only = compute_stimulus_onset_spikes(spikes_binned)

    _, height, width = ground_truth_images.shape
    n_cells = onset_spikes_only.shape[1]

    linear_decoder = load_linear_reconstruction_model(n_cells, (height, width), device)
    lcae_encoder, lcae_decoder = load_lcae_encoder_and_decoder(device)

    ######## Generate the autoencoder reconstructions ########################
    autoencoder_reconstructed_images, linear_reconstructed_images = generate_autoencoder_images(
        linear_decoder,
        lcae_encoder,
        lcae_decoder,
        ground_truth_images,
        onset_spikes_only,
        device,
        batch_size=args.batch
    )

    with open(args.output_path, 'wb') as pfile:
        save_data = {
            'ground_truth': ground_truth_images,
            'autoencoder': autoencoder_reconstructed_images,
            'linear': linear_reconstructed_images
        }

        pickle.dump(save_data, pfile)

    print('done')
