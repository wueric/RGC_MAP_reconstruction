import argparse
import pickle

import numpy as np

import torch
import torch.nn as nn

from data_util.load_data import load_cell_ordering, load_stacked_kim_dataset
from data_util.load_models import load_kim_hpf_nn_decoder, load_kim_deblur_network, load_kim_lpf_lin_decoder
from data_util.matched_cells_struct import OrderedMatchedCellsStruct

from kim_networks.ns_decoder import Parallel_NN_Decoder
from kim_networks.nn_deblur import ResnetGenerator
from linear_models.linear_decoding_models import ClosedFormLinearModel

def generate_full_kim_examples(linear_lpf_decoder: nn.Module,
                               hpf_decoder: nn.Module,
                               deblur_network: nn.Module,
                               ground_truth_images: np.ndarray,
                               observed_spikes: np.ndarray,
                               device: torch.device,
                               batch_size: int = 32) -> np.ndarray:

    n_tot_images, height, width = ground_truth_images.shape

    generated_images_buffer = np.zeros((n_tot_images, height, width), dtype=np.float32)
    for low in range(0, n_tot_images, batch_size):
        high = min(low + batch_size, n_tot_images)

        with torch.no_grad():

            # shape (batch, n_cells, n_bins)
            batched_spikes_torch = torch.tensor(observed_spikes[low:high, ...],
                                                dtype=torch.float32, device=device)
            batched_spikes_1bin_torch = torch.sum(batched_spikes_torch, dim=2)

            linear_reconstructed = linear_lpf_decoder(batched_spikes_1bin_torch)
            hpf_decoded = hpf_decoder(batched_spikes_torch).reshape(-1, height, width)

            combined = linear_reconstructed + hpf_decoded

            deblurred = deblur_network(combined[:, None, :, :]).squeeze(1)
            generated_images_buffer[low:high] = deblurred.detach().cpu().numpy()

    return generated_images_buffer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Generate reconstructions for Kim et al. method')
    parser.add_argument('output_path', type=str, help='save path for reconstructions')
    parser.add_argument('-b', '--batch', type=int, default=16, help='batch size for reconstruction')
    parser.add_argument('-gpu', '--gpu', action='store_true', default=False, help='use GPU')
    args = parser.parse_args()

    use_gpu= args.gpu
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    cells_ordered = load_cell_ordering() # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()
    cell_ids_as_ordered_list = []
    for ct in ct_order:
        cell_ids_as_ordered_list.extend(cells_ordered.get_reference_cell_order(ct))

    # images in the demo dataset are already cropped, and we don't do downsampling
    # images are also already rescaled to [-1, 1]
    ground_truth_images, spikes_binned = load_stacked_kim_dataset()

    n_images, n_cells, n_timebins = spikes_binned.shape
    _, height, width = ground_truth_images.shape

    lpf_linear_decoder = load_kim_lpf_lin_decoder(device) # type: ClosedFormLinearModel
    hpf_nonlinear_decoder = load_kim_hpf_nn_decoder(n_cells, n_timebins, (height, width), device) # type: Parallel_NN_Decoder
    deblur_network = load_kim_deblur_network(device) # type: ResnetGenerator

    kim_examples = generate_full_kim_examples(
        lpf_linear_decoder,
        hpf_nonlinear_decoder,
        deblur_network,
        ground_truth_images,
        spikes_binned,
        device
    )

    with open(args.output_path, 'wb') as pfile:
        save_data = {
            'ground_truth': ground_truth_images,
            'kim_post_deblur': kim_examples,
        }

        pickle.dump(save_data, pfile)

    print('done')
