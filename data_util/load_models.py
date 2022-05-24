from typing import Dict, List, Tuple

import numpy as np
import torch

from autoencoder.autoencoder import Neurips2017_Encoder, Neurips2017_Decoder
from data_util.load_data import renamed_load
from encoding_models.poisson_encoder import reinflate_uncropped_poisson_model
from kim_networks.nn_deblur import ResnetGenerator
from kim_networks.ns_decoder import Parallel_NN_Decoder
from linear_models.linear_decoding_models import ClosedFormLinearModel
from reconstruction_alg.glm_inverse_alg import FittedGLMFamily

import pickle


def load_fitted_glm_families() \
        -> Dict[str, FittedGLMFamily]:

    glm_model_paths = {
        'ON parasol' : 'resources/encoding_model_weights/glm/2018_08_07_5_on_parasol_glm_cpu_v4.p',
        'OFF parasol': 'resources/encoding_model_weights/glm/2018_08_07_5_off_parasol_glm_cpu_v4.p',
        'ON midget': 'resources/encoding_model_weights/glm/2018_08_07_5_on_midget_glm_cpu_v4.p',
        'OFF midget': 'resources/encoding_model_weights/glm/2018_08_07_5_off_midget_glm_cpu_v4.p',
    }

    output_dict = {}  # type: Dict[str, FittedGLMFamily]
    for key, path in glm_model_paths.items():
        with open(path, 'rb') as pfile:
            output_dict[key] = renamed_load(pfile)

    return output_dict


def load_fitted_lnps(ct_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:

    with open('resources/encoding_model_weights/lnp/2018_08_07_5_lnp_weights.p', 'rb') as pfile:
        all_poisson_fits = renamed_load(pfile)
        reinflated_filters, reinflated_biases = reinflate_uncropped_poisson_model(
            all_poisson_fits,
            ct_order
        )

    return reinflated_filters, reinflated_biases


def load_kim_lpf_lin_decoder(device: torch.device) -> ClosedFormLinearModel:
    with open('resources/kim_model_weights/kim_lpf_lin_weights.p', 'rb') as pfile:
        summary_dict = pickle.load(pfile)
        lpf_decoder_filters = summary_dict['coeffs']

    # put this into a linear reconstructor module
    linear_decoder = ClosedFormLinearModel(
        lpf_decoder_filters.shape[0],
        lpf_decoder_filters.shape[1],
        lpf_decoder_filters.shape[2]
    )

    decoder_filters_torch = torch.tensor(lpf_decoder_filters, dtype=torch.float32)
    linear_decoder.set_linear_filters(decoder_filters_torch)
    linear_decoder = linear_decoder.eval()

    return linear_decoder.to(device)


def load_kim_hpf_nn_decoder(n_cells: int,
                            n_timebins: int,
                            imshape: Tuple[int, int],
                            device: torch.device) -> Parallel_NN_Decoder:

    K_DIM = 25
    H_DIM = 40
    F_DIM = 5

    height, width = imshape

    #####################################################
    # load the pixel weights from the L1 problem
    with open('resources/kim_model_weights/kim_selection_coeffs.p', 'rb') as pfile:
        summary_dict = pickle.load(pfile)
        reconstruction_coeffs = summary_dict['coeffs']

    # figure out which cells are assigned to which pixels
    abs_recons_coeffs = np.abs(reconstruction_coeffs)
    argsort_by_cell = np.argsort(abs_recons_coeffs, axis=0)

    # shape (k_dim, height, width)
    biggest_coeffs_ix = argsort_by_cell[-K_DIM:, ...]

    # shape (n_pix, k_dim)
    flattened_coeff_sel = biggest_coeffs_ix.reshape(biggest_coeffs_ix.shape[0], -1).transpose(1, 0)

    # then load the nonlinear network$a
    hpf_decoder_nn_model = Parallel_NN_Decoder(flattened_coeff_sel,
                                               n_cells,
                                               n_timebins,
                                               K_DIM,
                                               H_DIM,
                                               height * width,
                                               F_DIM).to(device)
    saved_hpf_network = torch.load('resources/kim_model_weights/kim_hpf_decoder.pt',
                                   map_location=device)
    hpf_decoder_nn_model.load_state_dict(saved_hpf_network['decoder'])
    hpf_decoder_nn_model = hpf_decoder_nn_model.eval()

    return hpf_decoder_nn_model


def load_kim_deblur_network(device: torch.device) -> ResnetGenerator:

    deblur_network = ResnetGenerator(input_nc=1,
                                     output_nc=1,
                                     n_blocks=6).to(device)
    saved_deblur_network = torch.load('resources/kim_model_weights/kim_deblur.pt', map_location=device)
    deblur_network.load_state_dict(saved_deblur_network['deblur'])
    deblur_network = deblur_network.eval()
    return deblur_network


class LinearDecoderRenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == 'linear_decoding_models.linear_decoding_models':
            renamed_module = 'linear_models.linear_decoding_models'
        return super(LinearDecoderRenameUnpickler, self).find_class(renamed_module, name)


def load_linear_reconstruction_model(n_cells: int,
                                     imshape: Tuple[int, int],
                                     device: torch.device) -> ClosedFormLinearModel:

    linear_decoder= ClosedFormLinearModel(n_cells,
                                          imshape[0],
                                          imshape[1])


    linear_decoder_state_dict = torch.load('resources/LCAE_weights/linear_filters.pt',
                                map_location=device)
    linear_decoder.load_state_dict(linear_decoder_state_dict)
    linear_decoder.eval()
    return linear_decoder


def load_lcae_encoder_and_decoder(device: torch.device) \
        -> Tuple[Neurips2017_Encoder, Neurips2017_Decoder]:

    encoder = Neurips2017_Encoder(0.25).to(device)
    decoder = Neurips2017_Decoder(0.25).to(device)

    saved_autoencoder = torch.load('resources/LCAE_weights/autoencoder.pt', map_location=device)
    encoder.load_state_dict(saved_autoencoder['encoder'])
    decoder.load_state_dict(saved_autoencoder['decoder'])

    encoder.eval()
    decoder.eval()

    return encoder, decoder
