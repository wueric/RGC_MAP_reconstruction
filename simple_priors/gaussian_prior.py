import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Tuple


def make_zca_gaussian_prior_matrix(patch_shape: Tuple[int, int],
                                   dc_multiple: float = 2.0) -> np.ndarray:

    height, width = patch_shape
    max_dim = max(height, width)

    fft_freqs_y = np.fft.fftfreq(height).astype(np.float32)
    fft_freqs_x = np.fft.fftfreq(width).astype(np.float32)

    freqs_y_square = np.square(fft_freqs_y)
    freqs_x_square = np.square(fft_freqs_x)

    outer_norm_prod = np.sqrt(freqs_y_square[:, None] + freqs_x_square[None, :])# / max_dim

    fy = (1j * np.pi * 2 * fft_freqs_y).astype(np.csingle)
    fx = (1j * np.pi * 2 * fft_freqs_x).astype(np.csingle)

    del fft_freqs_y, fft_freqs_x

    ny = np.r_[0:height].astype(np.float32)
    nx = np.r_[0:width].astype(np.float32)

    # different frequencies -> different basis vectors
    exp_arg_y = fy[:, None] * ny[None, :]
    exp_arg_x = fx[:, None] * nx[None, :]

    del nx, ny

    # shape (height, height), each row is a different basis vector
    y_fourier_basis = np.exp(exp_arg_y)

    # shape (width, width), each row is a different basis vector
    x_fourier_basis = np.exp(exp_arg_x)

    print('Computing Fourier outer product')

    # shape (height, width, height, width)
    xy_fourier_basis = y_fourier_basis[:, None, :, None] * x_fourier_basis[None, :, None, :]

    xy_fourier_basis_flat = xy_fourier_basis.reshape(height * width, height * width)

    temporary_nonsense_val = 1e9
    outer_norm_prod[0, 0] = temporary_nonsense_val
    min_val = np.min(outer_norm_prod)
    outer_norm_prod[0, 0] = min_val / dc_multiple

    root_power_freq_flat = outer_norm_prod.reshape(-1)

    zca_mat = xy_fourier_basis_flat.T @ (root_power_freq_flat[:, None] * xy_fourier_basis_flat.conj())

    return np.real(zca_mat) / np.sqrt(height * width)


def make_full_gaussian_prior_precision_matrix(image_shape: Tuple[int, int],
                                              dc_multiple: float = 2.0) -> np.ndarray:
    '''
    Makes the Gaussian 1/f image prior precision matrix

    (Precision matrix instead
    :param image_shape:
    :param dc_multiple:
    :return:
    '''

    height, width = image_shape
    max_dim = max(height, width)

    fft_freqs_y = np.fft.fftfreq(height).astype(np.float32)
    fft_freqs_x = np.fft.fftfreq(width).astype(np.float32)

    freqs_y_square = np.square(fft_freqs_y)
    freqs_x_square = np.square(fft_freqs_x)

    outer_norm_prod = np.sqrt(freqs_y_square[:, None] + freqs_x_square[None, :])# / max_dim

    fy = (1j * np.pi * 2 * fft_freqs_y).astype(np.csingle)
    fx = (1j * np.pi * 2 * fft_freqs_x).astype(np.csingle)

    del fft_freqs_y, fft_freqs_x

    ny = np.r_[0:height].astype(np.float32)
    nx = np.r_[0:width].astype(np.float32)

    # different frequencies -> different basis vectors
    exp_arg_y = fy[:, None] * ny[None, :]
    exp_arg_x = fx[:, None] * nx[None, :]

    del nx, ny

    # shape (height, height), each row is a different basis vector
    y_fourier_basis = np.exp(exp_arg_y)

    # shape (width, width), each row is a different basis vector
    x_fourier_basis = np.exp(exp_arg_x)

    print('Computing Fourier outer product')

    xy_fourier_basis = y_fourier_basis[:, None, :, None] * x_fourier_basis[None, :, None, :]

    xy_fourier_basis_flat = xy_fourier_basis.reshape(height * width, height * width)

    temporary_nonsense_val = 1e9
    outer_norm_prod[0, 0] = temporary_nonsense_val
    min_val = np.min(outer_norm_prod)
    outer_norm_prod[0, 0] = min_val / dc_multiple

    power_freq = np.power(outer_norm_prod, 2.0)
    power_freq_flat = power_freq.reshape(-1)

    precision_mat = xy_fourier_basis_flat.T @ (power_freq_flat[:, None] * xy_fourier_basis_flat.conj())

    return np.real(precision_mat).astype(np.float32)


class Full1FGaussianPrior(nn.Module):

    def __init__(self,
                 gaussian_prior_reconstruction_matrix: np.ndarray,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dtype = dtype
        self.register_buffer('gaussian_prec_matrix',
                             torch.tensor(gaussian_prior_reconstruction_matrix, dtype=self.dtype))

    def forward(self, image_imshape: torch.Tensor) -> torch.Tensor:
        '''

        :param image_imshape: shape (height, width)
        :return:
        '''
        # shape (n_pixels, )
        image_flat = image_imshape.reshape(-1)

        # shape (n_pixels, n_pixels) @ (n_pixels, 1) -> (n_pixels, 1)
        left_mul = self.gaussian_prec_matrix @ image_flat[:, None]

        regularization_term = (image_flat[None, :] @ left_mul).squeeze()

        return regularization_term


class ConvPatch1FGaussianPrior(nn.Module):

    def __init__(self,
                 zca_matrix_conv_shape: np.ndarray,
                 patch_stride: int = 1,
                 dtype: torch.dtype = torch.float32):
        '''

        :param prec_matrix_conv_shape: shape (patch_height, patch_width, patch_height, patch_width)
        :param patch_stride:
        :param dtype:
        '''
        super().__init__()

        self.dtype = dtype
        self.stride = patch_stride

        self.ph, self.pw = zca_matrix_conv_shape.shape[2:]
        self.n_pix_p = self.ph * self.pw

        # shape (patch_height * patch_width, patch_height, patch_width)
        # -> (n_pix_p, ph, pw)
        self.register_buffer('zca_matrix_conv_shape',
                             torch.tensor(zca_matrix_conv_shape.reshape(-1, self.ph, self.pw),
                                          dtype=self.dtype))

    def forward(self, image_imshape: torch.Tensor) -> torch.Tensor:
        '''

        :param image_imshape: shape (batch, height, width)
        :return: shape (batch, )
        '''

        # shape (batch, 1, height, width) \ast (n_pix_p, 1, ph, pw)
        # -> (batch, n_pix_p, height - ph + 1, width - pw + 1)
        conv_patch_zca = F.conv2d(image_imshape[:, None, :, :], self.zca_matrix_conv_shape[:, None, :, :],
                                  stride=self.stride)

        # -> (batch, height - ph + 1, width - pw + 1)
        quad_form_per_patch = torch.sum(conv_patch_zca * conv_patch_zca, dim=1)

        # shape (batch, )
        regularization_term = torch.mean(quad_form_per_patch.reshape(quad_form_per_patch.shape[0], -1), dim=1)

        return regularization_term
