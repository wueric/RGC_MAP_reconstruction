import torch
import torch.nn as nn
import torch.utils.data as torch_data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import matplotlib.pyplot as plt

from typing import Callable, Tuple


class Neurips2017_Encoder(nn.Module):
    '''
    Benchmark for the HQS + CNN prior reconstruction method, taken from
    Parathasarathy, Batty, et al, Neural Networks for Efficient Bayesian Decoding of
    Natural Images from Retinal Neurons, Neurips 2017

    (the autoencoder natural image prior paper)

    The code below should be a literal Pytorch port of the original published work.

    Implementation notes, taken from their paper:
    (1) Images are supposed to be on the range [-0.5, 0.5) ; for our implementation,
        for easy comparison with our code, we use the range [-1, 1) which should
        be equivalent

    '''

    def __init__(self,
                 dropout_rate: float):
        super().__init__()

        self.encoder_network = nn.Sequential(

            # (160 x 256)
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            # (80 x 128)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # (40 x 64)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # (20 x 32)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, batched_input_image: torch.Tensor):
        '''

        :param batched_input_image: shape (batch, height, width)
        :return: shape (batch, 256, output_height, output_width)
        '''
        return self.encoder_network(batched_input_image[:, None, :, :])


class Neurips2017_Decoder(nn.Module):

    def __init__(self,
                 drop_rate: float):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(num_features=1),
        )

    def forward(self, batched_encoder_image: torch.Tensor):
        '''

        :param batched_input_image: shape (batch, 256, output_height, output_width)
        :return: shape (batch, height, width)
        '''
        return self.decoder(batched_encoder_image).squeeze(1)


class ThinDatasetWrapper(torch_data.Dataset):

    def __init__(self, images: np.ndarray, spikes: np.ndarray):
        self.images = images
        self.spikes = spikes

    def __getitem__(self, index):
        return self.images[index, ...], self.spikes[index, ...]

    def __len__(self):
        return self.images.shape[0]


def plot_examples(batch_ground_truth: np.ndarray,
                  batch_linear_reconstructions: np.ndarray,
                  batch_autoencoder_output: np.ndarray,
                  mask: np.ndarray):
    '''
    For every image in the example batch,
    :param forward_intermediates:
    :param linear_model:
    :param batched_observed_spikes:
    :return:
    '''

    batch_size = batch_ground_truth.shape[0]

    fig, axes = plt.subplots(batch_size, 3, figsize=(3 * 5, 5 * batch_size))
    for row in range(batch_size):
        ax = axes[row, 0]
        ax.imshow(batch_ground_truth[row, ...] * mask, vmin=-1.0, vmax=1.0, cmap='gray')
        ax.axis('off')

        ax = axes[row, 1]
        ax.imshow(batch_linear_reconstructions[row, ...] * mask, vmin=-1.0, vmax=1.0, cmap='gray')
        ax.axis('off')

        ax = axes[row, 2]
        ax.imshow(batch_autoencoder_output[row, ...] * mask, vmin=-1.0, vmax=1.0, cmap='gray')
        ax.axis('off')

    return fig


def eval_test_loss_autoencoder(test_dataset: ThinDatasetWrapper,
                               linear_reconstructor,
                               encoder_model: Neurips2017_Encoder,
                               decoder_model: Neurips2017_Decoder,
                               loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                               device: torch.device,
                               calculation_batch_size: int = 32) -> float:
    output_losses = np.zeros((len(test_dataset),), dtype=np.float32)

    dataloader = torch_data.DataLoader(test_dataset,
                                       batch_size=calculation_batch_size,
                                       shuffle=False,
                                       drop_last=True)

    loss_acc = []
    with torch.no_grad():
        for it, (images_np, spikes_np) in enumerate(dataloader):

            image_torch = torch.tensor(images_np, dtype=torch.float32, device=device)
            spikes_torch = torch.tensor(spikes_np, dtype=torch.float32, device=device)

            reconstructed_image_torch = linear_reconstructor(spikes_torch)

            encoded_images = encoder_model(reconstructed_image_torch)
            decoded_images = decoder_model(encoded_images)

            batched_loss = loss_callable(image_torch, decoded_images).detach().cpu().numpy()
            loss_acc.append(np.mean(batched_loss))

    return np.mean(loss_acc)


def train_autoencoder(dataset: ThinDatasetWrapper,
                      test_dataset: ThinDatasetWrapper,
                      linear_reconstructor,
                      encoder_model: Neurips2017_Encoder,
                      decoder_model: Neurips2017_Decoder,
                      loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                      n_epochs: int,
                      batch_size: int,
                      learning_rate: float,
                      device: torch.device,
                      summary_writer: SummaryWriter,
                      mask: np.ndarray,
                      generate_images_nsteps: int = 5) \
        -> Tuple[Neurips2017_Encoder, Neurips2017_Decoder]:
    dataloader = torch_data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=False)

    optimizer = optim.Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()),
                           lr=learning_rate)

    n_steps_per_epoch = len(dataloader)

    for epoch in range(n_epochs):

        for it, (images_np, spikes_np) in enumerate(dataloader):
            image_torch = torch.tensor(images_np, dtype=torch.float32, device=device)
            spikes_torch = torch.tensor(spikes_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                reconstructed_image_torch = linear_reconstructor(spikes_torch)

            optimizer.zero_grad()

            encoded_images = encoder_model(reconstructed_image_torch)
            decoded_images = decoder_model(encoded_images)

            loss = torch.mean(loss_callable(image_torch, decoded_images))
            loss.backward()

            optimizer.step()

            # log stuff out to Tensorboard
            # loss is updated every step
            summary_writer.add_scalar('training loss', loss.item(), epoch * n_steps_per_epoch + it)

            # images are produced every generate_images_nsteps steps
            if it % generate_images_nsteps == 0:
                summary_writer.add_figure('training example images',
                                          plot_examples(images_np.detach().cpu().numpy(),
                                                        reconstructed_image_torch.detach().cpu().numpy(),
                                                        decoded_images.detach().cpu().numpy(),
                                                        mask),
                                          global_step=epoch * n_steps_per_epoch + it)

            del image_torch, spikes_torch, reconstructed_image_torch
            del encoded_images, decoded_images

        test_loss = eval_test_loss_autoencoder(test_dataset,
                                               linear_reconstructor,
                                               encoder_model,
                                               decoder_model,
                                               loss_callable,
                                               device)

        # log stuff out to Tensorboard
        # loss is updated every step
        summary_writer.add_scalar('test loss ', test_loss, (epoch + 1) * n_steps_per_epoch)

    return encoder_model, decoder_model
