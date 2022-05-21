# taken from Github, yjkimnada/ns_decoding
# Original Author: Young Joon Kim


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
import torch.optim as torch_optim
import matplotlib.pyplot as plt


class ThinDatasetWrapper(torch_data.Dataset):

    def __init__(self, images: np.ndarray, spikes: np.ndarray):
        self.images = images
        self.spikes = spikes

    def __getitem__(self, index):
        return self.images[index, ...], self.spikes[index, ...]

    def __len__(self):
        return self.images.shape[0]


def plot_examples(batch_ground_truth: np.ndarray,
                  batch_decoder_output: np.ndarray):
    '''
    For every image in the example batch,
    :param forward_intermediates:
    :param linear_model:
    :param batched_observed_spikes:
    :return:
    '''

    batch_size = batch_ground_truth.shape[0]

    fig, axes = plt.subplots(batch_size, 2, figsize=(2 * 5, 5 * batch_size))
    for row in range(batch_size):
        ax = axes[row, 0]
        ax.imshow(batch_ground_truth[row, ...], vmin=-1.0, vmax=1.0, cmap='gray')
        ax.axis('off')

        ax = axes[row, 1]
        ax.imshow(batch_decoder_output[row, ...], vmin=-1.0, vmax=1.0, cmap='gray')
        ax.axis('off')

    return fig


def eval_test_loss_decoder(parallel_decoder: 'Parallel_NN_Decoder',
                           hpf_test_dataloader: torch_data.DataLoader,
                           loss_callable,
                           device: torch.device) -> float:

    loss_acc = []
    with torch.no_grad():
        for it, (hpf_np, spikes_np) in enumerate(hpf_test_dataloader):

            # shape (batch, height, width)
            hpf_torch= torch.tensor(hpf_np, dtype=torch.float32, device=device)
            batch, height, width = hpf_torch.shape

            # shape (batch, n_cells, n_timebins)
            spikes_torch = torch.tensor(spikes_np, dtype=torch.float32, device=device)

            output_flat = parallel_decoder(spikes_torch).reshape(batch, height, width)
            loss = loss_callable(output_flat, hpf_torch).detach().cpu().numpy()

            loss_acc.append(np.mean(loss))
    return np.mean(loss_acc)


def train_parallel_NN_decoder(parallel_decoder: 'Parallel_NN_Decoder',
                              hpf_dataloader: torch_data.DataLoader,
                              test_hpf_dataloader: torch_data.DataLoader,
                              loss_callable,
                              device: torch.device,
                              summary_writer,
                              learning_rate: float = 1e-1,
                              weight_decay: float = 1e-7,
                              momentum: float = 0.9,
                              n_epochs=16) -> 'Parallel_NN_Decoder':

    '''
    optimizer = torch_optim.SGD(parallel_decoder.parameters(),
                                momentum=momentum,
                                lr=learning_rate,
                                weight_decay=weight_decay)
                                '''
    optimizer = torch_optim.Adam(parallel_decoder.parameters(),
                                lr=1e-4,
                                weight_decay=weight_decay)
    n_steps_per_epoch = len(hpf_dataloader)
    for ep in range(n_epochs):

        for it, (images_np, spikes_np) in enumerate(hpf_dataloader):

            # shape (batch, height, width)
            hpf_torch= torch.tensor(images_np, dtype=torch.float32, device=device)
            batch, height, width = hpf_torch.shape

            # shape (batch, n_cells, n_timebins)
            spikes_torch = torch.tensor(spikes_np, dtype=torch.float32, device=device)

            optimizer.zero_grad()

            output_flat = parallel_decoder(spikes_torch).reshape(batch, height, width)
            loss = loss_callable(output_flat, hpf_torch)

            loss.backward()
            optimizer.step()

            # log stuff out to Tensorboard
            # loss is updated every step
            summary_writer.add_scalar('training loss', loss.item(), ep * n_steps_per_epoch + it)

            if it % 16 == 0:
                ex_fig = plot_examples(images_np, output_flat.detach().cpu().numpy())
                summary_writer.add_figure('training example images',
                                          ex_fig,
                                          global_step=ep*n_steps_per_epoch + it)
            del hpf_torch, spikes_torch, output_flat, loss

        test_loss = eval_test_loss_decoder(parallel_decoder,
                                           test_hpf_dataloader,
                                           loss_callable,
                                           device)

        # log stuff out to Tensorboard
        # loss is updated every step
        summary_writer.add_scalar('test loss ', test_loss, (ep + 1) * n_steps_per_epoch)

    return parallel_decoder


class Parallel_NN_Decoder(nn.Module):
    '''
    Massively parallel implementation of NN_Decoder by Eric Wu (wueric)
        using grouped 1D convolutions and fancy
        Pytorch indexing operations

    In our implementation, we only use known real cells
        so the terms "unit" and "cell" can be used
        interchangeably
    '''

    def __init__(self,
                 pix_cell_sel: np.ndarray,
                 cell_unit_count: int,
                 t_dim: int,
                 k_dim: int,
                 h_dim: int,
                 p_dim: int,
                 f_dim: int):
        '''

        :param pix_cell_sel: shape (p_dim, k_dim)
        :param cell_unit_count:
        :param t_dim: int, number of time bins
        :param k_dim: int, number of cells to select for each pixel
        :param h_dim: int, width of hidden layer
        :param p_dim: int, number of pixels
        :param f_dim: int, number of features per cell
        '''

        super().__init__()

        self.cell_unit_count = cell_unit_count
        self.t_dim = t_dim
        self.h_dim = h_dim
        self.p_dim = p_dim
        self.k_dim = k_dim
        self.f_dim = f_dim

        if pix_cell_sel.shape != (self.p_dim, self.k_dim):
            raise ValueError(f"pix_cell_sel must have shape {(self.p_dim, self.k_dim)}")

        # shape (p_dim, k_dim)
        self.register_buffer('pix_cell_sel', torch.tensor(pix_cell_sel, dtype=torch.long))

        # self.cell_unit_count parallel Linear layers,
        # with self.t_dim inputs, and self.f_dim outputs
        self.featurize = nn.Conv1d(self.cell_unit_count,
                                   self.f_dim * self.cell_unit_count,
                                   kernel_size=self.t_dim,
                                   groups=self.cell_unit_count,
                                   stride=1,
                                   padding=0,
                                   bias=True)

        # self.p_dim parallel Linear layers
        # with self.k_dim * self.f_dim inputs, and self.h_dim outputs
        self.hidden1 = nn.Conv1d(self.p_dim,
                                 self.h_dim * self.p_dim,
                                 kernel_size=self.k_dim * self.f_dim,
                                 groups=self.p_dim,
                                 stride=1,
                                 padding=0,
                                 bias=True)

        self.nl = nn.PReLU()

        # self.p_dim parallel Linear layers
        # with self.h_dim inputs, and 1 output
        self.output_layer = nn.Conv1d(self.p_dim,
                                      self.p_dim,
                                      kernel_size=self.h_dim,
                                      stride=1,
                                      padding=0,
                                      groups=self.p_dim,
                                      bias=True)

    def forward(self, time_binned_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param time_binned_spikes: shape (batch, n_cells, n_timebins)
            aka (batch, cell_unit_count, t_dim)
        :return:
        '''

        batch, n_cells, n_timebins = time_binned_spikes.shape
        if n_cells != self.cell_unit_count:
            raise ValueError(
                f'time_binned_spikes wrong number of cells, had {n_cells}, expected {self.cell_unit_count}')
        if n_timebins != self.t_dim:
            raise ValueError(f'time_binned_spikes wrong number of time bins, had {n_timebins}, expected {self.t_dim}')

        # shape (batch, n_cells, n_timebins)

        # shape (batch, n_cells, n_timebins) -> (batch, self.cell_unit_count * self.f_dim, 1)
        # -> (batch, self.cell_unit_count, self.f_dim)
        time_featurized_outputs_unshape = self.featurize(time_binned_spikes)
        #print('time_featurized_outputs_unshape', time_featurized_outputs_unshape.shape)
        time_featurized_outputs = time_featurized_outputs_unshape.reshape(batch, self.cell_unit_count, self.f_dim)
        #print('time_featurized_outputs', time_featurized_outputs.shape)

        # the input to the next layer, self.hidden1, must have shape (self.k_dim * self.f_dim)
        # in the conv1d "time-bin" dimension
        # the overall input should have shape (batch, self.p_dim, self.k_dim * self.f_dim)
        # We get this by collecting all of the features of the cell and concatenating them with gather

        # now we have to select across the cells, which is
        # (self.p_dim, self.k_dim) -> (batch, self.p_dim, self.k_dim, self.f_dim)
        selection_indices_repeated = self.pix_cell_sel[None, :, :, None].expand(batch, -1, -1, self.f_dim)

        # shape (batch, self.p_dim, self.cell_unit_count, self.f_dim)
        time_featurized_outputs_exp = time_featurized_outputs[:, None, :, :].expand(-1, self.p_dim, -1, -1)
        #print('selection_indices_repeated', selection_indices_repeated.shape)
        #print('time_featurized_outputs_exp', time_featurized_outputs_exp.shape)

        # -> (batch, self.p_dim, self.k_dim, self.f_dim)
        selected_cell_features = torch.gather(time_featurized_outputs_exp, 2, selection_indices_repeated)

        # -> (batch, self.p_dim, self.k_dim * self.f_dim)
        selected_cell_features_flat = selected_cell_features.reshape(batch, self.p_dim, -1)

        # -> (batch, self.p_dim * self.h_dim, 1) -> (batch, self.p_dim, self.h_dim)
        hidden1_applied = self.nl(self.hidden1(selected_cell_features_flat)).reshape(batch, self.p_dim, self.h_dim)

        # shape (batch, self.p_dim, 1) -> (batch, self.p_dim)
        output_layer_applied = self.output_layer(hidden1_applied).squeeze(2)

        return output_layer_applied


class NN_Decoder(nn.Module):
    '''
    Original implementation by Young Joon Kim, yjkimnada
    '''
    def __init__(self, unit_no, t_dim, k_dim, h_dim, p_dim, f_dim):
        super().__init__()
        self.unit_no = unit_no
        self.t_dim = t_dim
        self.h_dim = h_dim
        self.p_dim = p_dim
        self.k_dim = k_dim
        self.f_dim = f_dim

        self.featurize = nn.ModuleList([nn.Linear(self.t_dim,
                                                  self.f_dim) for i in range(self.unit_no)]).cuda()

        self.hidden1 = nn.ModuleList([nn.Linear(self.k_dim * self.f_dim,
                                                self.h_dim) for i in range(self.p_dim)]).cuda()
        self.hidden1_act = nn.ModuleList([nn.PReLU() for i in range(self.p_dim)]).cuda()

        self.output_layer = nn.ModuleList([nn.Linear(self.h_dim,
                                                     1) for i in range(self.p_dim)]).cuda()

    def forward(self, S, pix_units):

        F = torch.empty(S.shape[0], self.unit_no * self.f_dim).cuda()
        for n in range(self.unit_no):
            feat_n = self.featurize[n](S[:, n * self.t_dim: (n + 1) * self.t_dim])
            F[:, n * self.f_dim: (n + 1) * self.f_dim] = feat_n

        I = torch.empty(S.shape[0], self.p_dim).cuda()

        for x in range(self.p_dim):
            unit_ids = pix_units[x]
            feat_ids = torch.empty((self.k_dim * self.f_dim))
            for i in range(self.k_dim):
                feat_ids[i * self.f_dim: (i + 1) * self.f_dim] = torch.arange(self.f_dim) + unit_ids[i] * self.f_dim

            pix_feat = self.hidden1[x](F[:, feat_ids.long()])
            pix_feat = self.hidden1_act[x](pix_feat)

            out = self.output_layer[x](pix_feat)

            I[:, x] = out.reshape(-1)

        return I
