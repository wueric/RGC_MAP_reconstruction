import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from shapely.geometry import MultiPoint

from typing import Tuple

from data_util.matched_cells_struct import OrderedMatchedCellsStruct


def compute_convex_hull_boundary(original_mask_boolean: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''

    :param original_mask:
    :param shrinkage_factor:
    :return:
    '''

    xx, yy = np.meshgrid(np.r_[0:original_mask_boolean.shape[1]],
                         np.r_[0:original_mask_boolean.shape[0]])

    within_xx, within_yy = xx[original_mask_boolean], yy[original_mask_boolean]
    point_tuples = [(x, y) for x, y in zip(within_xx, within_yy)]

    mp_boundary = MultiPoint(point_tuples)
    x_coords, y_coords = mp_boundary.convex_hull.exterior.coords.xy

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    return x_coords, y_coords


def plot_rgc_mosaics(cell_order_struct: OrderedMatchedCellsStruct,
                     stafit_by_cell_id,
                     valid_mask: np.ndarray):

    x_boundary, y_boundary = compute_convex_hull_boundary(valid_mask)

    NUM_SIGMAS_RFFIT = 2
    height, width = 40, 80
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    DISPLAY_LOW_X, DISPLAY_HIGH_X = 10, 65
    DISPLAY_LOW_Y, DISPLAY_HIGH_Y = 0, 32

    SHADE_ALPHA = 0.2

    ax = axes[0][0]
    ax.axis('equal')
    ax.set_xlim([DISPLAY_LOW_X, DISPLAY_HIGH_X])
    ax.set_ylim([DISPLAY_LOW_Y, DISPLAY_HIGH_Y])
    ax.fill(8 + (x_boundary / 4.0), height - (y_boundary / 4.0), alpha=SHADE_ALPHA)
    ax.set_title('ON parasol', fontsize=18, fontstyle='italic')
    ax.axis('off')

    on_parasol_list = cell_order_struct.get_reference_cell_order('ON parasol')
    for cellid in on_parasol_list:
        sta_fit = stafit_by_cell_id[cellid]

        mu_x = sta_fit.center_x
        mu_y = sta_fit.center_y
        sigma_x = -sta_fit.std_y
        sigma_y = sta_fit.std_x

        # Convert the tilt to degrees, and flip to be consistent with Vision.
        degrees = sta_fit.rot * (180 / np.pi) * -1
        fit = Ellipse(xy=(mu_x, mu_y), width=NUM_SIGMAS_RFFIT * sigma_y,
                      height=NUM_SIGMAS_RFFIT * sigma_x,
                      angle=degrees)
        ax.add_artist(fit)

        fig.set_clip_box(ax)
        fit.set_alpha(1)
        fit.set_facecolor('none')
        fit.set_edgecolor('black')

    ax = axes[0][1]
    ax.axis('equal')
    ax.set_xlim([DISPLAY_LOW_X, DISPLAY_HIGH_X])
    ax.set_ylim([DISPLAY_LOW_Y, DISPLAY_HIGH_Y])
    ax.fill(8 + (x_boundary / 4.0), height - (y_boundary / 4.0), alpha=SHADE_ALPHA)
    ax.set_title('OFF parasol', fontsize=18, fontstyle='italic')
    ax.axis('off')

    off_parasol_list = cell_order_struct.get_reference_cell_order('OFF parasol')
    for cellid in off_parasol_list:
        sta_fit = stafit_by_cell_id[cellid]

        mu_x = sta_fit.center_x
        mu_y = sta_fit.center_y
        sigma_x = -sta_fit.std_y
        sigma_y = sta_fit.std_x

        # Convert the tilt to degrees, and flip to be consistent with Vision.
        degrees = sta_fit.rot * (180 / np.pi) * -1
        fit = Ellipse(xy=(mu_x, mu_y), width=NUM_SIGMAS_RFFIT * sigma_y,
                      height=NUM_SIGMAS_RFFIT * sigma_x,
                      angle=degrees)
        ax.add_artist(fit)
        fig.set_clip_box(ax)
        fit.set_alpha(1)
        fit.set_facecolor('none')
        fit.set_edgecolor('black')

    ax = axes[1][0]
    ax.axis('equal')
    ax.set_xlim([DISPLAY_LOW_X, DISPLAY_HIGH_X])
    ax.set_ylim([DISPLAY_LOW_Y, DISPLAY_HIGH_Y])
    ax.fill(8 + (x_boundary / 4.0), height - (y_boundary / 4.0), alpha=SHADE_ALPHA)
    ax.set_title('ON midget', fontsize=18, fontstyle='italic')
    ax.axis('off')

    on_midget_list = cell_order_struct.get_reference_cell_order('ON midget')
    for cellid in on_midget_list:
        sta_fit = stafit_by_cell_id[cellid]

        mu_x = sta_fit.center_x
        mu_y = sta_fit.center_y
        sigma_x = -sta_fit.std_y
        sigma_y = sta_fit.std_x

        # Convert the tilt to degrees, and flip to be consistent with Vision.
        degrees = sta_fit.rot * (180 / np.pi) * -1
        fit = Ellipse(xy=(mu_x, mu_y), width=NUM_SIGMAS_RFFIT * sigma_y,
                      height=NUM_SIGMAS_RFFIT * sigma_x,
                      angle=degrees)
        ax.add_artist(fit)
        fig.set_clip_box(ax)
        fit.set_alpha(1)
        fit.set_facecolor('none')
        fit.set_edgecolor('black')

    ax = axes[1][1]
    ax.axis('equal')
    ax.set_xlim([DISPLAY_LOW_X, DISPLAY_HIGH_X])
    ax.set_ylim([DISPLAY_LOW_Y, DISPLAY_HIGH_Y])
    ax.fill(8 + (x_boundary / 4.0), height - (y_boundary / 4.0), alpha=SHADE_ALPHA)
    ax.set_title('OFF midget', fontsize=18, fontstyle='italic')
    ax.axis('off')

    off_midget_list = cell_order_struct.get_reference_cell_order('OFF midget')
    for cellid in off_midget_list:
        sta_fit = stafit_by_cell_id[cellid]

        mu_x = sta_fit.center_x
        mu_y = sta_fit.center_y
        sigma_x = -sta_fit.std_y
        sigma_y = sta_fit.std_x

        # Convert the tilt to degrees, and flip to be consistent with Vision.
        degrees = sta_fit.rot * (180 / np.pi) * -1
        fit = Ellipse(xy=(mu_x, mu_y), width=NUM_SIGMAS_RFFIT * sigma_y,
                      height=NUM_SIGMAS_RFFIT * sigma_x,
                      angle=degrees)
        ax.add_artist(fit)
        fig.set_clip_box(ax)
        fit.set_alpha(1)
        fit.set_facecolor('none')
        fit.set_edgecolor('black')
    plt.tight_layout()
    plt.show()