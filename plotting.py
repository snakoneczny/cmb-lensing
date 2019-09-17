import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import healpy as hp


def plot_map(hpxmap, unit='unit'):
    mpl.rcParams['image.cmap'] = 'jet'
    hp.mollzoom(hpxmap, unit=unit, title='title')
    hp.graticule()


def plot_cmb_imgs(data, img_col='image', n_img=40, vmin=None, vmax=None):
    # Make plot adjusted for a grid of square images
    width = 20
    n_cols = 4
    fig = plt.figure(figsize=(width, n_img / n_cols / n_cols * width))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # Read and plot images in a grid
    for i in range(n_img):
        # Plot image in a proper grid cell
        ax = fig.add_subplot(n_img / n_cols, n_cols, i + 1)

        im = ax.imshow(data.iloc[i][img_col], cmap='jet')  # extent
        fig.colorbar(im, ax=ax)
        #     ax.set_xlabel('(arcmin)')
        #     ax.set_ylabel('(arcmin)')

        # Add title with halo information
        title = 'ID: {}'.format(data.iloc[i]['ID'])
        if 'M200b' in data.columns:
            title += ', M500c: {:.2f}, z: {:.2f}'.format(data.iloc[i]['M500c'], data.iloc[i]['z_halo'])
        ax.set_title(title)


# TODO: delete
def plot_external_cmb_imgs(imgs_df):
    # Make plot adjusted for a grid of square images
    n_img = imgs_df.shape[0]
    width = 20
    n_cols = 4
    fig = plt.figure(figsize=(width, n_img / n_cols / n_cols * width))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    mpl.rcParams['image.cmap'] = 'jet'

    # Read and plot images in a grid
    for i, (_, row) in enumerate(imgs_df.iterrows()):

        if i == n_img:
            break

        # Calculate image extent
        gsize = row['pix_size']
        nside = row['Ngrid']
        min = -0.5 * gsize * nside
        max = +0.5 * gsize * nside
        extent = (min, max, min, max)

        # Plot image in a proper grid cell
        ax = fig.add_subplot(n_img / n_cols, n_cols, i + 1)
        ax.imshow(row['image'], extent=extent)
        #     ax.set_xlabel('(arcmin)')
        #     ax.set_ylabel('(arcmin)')

        # Add title with halo information
        title = 'M200b: {:.4f}, M500c: {:.4f}, z: {:.4f}'.format(row['M200b'], row['M500c'], row['Redshift'])
        ax.set_title(title)
