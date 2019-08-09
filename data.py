from os import listdir
from os.path import join

import numpy as np
import pandas as pd
import astropy.io.fits as fits
from tqdm import tqdm, tqdm_notebook

CMB_LENS_COLS = {
    'ID',
    'Ngrid',  # Number of pixels along x or y direction
    'pix_size',  # Angular size of each pixel in unit of arcmin
    # Information of cluster at the center of image
    'M200b',  # Spherical overdensity mass with respect to 200 times mean mass density in unit of 10^14 Msun/h
    'M200c',  # SO mass with respect to 200 times critical density in unit of 10^14 Msun/h
    'M500c',  # SO mass with respect to 200 times critical density in unit of 10^14 Msun/h
    'Mvir',  # SO mass with respect to virial overdensity in unit of 10^14 Msun/h
    'Redshift',
    # Scale radius inferred by fitting with The Navarro–Frenk–White (NFW) profile in unit of comoving Mpc/h
    'R_scale',
    'theta_i',
    'phi_i',
}


def read_train_data(folder, col_y='M500c', n_img=None, file_list_path=None, image_size=None):
    file_list = np.loadtxt(file_list_path, dtype=str) if file_list_path is not None else listdir(folder)
    # n_img = 20000 if n_img is None else n_img

    n_img = len(file_list) if n_img is None else min(n_img, len(file_list))
    if n_img < len(file_list):
        file_list = file_list[:n_img]

    # MacOS requirements for number of open files
    # resource.setrlimit(resource.RLIMIT_NOFILE, (n_img + 100, -1))

    X, y = [], []
    for i, file_name in enumerate(tqdm(file_list, desc='Reading data')):
        file_path = join(folder, file_name)
        img, img_info = read_cmb_lensed_img(file_path)

        if image_size:
            img_center = img.shape[0] / 2
            left = img_center - image_size / 2
            right = img_center + image_size / 2
            print(img)
            print(img.shape)
            img = img[left: right, left:right]
            print(img)
            print(img.shape)
            exit()

        X.append(img)
        y.append(img_info[col_y])

    return np.array(X), np.array(y)


def read_cmb_lensed_folder(folder, n_img=None):
    file_list = listdir(folder)

    n_img = len(file_list) if n_img is None else min(n_img, len(file_list))
    if n_img < len(file_list):
        file_list = file_list[:n_img]

    # MacOS requirements for number of open files
    # resource.setrlimit(resource.RLIMIT_NOFILE, (n_img + 1000, -1))

    data = pd.DataFrame()
    for i, file_name in enumerate(tqdm_notebook(file_list, desc='Reading data')):
        file_path = join(folder, file_name)
        img, img_info = read_cmb_lensed_img(file_path)
        img_info['image'] = img
        data = data.append(img_info, ignore_index=True)

    return data


def read_cmb_lensed_img(file):
    # Open an image file with a header
    with fits.open(file) as hdu_list:
        hdu = hdu_list[0]
        data = hdu.data
        header = hdu.header

        # Extract useful information from a header
        info_dict = {key: header[key] for key in CMB_LENS_COLS if key in header}

    return data, info_dict


def get_flat_mass_dist_idx(data, n_bins=100, max_bin_size=100):
    np.random.seed(15324)
    binned = pd.cut(data, n_bins)
    index_final = []
    for bin in binned.unique():
        index = np.where(binned == bin)[0]
        bin_size = len(index)
        if bin_size > max_bin_size:
            index = np.random.choice(index, max_bin_size)
        index_final.extend(index)
    return index_final


def get_flat_mass_dist(X, y, n_bins=100, max_bin_size=100):
    idx = get_flat_mass_dist_idx(y, n_bins=n_bins, max_bin_size=max_bin_size)
    return X[idx], y[idx]


def get_flat_mass_dist_for_df(data, col='M500c', n_bins=100, max_bin_size=100):
    idx = get_flat_mass_dist_idx(data[col], n_bins=n_bins, max_bin_size=max_bin_size)
    return data.iloc[idx]
