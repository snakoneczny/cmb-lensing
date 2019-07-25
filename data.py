from os import listdir
from os.path import join
import resource

import numpy as np
import pandas as pd
import astropy.io.fits as fits
from tqdm import tqdm

from utils import safe_indexing


CMB_LENS_COLS = [
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
]


def read_train_data(folder, col_y='M500c', n_img=None, file_list=None):
    if file_list is None:
        file_list = listdir(folder)
    if n_img is None:
        n_img = 20000

    n_img = min(len(file_list), n_img)
    if n_img > len(file_list):
        file_list = file_list[:n_img]

    # MacOS requirements for number of open files
    resource.setrlimit(resource.RLIMIT_NOFILE, (n_img + 100, -1))

    X, y = [], []
    for i, file_name in enumerate(tqdm(file_list, desc='Reading data')):
        file_path = join(folder, file_name)
        img, img_info = read_cmb_lensed_img(file_path)

        # if img_info['M200b'] > 2.0:
        X.append(img)
        y.append(img_info[col_y])

    return np.array(X), np.array(y)


def read_cmb_lensed_folder(folder, n_img=None):
    file_list = listdir(folder)
    if n_img:
        # MacOS requirements for number of open files
        resource.setrlimit(resource.RLIMIT_NOFILE, (n_img + 1000, -1))
        file_list = file_list[:n_img]

    data = pd.DataFrame()
    for i, file_name in enumerate(tqdm(file_list, desc='Reading data')):
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
        info_dict = {key: header[key] for key in CMB_LENS_COLS}

    return data, info_dict


def read_tng_data():
    data = np.load('/users/snakoneczny/data/TNGclusterdat_augmented.npy')
    print(data.dtype.names)
    labels = np.log10(data['m500']).reshape(-1, 1)
    return data['data'], labels, data['id'] % 10  # data, labels, folds


# def read_fits_to_pandas(filepath, columns=None):
#     table = Table.read(filepath, format='fits')
#
#     # Limit table to useful columns and check if SDSS columns are present from cross-matching
#     if columns is None:
#         columns_errors = ['MAGERR_GAAP_{}'.format(band) for band in BANDS]
#         columns = COLUMNS_KIDS + BAND_COLUMNS + COLOR_NEXT_COLUMNS + columns_errors + FLAGS_GAAP_COLUMNS + [
#             'IMAFLAGS_ISO']
#         if COLUMNS_SDSS[0] in table.columns:
#             columns += COLUMNS_SDSS
#     # Get proper columns into a pandas data frame
#     table = table[columns].to_pandas()
#
#     # Binary string don't work with scikit metrics
#     if 'CLASS' in table:
#         table.loc[:, 'CLASS'] = table['CLASS'].apply(lambda x: x.decode('UTF-8').strip())
#     table.loc[:, 'ID'] = table['ID'].apply(lambda x: x.decode('UTF-8').strip())
#
#     # Change type to work with it as with a bit map
#     if 'IMAFLAGS_ISO' in table:
#         table.loc[:, 'IMAFLAGS_ISO'] = table['IMAFLAGS_ISO'].astype(int)
#
#     return table

def get_flat_mass_distribution(X, y, n_bins=100, max_bin_size=100):
    np.random.seed(15324)
    binned = pd.cut(y, n_bins)
    index_final = []
    for bin in binned.unique():
        index = np.where(binned == bin)[0]
        bin_size = len(index)
        if bin_size > max_bin_size:
            index = np.random.choice(index, max_bin_size)
        index_final.extend(index)
    return X[index_final], y[index_final]
