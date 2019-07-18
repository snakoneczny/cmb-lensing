from os import listdir
from os.path import join

import numpy as np
import astropy.io.fits as fits
from tqdm import tqdm


def read_train_data(folder, col_y='M500c', n_rows=None):
    X, y = [], []
    file_list = listdir(folder)

    if n_rows:
        file_list = file_list[:n_rows]

    for i, file_name in enumerate(tqdm(file_list, desc='Reading data')):
        file_path = join(folder, file_name)
        img, img_info = read_cmb_lensed_img(file_path)
        X.append(img)
        y.append(img_info[col_y])

    return np.array(X), np.array(y)


def read_cmb_lensed_img(file):
    # Open an image file with a header
    with fits.open(file) as hdu_list:
        hdu = hdu_list[0]
        data = hdu.data
        header = hdu.header

        # Extract useful information from a header
        to_extract = [
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
        info_dict = {key: header[key] for key in to_extract}

    return data, info_dict


def read_tng_data():
    data = np.load('/users/snakoneczny/data/TNGclusterdat_augmented.npy')
    print(data.dtype.names)
    labels = np.log10(data['m500']).reshape(-1, 1)
    return data['data'], labels, data['id'] % 10

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
