from functools import partial
from math import pi

import healpy as hp
import numpy as np
from tqdm import tqdm_notebook


# def get_cut_imgs(halo_data, cmb_map=None, cmb_map_projected=None, nres=13, cut_size=40):
#     nside = 2 ** nres
#     # cmb_res = hp.pixelfunc.nside2resol(nside, arcmin=True)
#     # n_pix_x = int(360 * 60 / cmb_res)  # factor of 3.2 moves cut size from 40 to 128
#     x_size = 4000
#
#     projector = hp.projector.CartesianProj(xsize=40, ysize=40, lonra=[], latra=[])
#
#     if cmb_map_projected is None:
#         cmb_map_projected = projector.projmap(cmb_map, partial(hp.pixelfunc.vec2pix, nside))
#
#     i_max, j_max = cmb_map_projected.shape
#     images = np.empty([halo_data.shape[0], cut_size, cut_size])
#     for index, row in tqdm_notebook(halo_data.iterrows()):
#
#         # Get projection coordinates
#         theta, phi = row['theta_i'], row['phi_i']
#         x, y = projector.ang2xy(theta, phi)
#         i, j = projector.xy2ij(x, y)
#
#         # Continue if too close to borders to cut a full square
#         if i < cut_size / 2 or j < cut_size / 2 or i > i_max - cut_size / 2 or j > j_max - cut_size / 2:
#             continue
#
#         cut_i_min = int(i - cut_size / 2)
#         cut_i_max = int(i + cut_size / 2)
#         cut_j_min = int(j - cut_size / 2)
#         cut_j_max = int(j + cut_size / 2)
#         cmb_cut = cmb_map_projected[cut_i_min: cut_i_max, cut_j_min: cut_j_max]
#
#         images[index, :, :] = cmb_cut
#
#     return images


# TODO: add pix size
def get_cut_imgs(halo_data, cmb_map, nres=13, cut_size=40):
    nside = 2 ** nres
    # cmb_res = hp.pixelfunc.nside2resol(nside, arcmin=True)
    # n_pix_x = int(360 * 60 / cmb_res)  # factor of 3.2 moves cut size from 40 to 128

    # i_max, j_max = cmb_map_projected.shape
    images = np.empty([halo_data.shape[0], cut_size, cut_size])
    for index, row in tqdm_notebook(halo_data.iterrows()):

        theta, phi = row['theta_i'], row['phi_i']
        # print('theta {}, phi {}'.format(theta, phi))
        lat, lon = -np.degrees(theta - pi / 2), -np.degrees(phi - pi)
        # print('theta {}, phi {}'.format(theta, phi))
        # TODO: pix_size = 0.5  # pixel size in arcminc
        delta = 10 / 60  # 20 arcmins in degrees
        # lonra = [longitude - delta, longitude + delta]
        # print('lonra {}'.format(lonra))
        # latra = [latitude - delta, latitude + delta]
        # print('latra {}'.format(latra))

        # rot (lon lat psi)
        projector = hp.projector.CartesianProj(xsize=cut_size, ysize=cut_size, rot=(lon, lat, 0),
                                               lonra=[-delta, delta], latra=[-delta, delta])
        cmb_cut = projector.projmap(cmb_map, partial(hp.pixelfunc.vec2pix, nside))

        # Continue if too close to borders to cut a full square
        # if i < cut_size / 2 or j < cut_size / 2 or i > i_max - cut_size / 2 or j > j_max - cut_size / 2:
        #     continue

        # cut_i_min = int(i - cut_size / 2)
        # cut_i_max = int(i + cut_size / 2)
        # cut_j_min = int(j - cut_size / 2)
        # cut_j_max = int(j + cut_size / 2)
        # cmb_cut = cmb_map_projected[cut_i_min: cut_i_max, cut_j_min: cut_j_max]

        images[index, :, :] = cmb_cut

    return images
