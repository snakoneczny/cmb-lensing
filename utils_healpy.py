from functools import partial

import healpy as hp
import numpy as np
from tqdm import tqdm_notebook


# TODO: add pix size
def get_cut_imgs(halo_data, cmb_map, nres=13, cut_size=40, pixel_size=0.5):
    nside = 2 ** nres
    images = np.empty([halo_data.shape[0], cut_size, cut_size])
    for i, (index, row) in enumerate(tqdm_notebook(halo_data.iterrows())):
        lon, lat = hp.pixelfunc.pix2ang(nside, row['ipix'], lonlat=True)
        delta = cut_size / 2 * pixel_size / 60

        projector = hp.projector.CartesianProj(xsize=cut_size, ysize=cut_size, rot=(lon, lat, 0),
                                               lonra=[-delta, delta], latra=[-delta, delta])
        cmb_cut = projector.projmap(cmb_map, partial(hp.pixelfunc.vec2pix, nside))

        images[i, :, :] = cmb_cut

    return images
