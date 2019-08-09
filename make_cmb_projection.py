from os import path
from functools import partial

import numpy as np
import healpy as hp

from utils import DATA_DIR_RELATIVE

data_dir = DATA_DIR_RELATIVE

nres = 13
cmb_file = path.join(data_dir, 'lensed_cmbmap_betazero_nres{}r000.fits'.format(nres))
print('Reading CMB map...')
cmb_map = hp.read_map(cmb_file)

nside = 2 ** nres
cmb_res = hp.pixelfunc.nside2resol(nside, arcmin=True)
# n_pix_x = int(360 * 60 / cmb_res)  # factor of 3.2 moves cut size from 40 to 128
n_pix_x = 28378

projector = hp.projector.CartesianProj(xsize=n_pix_x)
print('Projecting CMB map...')
cmb_map_projected = projector.projmap(cmb_map, partial(hp.pixelfunc.vec2pix, nside))

projection_path = path.join(data_dir, 'lensed_cmbmap_betazero_nres{}r000_cart-28378'.format(nres))
print('Saving projected CMB map...')
np.save(projection_path, cmb_map_projected)

print('Success!')
