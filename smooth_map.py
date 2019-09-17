from os import path
import math

import healpy as hp

from utils import DATA_DIR_RELATIVE

nres = 13
map_file_name = 'allskymap_nres{}r000_zs66'.format(nres)
smoothing_radius = 4  # arcmin
n_iterations = 3

# Read CMB convergence
map_path = path.join(DATA_DIR_RELATIVE, map_file_name + '.fits')
map = hp.read_map(map_path)

smoothing_radius_radians = math.radians(smoothing_radius / 60.0)
map_smooth = hp.sphtfunc.smoothing(map, fwhm=smoothing_radius_radians, iter=n_iterations, verbose=True)

# example of saving data as a fits file
map_file_name += '_fwhm-{}arcmin-{}iter'.format(smoothing_radius, n_iterations)
output_dir = path.join(DATA_DIR_RELATIVE, 'maps_smooth', map_file_name + '.fits')
hp.fitsfunc.write_map(output_dir, map_smooth, overwrite=True)
print('Smoothed map file written to {}'.format(output_dir))
