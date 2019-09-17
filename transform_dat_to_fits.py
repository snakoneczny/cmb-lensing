from os import path

import numpy as np
import healpy as hp

from utils import DATA_DIR_RELATIVE

# input file
file_name_base = 'allskymap_nres13r000.zs6'
filename = path.join(DATA_DIR_RELATIVE, '{}.mag.dat'.format(file_name_base))

skip = [0, 536870908, 1073741818, 1610612728, 2147483638, 2684354547, 3221225457]
load_blocks = [skip[i + 1] - skip[i] for i in range(0, 6)]

with open(filename, 'rb') as f:
    rec = np.fromfile(f, dtype='uint32', count=1)[0]
    nside = np.fromfile(f, dtype='int32', count=1)[0]
    npix = np.fromfile(f, dtype='int64', count=1)[0]
    rec = np.fromfile(f, dtype='uint32', count=1)[0]
    print("nside:{} npix:{}".format(nside, npix))

    rec = np.fromfile(f, dtype='uint32', count=1)[0]

    print('Reading kappa..')
    kappa = np.array([])
    r = npix
    for i, l in enumerate(load_blocks):
        blocks = min(l, r)
        load = np.fromfile(f, dtype='float32', count=blocks)
        np.fromfile(f, dtype='uint32', count=2)
        kappa = np.append(kappa, load)
        r = r - blocks
        if r == 0:
            break
        elif r > 0 and i == len(load_blocks) - 1:
            load = np.fromfile(f, dtype='float32', count=r)
            np.fromfile(f, dtype='uint32', count=2)
            kappa = np.append(kappa, load)

    print('Reading gamma 1..')
    gamma1 = np.array([])
    r = npix
    for i, l in enumerate(load_blocks):
        blocks = min(l, r)
        load = np.fromfile(f, dtype='float32', count=blocks)
        np.fromfile(f, dtype='uint32', count=2)
        gamma1 = np.append(gamma1, load)
        r = r - blocks
        if r == 0:
            break
        elif r > 0 and i == len(load_blocks) - 1:
            load = np.fromfile(f, dtype='float32', count=r)
            np.fromfile(f, dtype='uint32', count=2)
            gamma1 = np.append(gamma1, load)

    print('Reading gamma 2..')
    gamma2 = np.array([])
    r = npix
    for i, l in enumerate(load_blocks):
        blocks = min(l, r)
        load = np.fromfile(f, dtype='float32', count=blocks)
        np.fromfile(f, dtype='uint32', count=2)
        gamma2 = np.append(gamma2, load)
        r = r - blocks
        if r == 0:
            break
        elif r > 0 and i == len(load_blocks) - 1:
            load = np.fromfile(f, dtype='float32', count=r)
            np.fromfile(f, dtype='uint32', count=2)
            gamma2 = np.append(gamma2, load)

    print('Reading omega..')
    omega = np.array([])
    r = npix
    for i, l in enumerate(load_blocks):
        blocks = min(l, r)
        load = np.fromfile(f, dtype='float32', count=blocks)
        np.fromfile(f, dtype='uint32', count=2)
        omega = np.append(omega, load)
        r = r - blocks
        if r == 0:
            break
        elif r > 0 and i == len(load_blocks) - 1:
            load = np.fromfile(f, dtype='float32', count=r)
            np.fromfile(f, dtype='uint32', count=2)
            omega = np.append(omega, load)

# print data
# for i in range(npix):
#     print(i, kappa[i], gamma1[i], gamma2[i], omega[i])

print('Writing fits file..')

# example of saving data as a fits file
file_name_base = file_name_base.replace('.', '_')
output_dir = path.join(DATA_DIR_RELATIVE, file_name_base + '.fits')
hp.fitsfunc.write_map(output_dir, kappa)
print('FITS file written to {}'.format(output_dir))
