#!/usr/bin/env python
# coding: utf-8

import math
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import os

argvs = sys.argv  # コマンドライン引数を格納したリストの取得
argc = len(argvs)  # 引数の個数
if (argc != 4):  # 引数が足りない場合は、その旨を表示
    print('Usage: # python %s output base Nstack' % argvs[0])
    quit()  # プログラムの終了

output = argvs[1]

dir = "/Volumes/ALLSKY_Takahashi/code/ggl/kappa_around_halo/output"
base = argvs[2]


# base="r054_nres13_kappa_M200bmin3e14_z_0.10_0.30"

def read_data_and_gen_map(file, gsize, nside):
    df = pd.read_csv(file, delim_whitespace=True, header=None, skiprows=2)
    df.columns = ['x', 'y', 'kappa']

    header = pd.read_csv(file, delim_whitespace=True, header=None, nrows=1)
    # M200b = 3.490500e+14, M200c = 1.939500e+14, M500c = 1.140800e+14, Mvir = 2.939000e+14, z = 1.015752e-01, Rs = 6.211580e-01
    header.columns = ["h1",
                      "h2", "h3", "M200b",
                      "h4", "h5", "M200c",
                      "h6", "h7", "M500c",
                      "h8", "h9", "Mvir",
                      "h10", "h11", "z",
                      "h12", "h13", "rs"]

    # print(header)
    header['M200b'] = header['M200b'].astype(str)
    header['M200b'] = header['M200b'].str.replace(',', '')
    header['M200b'] = header['M200b'].astype(float)

    header['M200c'] = header['M200c'].astype(str)
    header['M200c'] = header['M200c'].str.replace(',', '')
    header['M200c'] = header['M200c'].astype(float)

    header['M500c'] = header['M500c'].astype(str)
    header['M500c'] = header['M500c'].str.replace(',', '')
    header['M500c'] = header['M500c'].astype(float)

    header['Mvir'] = header['Mvir'].astype(str)
    header['Mvir'] = header['Mvir'].str.replace(',', '')
    header['Mvir'] = header['Mvir'].astype(float)

    header['z'] = header['z'].astype(str)
    header['z'] = header['z'].str.replace(',', '')
    header['z'] = header['z'].astype(float)

    M200b = header['M200b'][0]
    M500c = header['M500c'][0]
    M200c = header['M200c'][0]
    Mvir = header['Mvir'][0]
    z = header['z'][0]
    rs = header['rs'][0]

    x = np.array(df['x'])
    y = np.array(df['y'])
    k = np.array(df['kappa'])

    map = np.zeros((nside, nside))
    norm = np.zeros((nside, nside))

    min = -0.5 * gsize * nside

    n = len(x)

    for i in range(n):
        x_index = int((x[i] - min) / gsize)
        y_index = int((y[i] - min) / gsize)

        # CIC
        dx = x[i] - (min + x_index * gsize)
        dy = y[i] - (min + y_index * gsize)

        w11 = (1.0 - dx / gsize) * (1.0 - dy / gsize)
        w22 = dx / gsize * dy / gsize
        w12 = dx / gsize * (1.0 - dy / gsize)
        w21 = (1.0 - dx / gsize) * dy / gsize

        if w11 < 0: w11 = 0
        if w22 < 0: w22 = 0
        if w12 < 0: w12 = 0
        if w21 < 0: w21 = 0

        if x_index >= 0 and x_index < nside:
            if y_index >= 0 and y_index < nside:
                map[y_index][x_index] += k[i] * w11
                norm[y_index][x_index] += w11

        if x_index + 1 >= 0 and x_index + 1 < nside:
            if y_index + 1 >= 0 and y_index + 1 < nside:
                map[y_index + 1][x_index + 1] += k[i] * w22
                norm[y_index + 1][x_index + 1] += w22

        if x_index + 1 >= 0 and x_index + 1 < nside:
            if y_index >= 0 and y_index < nside:
                map[y_index][x_index + 1] += k[i] * w12
                norm[y_index][x_index + 1] += w12

        if x_index >= 0 and x_index < nside:
            if y_index + 1 >= 0 and y_index + 1 < nside:
                map[y_index + 1][x_index] += k[i] * w21
                norm[y_index + 1][x_index] += w21

    for i in range(nside):
        for j in range(nside):
            if norm[i][j] > 0:
                map[i][j] = map[i][j] / norm[i][j]

    return M200b, M500c, M200c, Mvir, z, rs, map


Nstack = int(argvs[3])

nside = 40
gsize = 0.50  # arcmin

for i in range(Nstack):

    if i > 0 and np.mod(i + 1, 100) == 0: print("#", flush=True, end="")

    file = dir + '/' + base + '_' + str(i) + '.txt'
    M200b, M500c, M200c, Mvir, z, rs, map = read_data_and_gen_map(file, gsize, nside)

    fileout = output + '_' + str(i) + '.fits'
    hdu = fits.PrimaryHDU(map)
    hdu.header['Ngrid'] = nside
    hdu.header['pix_size'] = gsize
    hdu.header['M200b'] = M200b / 1e14
    hdu.header['M200c'] = M200c / 1e14
    hdu.header['M500c'] = M500c / 1e14
    hdu.header['Mvir'] = Mvir / 1e14
    hdu.header['Redshift'] = z
    hdu.header['R_scale'] = rs
    hdu.writeto(fileout, overwrite=True)

print("done!")
