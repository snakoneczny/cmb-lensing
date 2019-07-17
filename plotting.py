import matplotlib.pyplot as plt
import healpy as hp


def plot_map(hpxmap, unit='unit', is_cmap=True):
    cmap = plt.get_cmap('hot_r') if is_cmap else None
    hp.mollzoom(hpxmap, cmap=cmap, nest=False, xsize=1600, unit=unit, title='title')
    hp.graticule()
