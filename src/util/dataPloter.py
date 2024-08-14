import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.colors import from_levels_and_colors

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


class PiecewiseNorm(Normalize):
    def __init__(self, levels, clip=False):
        # the input levels
        self._levels = np.sort(levels)
        # corresponding normalized values between 0 and 1
        self._normed = np.linspace(0, 1, len(levels))
        Normalize.__init__(self, None, None, clip)

    def __call__(self, value, clip=None):
        # linearly interpolate to get the normalized value
        return np.ma.masked_array(np.interp(value, self._levels, self._normed))

def getCmapAndNorm(cmapName, levels, extend):
    cmap = plt.get_cmap(cmapName)
    if extend == "both":
        cmap, norm = from_levels_and_colors(levels, [cmap(i/len(levels)) for i in range(len(levels)+1)], extend=extend)
    elif extend == "max" or extend == "min":
        cmap, norm = from_levels_and_colors(levels, [cmap(i/len(levels)) for i in range(len(levels))], extend=extend)
    return cmap, norm
