import numpy as np
import matplotlib.pyplot as plt
import random


def get_cmap(n):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    cmap_fn = plt.cm.get_cmap('hsv', n+1)
    colors = [cmap_fn(i + 1)[:3] for i in range(n)]
    random.shuffle(colors)
    cmap = (np.array(colors) * 255.0).astype(np.uint8)
    return cmap