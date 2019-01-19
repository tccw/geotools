import numpy as np
from matplotlib import cm

matplotlib_list = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys',
 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd',
 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink','spring', 'summer',
 'autumn', 'winter', 'cool', 'Wistia','hot', 'afmhot', 'gist_heat', 'copper',
 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu','RdYlBu', 'RdYlGn', 'Spectral',
 'coolwarm', 'bwr', 'seismic','hsv','Pastel1','Pastel2', 'Paired', 'Accent',
 'Dark2', 'Set1', 'Set2', 'Set3','tab10', 'tab20','tab20b', 'tab20c','flag',
 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern','gnuplot', 'gnuplot2',
 'CMRmap', 'cubehelix', 'brg','gist_rainbow', 'rainbow', 'jet', 'nipy_spectral',
 'gist_ncar']

def cmap_petrel(cmap_names):
    '''
    Takes matplotlib color maps and converts them to csv files which can be
    read by Petrel as "Color tables (Simple RGB color table)" format.
    '''
    for c in cmap_names:
        header = f'InterPolate false \nUseTrimming false \nZeroCentric false \nName {c} \nseparator ,'
        cmap_new = np.array([list(getattr(cm, c)(i)[:3]) for i in range(256)])
        cmap_new = np.round(cmap_new * 255, 0)
        np.savetxt(f'{c}.txt', cmap_new, delimiter=',', newline='\n',
                    fmt='%i', header=header,comments='')
