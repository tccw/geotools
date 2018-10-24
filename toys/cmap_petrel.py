import numpy as np
from matplotlib import cm

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
