
"""
Uses the seismic package from Fatiando to create 2D finite difference
models for wave propogation in different velocity fields.

The BP velocity model is 67km long and 12km deep, and was built on a 6.25m x 12.5m grid.
In order for Fatiando to work the cells have to be square so I ignore this.
http://software.seg.org/datasets/2D/2004_BP_Vel_Benchmark/eage_abstract.pdf

The Marmousi2 Vp model is 3.5 km in depth and 17 km across on a 1.25m x 1.25m grid.
http://www.ahay.org/RSF/book/data/marmousi2/paper.pdf

Ignoring the density field for the BP model, and the Marmousi2 density field is
constant so it is also ignored.
"""

import segyio
import numpy as np
from matplotlib import animation
import cv2
import matplotlib.pyplot as plt
from fatiando.seismic import wavefd
from fatiando.vis import mpl


fname1 = r'data/vel_z6.25m_x12.5m_exact.segy'
fname2 = r'data/vp_marmousi-ii.segy'
with segyio.open(fname1) as f:
    BP_2004_vpModel = segyio.tools.cube(f)
    BP_2004_vpModel = np.squeeze(BP_2004_vpModel.T)
    # np.save('data\BP_2004_vpModel_6.25x12.5m_.npy',BP_2004_vpModel)

with segyio.open(fname2) as f:
    vp_marmousi = segyio.tools.cube(f)
    vp_marmousi = np.squeeze(vp_marmousi.T * 1000) # rescaling from km/s to m/s
    # np.save(r'data\vp_marmousi-ii_1.25x1.25m_.npy', vp_marmousi)

print('The BP velocity model has shape: ' + str(BP_2004_vpModel.shape))
print('The Marmousi velocity model has the shape: ' + str(vp_marmousi.shape))

# Clipping the model to a sqaure. Seems to break with a rectangle but
# I've made rectangular models work before... not sure what is wrong.

BP_2004_vpModel_sq = BP_2004_vpModel[0:1910, 0:(0+1910)]
vp_marmousi_sq = vp_marmousi[:2800, 6500:9300]

# Downsample the square models for faster computing
src_model = vp_marmousi_sq
dst = src_model

# pyrDown likes integers
rows, cols = src_model.shape
rows = int(rows)
cols = int(cols)

# Rows need to be divisible by 2(ish...) for this method of downsampling to work
dst = cv2.pyrDown(src_model, (rows/2, cols/2))
# dst = cv2.pyrDown(dst, (int(dst.shape[0])/2, int(dst.shape[1]/2)))

print('The original model is ' + str(src_model.shape))
print('The downsampled model is ' + str(dst.shape))


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# Initializing and plotting the finite-difference model

# Initialize a blank finite-difference grid with a spacing of your choosing
# shape = BP_2004_vpModel_sq.shape
shape = dst.shape
ds = 2.5  # spacing
area = [0, shape[0] * ds, 0, shape[1] * ds]

# Fill the velocity field
# velocity = BP_2004_vpModel_sq
velocity = dst

# Instantiate the source
fc = 50. # The approximate frequency of the source
source = [wavefd.GaussSource(700 * ds, 2 * ds, area, shape,  800., fc)]
# source = [wavefd.MexHatSource(950 * ds, 2 * ds, area, shape, 400., fc)]
dt = wavefd.scalar_maxdt(area, shape, np.max(velocity))
duration = 3
maxit = int(duration / dt)

# Generate the stations and reciever location
num_stations = 20
spac = velocity.shape[0]/num_stations # station spacing
stations = [[i*spac*ds, 3 * ds] for i in range(1,num_stations)] # geophone coordinates (x,z)
seismogram_list = ['seis' + str(i) for i in range(1,num_stations)] # Supposed to be for labeling geophones

snapshots = 40  # number of iterations before the plot updates
simulation = wavefd.scalar(velocity, area, dt, maxit, source, stations, snapshots)

# Making the animation
plot_spacing = 250

fig = mpl.figure(figsize=(10, 8))
mpl.subplots_adjust(right=0.98, left=0.11, hspace=0.0, top=0.93)
mpl.subplot2grid((2, 8), (0, 0), colspan=5, rowspan=3)
mpl.imshow(velocity, extent=area, origin='lower',cmap='plasma_r')
mpl.colorbar(shrink = 0.59)
wavefield = mpl.imshow(np.zeros_like(velocity), extent=area,
                       cmap='gray_r', vmin=-400, vmax=400, alpha = 0.6)
mpl.points(stations, '^b', size=8)
mpl.ylim(area[2:][::-1])
mpl.xlabel('x (km)')
mpl.ylabel('z (km)')
mpl.m2km()
mpl.subplot2grid((2, 8), (0, 5), colspan=3, rowspan=3)
mpl.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)

for i in range(len(seismogram_list)):
    seismogram_list[i], = mpl.plot([], [], '-k')
    mpl.plot(plot_spacing*i, 0, 'vb', markersize=20)

mpl.ylim(duration, 0)
mpl.xlim(-800, 5500)
#mpl.xlabel('Amplitude')
mpl.ylabel('TWT (s)')
mpl.tight_layout()
times = np.linspace(0, dt * maxit, maxit)

# This function updates the plot every few timesteps
def animate(i):
    t, u, seismogram = simulation.next()
    for j in range(len(seismogram_list)):
        seismogram_list[j].set_data(seismogram[j][:t + 1] + plot_spacing*j, times[:t + 1])
    wavefield.set_data(u[::-1])
    return wavefield, seismogram_list


anim = animation.FuncAnimation(
    fig, animate, frames=maxit / snapshots, interval=1)
# anim.save('p_wave_multi_geoph_Marmousi2_2.5m_.mp4', fps=30, dpi=200, bitrate=4000)
anim # call the animation function
mpl.show()
