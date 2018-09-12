
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
import cv2 as cv
import matplotlib.pyplot as plt
from fatiando.seismic import wavefd

fname1 = r'data/vel_z6.25m_x12.5m_exact.segy'
fname2 = r'data/vp_marmousi-ii.segy'
# fname3 = r"H:\DATA\shenzi_raz_3d_t4_sb11_velocity_mod_depth Random line [2D Converted] 1.sgy"
with segyio.open(fname1) as f:
    BP_2004_vpModel = segyio.tools.cube(f)
    BP_2004_vpModel = np.squeeze(BP_2004_vpModel.T)
    # np.save('data\BP_2004_vpModel_6.25x12.5m_.npy',BP_2004_vpModel)

with segyio.open(fname2) as f:
    vp_marmousi = segyio.tools.cube(f)
    vp_marmousi = np.squeeze(vp_marmousi.T * 1000) # rescaling from km/s to m/s
    # np.save(r'data\vp_marmousi-ii_1.25x1.25m_.npy', vp_marmousi)
# with segyio.open(fname3) as f:
#     vp_shenzi = segyio.tools.cube(f)
#     vp_shenzi = np.squeeze(vp_shenzi.T * 0.3048) # conver from ft/s to m/s

# Fun with Repsol logos
def rescale(array, new_min, new_max):
    array_rscl = (array - array.min()) * (new_max - new_min) / (array.max() - array.min()) + new_min
    return array_rscl

# img = cv.imread(r"C:\Users\Thomas Cowan\Documents\GitHub\geotools\toys\velocities_from_images\Orange_Line_Logo_sq.png")
# img_gryscl = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# vel_repsol = np.where(img_gryscl == 255, 1500, 3000)
fname_logo = r"C:\Users\Thomas Cowan\Documents\GitHub\geotools\toys\velocities_from_images\REPSOL.png"
color_logo = cv.imread(fname_logo)
color_logo_rgb = cv.cvtColor(color_logo, cv.COLOR_BGR2RGB)
color_logo = np.sum(color_logo,axis=2)
vel_repsol_color = rescale(color_logo, 6500,1500)


print('The BP velocity model has shape: ' + str(BP_2004_vpModel.shape))
print('The Marmousi velocity model has the shape: ' + str(vp_marmousi.shape))


# Clipping the model to a sqaure. Seems to break with a rectangle but
# I've made rectangular models work before... not sure what is wrong.

BP_2004_vpModel_sq = BP_2004_vpModel[0:1910, 1910:3820]
vp_marmousi_sq = vp_marmousi[:2800, 6500:9300]
# vp_shenzi_sq = vp_shenzi[5:1270,225:1490]
# print('The Shenzi velocity model has the shape: ' + str(vp_shenzi_sq.shape))
# Downsampled the square models for faster computing
src_model = vp_marmousi_sq
dst = src_model

# pyrDown likes integers
rows, cols = src_model.shape
rows = int(rows)
cols = int(cols)

# Rows need to be divisible by 2(ish...) for this method of downsaplting to work
dst = cv.pyrDown(src_model, (rows/2, cols/2))
# dst = cv2.pyrDown(dst, (int(dst.shape[0])/2, int(dst.shape[1]/2)))

print('The original model is ' + str(src_model.shape))
print('The downsampled model is ' + str(dst.shape))


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# Initializing and plotting the finite-difference model

# Initialize a blank finite-difference grid with a spacing of your choosing
# shape = BP_2004_vpModel_sq.shape
shape = vel_repsol_color.shape
ds = 1.6667  # spacing in meters
area = [0, shape[0] * ds, 0, shape[1] * ds]

# Fill the velocity field
velocity = vel_repsol_color

# Instantiate the source
fc = 100. # The approximate frequency of the source
source = [wavefd.GaussSource((velocity.shape[0] / 2 -2) * ds,
         2 * ds, area, shape,  1000., fc)]
# source = [wavefd.MexHatSource(950 * ds, 2 * ds, area, shape, 400., fc)]
dt = wavefd.scalar_maxdt(area, shape, np.max(velocity))
duration = 1
maxit = int(duration / dt)

# Generate the stations and reciever location
num_stations = 100
spac = velocity.shape[0]/num_stations # station spacing
stations = [[i*spac*ds, 3 * ds] for i in range(1,num_stations)] # geophone coordinates (x,z)
seismogram_list = ['seis' + str(i) for i in range(1,num_stations)] # Supposed to be for labeling geophones

snapshots = 10  # number of iterations before the plot updates
simulation = wavefd.scalar(velocity, area, dt, maxit, source, stations, snapshots, padding = 500, taper = 0.0005)

# Making the animation
plot_spacing = 50

x_rec = [i[0] for i in stations] # for plotting geophones
y_rec = [i[1] for i in stations]
x_src = (velocity.shape[0] / 2 -2) * ds
y_src = 2 * ds

fig = plt.figure(figsize=(15, 10))
# plt.rc('text', usetex=True)
plt.subplots_adjust(right=0.98, left=0.11, hspace=0.0, top=0.93)
plt.subplot2grid((2, 8), (0, 0), colspan=5, rowspan=3)
plt.imshow(color_logo_rgb, extent=area, origin='lower')# cmap='plasma_r')
ticksx = plt.gca().get_xticks() / 1000
ticksy = plt.gca().get_yticks() / 1000
fig.gca().set_xticklabels(ticksx.astype(float))
fig.gca().set_yticklabels(ticksy.astype(float))
plt.xticks(rotation=45)
# plt.colorbar(shrink = 0.59,label = r'P-velocity $\frac{m}{s}$',
#             orientation = 'horizontal')
plt.title('2D P-wave simulation', size = 20)
wavefield = plt.imshow(np.zeros_like(velocity), extent=area,
                       cmap='gray_r', vmin=-100, vmax=100, alpha = 0.4)
plt.scatter(x_rec,y_rec, color = 'b', marker = 'v', s=30)
plt.scatter(x_src,y_src, color = 'r', marker = 'D', s=30)
plt.ylim(area[2:][::-1])
plt.xlabel('x (km)', size = 12)
plt.ylabel('z (km)', size = 12)
plt.subplot2grid((2, 8), (0, 5), colspan=3, rowspan=3)
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
plt.grid(linestyle = '--', alpha = 0.3)
for i in range(len(seismogram_list)):
    seismogram_list[i], = plt.plot([], [], '-k', linewidth=1)
    plt.plot(plot_spacing*i, 0, 'vb', markersize=5, linewidth=1)

plt.ylim(duration, 0)
plt.xlim(-135, 5025)
plt.ylabel('TWT (s)')
plt.tight_layout()
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
anim.save('repsol_color_logo.mp4', fps=30, dpi=200, bitrate=4000)
# anim # call the animation function
# plt.show()
