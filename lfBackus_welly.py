
import numpy as np
import bruges as b
from welly import Well
from tkinter import filedialog
import matplotlib.pyplot as plt

lb = np.array([1, 3, 7, 10, 40, 80])
freqs = np.array([5, 10, 20, 30, 40, 50, 65])

def lfBackus(lb,freqs, test = False, log_plot = True, dt = 0.002, f = 35):
    """
    Liner & Fei Backus thickness determination via the 'Backus Number.'

    This function uses the work of Liner & Fei to calculate the Backus average
    of the input curves at varying layer thicknesses. It then plots the orginal
    curves and the average curves. A second plot is used to illustrate the maximum
    bed thickness which will maintain all primaries and scattering reflection
    information for selected frequencies ($B$ <1/3) as well as maximum bed thickness
    which will maintain the direct arrival only ($B$ <2) and is suitable for
    migration velocity analysis, etc.

    B = (L'*f)/Vs min

    Variables:
        B = Backus number
        L' = Backus layer thickness
        f = frequency
        Vs min = The minimum shear velocity after backus averaging

    References:

    [https://library.seg.org/doi/abs/10.1190/1.2723204]

    The Backus number
    Liner,Chris et al.
    The Leading Edge(2007),26(4):420
    http://dx.doi.org/10.1190/1.2723204

    """

    if test == False:
        lasPath = filedialog.askopenfilename()
        lasFile = Well.from_las(lasPath)

        print(lasFile.header)
        print(lasFile.data)

        # Check what the names of the curves you are looking for are
        print('What is your compressional sonic called?: ')
        dtc = lasFile.data[str(input())]
        print('What is your shear sonic called?: ')
        dts = lasFile.data[str(input())]
        print('What is your density curve called?: ')
        rhob = lasFile.data[str(input())]

        # Get the z-step
        depth = lasFile.data['RHOB'].basis
        steps = np.diff(depth)

        if len(np.unique(steps)==1):
            dz = steps[0]
        else:
            dz = np.mean(steps)
            print('Z step was not constant.')
    elif test == True:
        fname = r"C:\Users\goril\Dropbox\PythonScratch\functions\100033305723W500.las"
        lasFile=Well.from_las(fname)

        dtc = lasFile.data['DTC']
        dts = lasFile.data['DTS']
        rhob = lasFile.data['RHOB']
        depth = lasFile.data['RHOB'].basis
        steps = np.diff(depth)
        dz = steps[0]
        #dz = np.diff(lasFile.data['DTC'].basis[1])

    # Handle any negative values that weren't caught on import
    dtc = np.where(dtc < 0, np.nan, dtc)
    dts = np.where(dts < 0, np.nan, dts)
    rhob = np.where(rhob < 0, np.nan, rhob)

    # round dz (this was for when I thought there was an issue with precision)
    dz = np.round(dz,4)
    print(f'\nThe z step was found to be: {dz} \n')

    vs = 1e6 / (3.23084 * dts)
    vp = 1e6 / (3.23084 * dtc)

    # Convert to time and generate synthetics
    bakus = np.array([b.rockphysics.backus(vp,vs,rhob,i,dz) for i in lb])

    time_curves = [] # should be able to do with all with LC but couldn't get it to work correctly
    for i in range(len(bakus)):
        time_curves.append([b.transform.depth_to_time(bakus[i,j,:-1], vp[:-1], dz, dt, return_t=True) for j in range(bakus.shape[1])])

    time_curves = np.array(time_curves)
    twt = time_curves[0,0,1]
    rc = np.array([b.reflection.acoustic_reflectivity(time_curves[i,0,0], time_curves[i,2,0]) for i in range(len(lb))])

    wavelet = b.filters.ricker(0.128, dt, f)
    synth = np.apply_along_axis(lambda r: np.convolve(r, wavelet, mode='same'), axis=1, arr=rc)

    vsMin = [np.nanmin(bakus[i][1]) for i in range(len(lb))]

    # Plot everything up
    if log_plot == True:
        plt.figure(figsize=(15,10))
        for i in np.arange(len(lb)):
            plt.subplot(1, len(lb), i+1)
            plt.plot(vp,depth,'k',alpha=0.25)
            plt.plot(vs,depth,'k',alpha=0.25)
            plt.plot(bakus[i][0], depth,'b',alpha=0.75, label='Vp')
            plt.plot(bakus[i][1],depth,'g',alpha=0.75, label='Vs')
            plt.gca().invert_yaxis()
            plt.title( '%d m Backus layer' % lb[i] )
            plt.grid()
            plt.xlim(np.nanmin(vs) - 100, np.nanmax(vp) + 100)
            plt.legend()
        plt.tight_layout()

        f, axarr = plt.subplots(nrows=1, ncols=2)
        axarr[1].set_ylim(0,3)
        axarr[1].set_xlim(0,np.max(lb))
        for i in np.arange(len(freqs)):
            axarr[0].plot(lb,vsMin,'o',lb,vsMin,'g--')
            axarr[0].set_title('$L$\'(m) vs Vs $min$')
            axarr[0].set_xlabel('$L$\' (backus length)',fontsize=10)
            axarr[0].set_ylabel('Vs $min$')
            axarr[1].plot(lb,(np.ones(len(lb))/3),'r--')
            axarr[1].plot(lb,(np.ones(len(lb))*2),'b--')
            axarr[1].set_title('Frequency ($Hz$) vs $L$\'')
            axarr[1].set_xlabel('$L$\' (backus length)')
            axarr[1].set_ylabel('$L$\' Backus Number')
            axarr[1].plot(lb,(freqs[i]*lb)/vsMin,label='%s Hz' % freqs[i])
            axarr[1].set_xlim(0, np.max(lb))
        plt.tight_layout()
        axarr[1].legend(loc='upper left',fontsize='large')

    elif log_plot == False:
        f, axarr = plt.subplots(1,2)
        axarr[1].set_ylim(0,3)
        axarr[1].set_xlim(0,np.max(lb))
        for i in np.arange(len(freqs)):
            axarr[0].plot(lb,vsMin,'o',lb,vsMin,'g--')
            axarr[0].set_title('$L$\'(m) vs Vs $min$')
            axarr[0].set_xlabel('$L$\' (backus length)',fontsize=10)
            axarr[0].set_ylabel('Vs $min$')
            axarr[1].plot(lb,(np.ones(len(lb))/3),'r--')
            axarr[1].plot(lb,(np.ones(len(lb))*2),'b--')
            axarr[1].set_title('Frequency ($Hz$) vs $L$\'')
            axarr[1].set_xlabel('$L$\' (backus length)')
            axarr[1].set_ylabel('$L$\' Backus Number')
            axarr[1].plot(lb,(freqs[i]*lb)/vsMin,label='%s Hz' % freqs[i])
            axarr[1].set_xlim(0, np.max(lb))
        plt.tight_layout()
        axarr[1].legend(loc='upper left',fontsize='large')

    plt.show()

    return time_curves, twt, rc, synth
