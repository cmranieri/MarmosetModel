import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler

d = '/home/caetano/datasets/lfp/pd'
start = 7000


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



for fname in os.listdir( d ):
    data = np.load( os.path.join( d, fname ) )
    sample = data[ :, start : start + 2000 ]
    print( sample.shape )

    matplotlib.rcParams['xtick.labelsize']='large'
    matplotlib.rcParams['ytick.labelsize']='large'
    fig, axs = plt.subplots( 7, 1, figsize=(9,5) )
    titles = ["M1", "PUT", "GPe", "GPi", "VL", "STN", "VPL"]
    for i, k in enumerate( [ 1, 2, 3, 4, 6, 0, 5 ] ):
        sample_ch = sample[ k ]

        sample_ch = butter_bandpass_filter( sample_ch, 8, 50, 1000 )
        sample_ch = np.reshape( sample_ch, [-1,1] )
        scaler = StandardScaler()
        sample_ch = scaler.fit_transform( sample_ch )

        axs[i].plot( sample_ch )
        axs[i].set_ylabel( '%s\n[a.u.]' % titles[k], fontsize='large' )
        axs[i].set_xlim( 0, len(sample[k]) )
        axs[i].set_ylim( np.min( sample_ch ), np.max( sample_ch ) )
        axs[i].set_xticks( [] )
        axs[i].set_yticks( [] )
        #axs[i].ticklabel_format( style='sci', scilimits=(0,0) )
    axs[6].set_xticks( [ 0, 500, 1000, 1500, 2000 ] )
    axs[6].set_xlabel( 'Time ($ms$)', fontsize='large' )
    fig.tight_layout()
    plt.subplots_adjust( hspace = 1.5 )
    plt.savefig( 'samples/sample_%s.pdf'%fname )
    plt.close()
