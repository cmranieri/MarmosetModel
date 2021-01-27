import numpy as np
import os
from scipy.signal import find_peaks, welch
import scipy.io as sio

flag_matlab = False


def window_data( data, window_size = 2000 ):
    w_data = [ data[ i*window_size : (i+1)*window_size ]
               for i in range( len(data) // window_size ) ]
    return np.array( w_data )


def clean_windowed_trial( trial ):
    clean_trial = list()
    for window in trial:
        peaks = [ find_peaks( window[:,i], prominence=None )[0]
                  for i in range(window.shape[-1]) ]
        avg = np.mean( window, axis=0 )
        std = np.std ( window, axis=0 )
        c = [ ( avg[i]==0. ) or
              ( np.abs( avg[i] ) < 0.2 and std[i] > 0.1 and
              len( peaks[i] ) >= 10 )
              for i in range( len(avg) ) ]
        if all(c):
            clean_trial.append( window )
    return np.array( clean_trial )


if __name__ == '__main__':
    condition = 'pd'
    os.chdir( '/home/caetano/datasets/lfp' )
    for filename in os.listdir( condition ):
        print( 'Processing %s...' % filename )
        data = np.load( os.path.join( condition, filename ) )
        # [ t, ch ]
        data = np.transpose( data, [ 1, 0 ] )
        # [ w, t, ch ]
        data = window_data( data )
        data = clean_windowed_trial( data )
        if len(data)==0:
            continue
        # [ w, ch, t ]
        data = np.transpose( data, [ 0, 2, 1 ] )
        #data_fft = np.zeros( data.shape, dtype=np.complex )
        lfp_f, lfp_dimensions = welch(data[0,0], 1000, nperseg=1024, detrend=False)
        data_fft = np.zeros( [ data.shape[0], data.shape[1], lfp_dimensions.shape[0] ],
                             dtype=np.float32 )
        # sample
        for i in range( data.shape[0] ):
            # electrode
            for j in range( data.shape[1] ):
                #data_fft[ i, j ] = np.fft.fft( data[ i, j ] )
                lfp_f, data_fft[ i, j ] = welch( data[ i, j ], 1000, nperseg=1024, detrend=False )
        del data
        np.save( os.path.join( 'fft', condition, filename ), data_fft )
        if flag_matlab:
            sio.savemat( os.path.join( 'fft', condition, filename+'.mat' ), {'psd':data_fft} )
        del data_fft
    np.save( os.path.join( 'fft', 'lfp_f' ), lfp_f )
