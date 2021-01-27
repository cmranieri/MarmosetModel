import numpy as np
import os
import re
from collections import defaultdict
import mne
from mne.io import RawArray
import neo
from neo.io import PickleIO
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


plots_dir = '../plots'
lfp_dir   = '/home/caetano/datasets/lfp/lfp_all'

areas_idx = { 'MI': 0,
              'PUT': 1,
              'GPE': 2,
              'GPI': 3,
              'VL': 4,
              'STN': 5,
              'VPL': 6 }


def plot_filters( seg_name, ch_name, data, filter1, filter2, high_corr ):
    seg_name = seg_name.replace( '.mat', '' )
    if not os.path.exists( os.path.join( plots_dir, seg_name ) ):
        os.mkdir( os.path.join( plots_dir, seg_name ) )
    fig, axs = plt.subplots(nrows = 2, ncols=2, figsize=(30,20) )

    start = np.random.randint( len(data[0]) - 128 )
    idx = np.arange( start, start+128 )

    axs[0,0].set_title( 'Raw data' )
    for line in data:
        axs[0,0].plot( line[ idx ] )
    axs[0,1].set_title( 'Bandpass filters' )
    for line in filter1:
        axs[0,1].plot( line[ idx ] )
    axs[1,0].set_title( 'Notch filter' )
    for line in filter2:
        axs[1,0].plot( line[ idx ] )
    axs[1,1].set_title( 'Good channels' )
    for line in high_corr:
        axs[1,1].plot( line[ idx ] )
    plt.subplots_adjust( hspace = 0.1 )
    plt.savefig( os.path.join( plots_dir, seg_name, ch_name ) + '.png', format='png' )
    plt.close()



def remove_bad_corr( raw ):
    coefs = np.corrcoef( raw.get_data() )
    coefs_ch = np.mean( coefs, axis = 0 )
    while np.min( coefs_ch ) < 0.7:
        ch_name = raw.ch_names[ np.argmin( coefs_ch ) ]
        raw = raw.drop_channels( ch_name )
        print( 'Removed %s' % ch_name )
        coefs = np.corrcoef( raw.get_data() )
        if not type(coefs) is np.ndarray: break
        coefs_ch = np.mean( coefs, axis = 0 )
    return raw


def process_signals( asigs_dict, seg_len, sfreq, seg_name ):
    lfp_sample = np.zeros( [ 7, seg_len ] )
    for key in asigs_dict:
        data = np.array( asigs_dict[ key ] )
        ch_names = [ key + str(i) for i in range( len( data ) ) ]
        info = mne.create_info( ch_names = ch_names, sfreq = sfreq )

        raw = RawArray( np.copy( data ), info )
        raw = raw.filter( l_freq=0.5, h_freq=250, picks = 'all' )
        filter1 = np.copy( raw.get_data() )

        raw = raw.notch_filter( [60, 120, 180], picks = 'all' )
        filter2 = np.copy( raw.get_data() )
        
        # z-score normalization
        scaler = StandardScaler()
        scaled = scaler.fit_transform( np.transpose( raw.get_data(), [1,0] ) )
        del raw
        raw = RawArray( np.transpose( scaled ), info )
        raw = remove_bad_corr( raw )
        high_corr = raw.get_data()

        #plot_filters( seg_name, key, data, filter1, filter2, high_corr )
        del data, filter1, filter2
        if key in areas_idx.keys():
            print(lfp_sample.shape[1])
            print( high_corr.shape )
            high_corr = high_corr[ :, :lfp_sample.shape[1] ]
            print( high_corr.shape )
            lfp_sample[ areas_idx[key] ] = np.mean( high_corr, axis = 0 )
    return lfp_sample



def seg2Dicts( seg ):
    asigs_L = defaultdict( list )
    asigs_R = defaultdict( list )
    # Signals, several for each region, referenced by name
    for asig in seg.analogsignals:
        name = asig.name
        # Discard underrepresented regions
        if re.match( '.*IC|.*EMG|.*SPKC|.*SI|.*WB', name ): continue
        # Given a signal, get its region
        name = re.match( 'LFP_([A-Z]+)', name ).groups()[0]
        name = re.sub( 'M\D*', 'MI', name )
        if asig.name[-1] == 'L':
            asigs_L[ name ].append( asig.rescale('V').magnitude.ravel() )
        elif asig.name[-1] == 'R':
            asigs_R[ name ].append( asig.rescale('V').magnitude.ravel() )
    return ( asigs_L, asigs_R )


def convert_segment( seg ):
    # Prepare information for the filters
    sfreq = int( seg.analogsignals[0].sampling_rate.magnitude )
    seg_len = np.min( [ len(x) for x in seg.analogsignals ] )
    # Convert segment to dict
    asigs_L, asigs_R = seg2Dicts( seg )
    # Process signals from dicts and return lfp arrays
    lfp_L = process_signals( asigs_L, seg_len, sfreq, seg.name + '_L' )
    lfp_R = process_signals( asigs_R, seg_len, sfreq, seg.name + '_R' )
    return ( lfp_L, lfp_R )



def convert_file( filename, out_dir = '/home/caetano/datasets/lfp' ):
    block = PickleIO( filename ).read()[0]
    print(block.name)

    for seg in block.segments:
        print(seg.name)
        if seg.analogsignals != []:
            lfp_L, lfp_R = convert_segment( seg )
            np.save( os.path.join( lfp_dir, seg.name.replace('.mat','') + '_L.npy' ), lfp_L )
            np.save( os.path.join( lfp_dir, seg.name.replace('.mat','') + '_R.npy' ), lfp_R )
            del lfp_L, lfp_R
    


def convert_all( data_dir = '/home/caetano/neuro4pd/datasets/python' ):
    filenames = [ os.path.join( data_dir, x ) for x in os.listdir( data_dir ) ]
    # Open each block
    for filename in filenames:
        convert_file( filename )

convert_all()

