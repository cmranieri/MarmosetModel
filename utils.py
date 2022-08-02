import numpy as np
from elephant.statistics import mean_firing_rate


def compute_mfr( all_spikes, t_sim, bins=50, step=None ):
    if step is None:
        step = bins
    regions = dict()
    # Process each region separately
    for key in sorted( all_spikes.keys() ):
        mfr_region = list()
        # Spike times of each electrode within the given region
        spikes_list = [ spk.times for spk in all_spikes[ key ] ]
        spikes = sorted( sum(spikes_list, [] ) )
        spikes = np.array( spikes, dtype=np.float32 )
        mfr = list()
        for t_start in range( 0, t_sim-bins+1, step ):
        #for t_start in range( t_sim - bins ):
            mfr.append( mean_firing_rate( spikes,
                                          t_start = t_start,
                                          t_stop  = t_start + bins ) )
        regions[ key ] = np.array( mfr, dtype=np.float32 )
    return regions