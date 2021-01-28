import GenericBG
import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from elephant.statistics import mean_firing_rate
from sklearn.decomposition import PCA
import GA
from GA_utils import get_clusters, load_all_genotypes
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import f_oneway
import seaborn as sns
import pandas
import random
from statannot import add_stat_annotation

t_sim = 2000
bins  = 50


def simulate_net( genotype = None, pd = False, seeds=None ):
    random.seed( 1 )
    net = GenericBG.Network( t_sim = t_sim, has_pd = pd )
    if genotype is not None:
        net = GA.set_genotype( genotype, net )
    sim = net.simulate( dt = 0.1, lfp = False, seeds=seeds )
    all_spikes = net.extractSpikes()
    return all_spikes


def get_mfr( all_spikes, join_regions=False ):
    all_mfr = list()
    regions_mean = list()
    # Process each region separately
    for key in all_spikes.keys():
        mfr_region = list()
        # Spike times of each electrode within the given region
        spikes_list = [ spk.times for spk in all_spikes[ key ] ]
        # For each electrode
        for spikes in spikes_list:
            spikes = np.array( spikes )
            mfr = list()
            for t_start in range( 0, t_sim, bins ):
                mfr.append( mean_firing_rate( spikes,
                                              t_start = t_start,
                                              t_stop  = t_start + bins ) )
            all_mfr.append( np.array( mfr ) )
            mfr_region.append( np.array(mfr) )
        regions_mean.append( np.mean(mfr_region, axis=0) )
    if join_regions:
        return regions_mean
    return np.transpose( all_mfr, [1, 0] )


def mfr2comps( all_mfr ):
    pca = PCA( n_components = 3 )
    comps = pca.fit_transform( all_mfr )
    comps = np.transpose( comps, [1,0] )
    return comps


def get_spk_comps( genotype=None, pd='False', seeds=None ):
    all_spikes = simulate_net( genotype, pd=pd, seeds=seeds )
    all_mfr    = get_mfr( all_spikes )
    comps      = mfr2comps( all_mfr )
    return comps


def calc_all_comps():
    genotypes, _, _, _ = load_all_genotypes()
    clusters = get_clusters( genotypes )
    for cluster_id in range( max(clusters) + 1 ):
        comps_h  = list()
        comps_pd = list()
        for genotype in genotypes[ np.where( clusters == cluster_id )[0] ]:
            comps_h.append(  get_spk_comps( genotype, pd=False ) )
            comps_pd.append( get_spk_comps( genotype, pd=True ) )
        with open( os.path.join( 'results', 'comps_%d.pickle'%cluster_id ), 'wb' ) as f:
            pickle.dump( { 'comps_h':  comps_h,
                           'comps_pd': comps_pd }, f )


def load_all_comps( cluster_id ):
    with open( os.path.join( 'results', 'comps_%d.pickle'%cluster_id ), 'rb' ) as f:
        comps = pickle.load( f )
    ret = ( np.transpose(comps[ 'comps_h' ], [0,2,1]),
            np.transpose(comps[ 'comps_pd' ], [0,2,1]) )
    return ret


def calc_all_dtw( cluster_id ):
    comps_h, comps_pd = load_all_comps( cluster_id )
    dists_h    = list()
    dists_pd   = list()
    dists_h_pd = list()
    for i in range( comps_h.shape[0] ):
        for j in range( i+1, comps_h.shape[0] ):
            print( i, j )
            distance, path = fastdtw( comps_h[i], comps_h[j], dist=euclidean )
            dists_h.append( distance )
            distance, _ = fastdtw( comps_pd[i], comps_pd[j] )
            dists_pd.append( distance )
            distance, _ = fastdtw( comps_h[i], comps_pd[j] )
            dists_h_pd.append( distance )
    with open( os.path.join( 'results', 'dists_%d.pickle'%cluster_id ), 'wb' ) as f:
        pickle.dump( { 'dists_h':    dists_h,
                       'dists_pd':   dists_pd,
                       'dists_h_pd': dists_h_pd}, f )
 

def load_all_dtw( cluster_id ):
    with open( os.path.join( 'results', 'dists_%d.pickle'%cluster_id ), 'rb' ) as f:
        dists = pickle.load( f )
    ret = ( dists['dists_h'], dists['dists_pd'], dists['dists_h_pd'] )
    return ret


def dists_stats( dists_h, dists_pd, dists_h_pd ):
    print( np.mean(dists_h), u'\u00B1', np.std(dists_h), 'size: %d'%len(dists_h) )
    print( np.mean(dists_pd), u'\u00B1', np.std(dists_pd), 'size: %d'%len(dists_pd) )
    print( np.mean(dists_h_pd), u'\u00B1', np.std(dists_h_pd), 'size: %d'%len(dists_h_pd) )


def oneway_anova( *args ):
    stats, pvalue = f_oneway( *args )
    return pvalue


def plot_hists():
    dists_h, dists_pd, dists_h_pd = load_all_dtw() 
    fig, axs = plt.subplots( 1, 3, sharex=True, sharey=True, figsize=(10,4) )
    axs[0].set_title( 'Healthy x Healthy' )
    axs[0].hist(dists_h, bins=25)
    axs[1].set_title( 'PD x PD' )
    axs[1].hist(dists_pd, bins=25)
    axs[2].set_title( 'Healthy x PD' )
    axs[2].hist(dists_h_pd, bins=25)
    plt.subplots_adjust( wspace=0.7 )
    plt.tight_layout()
    plt.savefig( os.path.join( 'results', 'hist_dynamics.png' ) )
    plt.show()


def plot_dists( n_clusters ):
    sns.set( style='darkgrid' )
    #for i in range( n_clusters ):
    # Plot only cluster 1 (ixd = 0)
    for i in range( n_clusters ):
        fig, axs = plt.subplots( ncols=1, figsize=(3,3) )
        axs.set_title( 'Cluster %d'%(i+1) )
        dists_list = load_all_dtw( cluster_id = i )
        dists_arr  = np.transpose( dists_list, [1,0] )

        dists_stats( dists_arr[:,0], dists_arr[:,1],  dists_arr[:,2] )

        dists  = [ [ 'H x H', d ] for d in dists_arr[:,0] ]
        dists += [ [ 'PD x PD', d ] for d in dists_arr[:,1] ]
        dists += [ [ 'H x PD', d ] for d in dists_arr[:,2] ]
        df = pandas.DataFrame( dists, columns=[ 'Conditions', 'DTW' ] )
        
        sns.boxplot( ax = axs, data = df, x='Conditions', y='DTW' )
        box_pairs = [ ('H x H', 'PD x PD'),
                      ('H x H', 'H x PD'),
                      ('PD x PD', 'H x PD')]

        test_results = add_stat_annotation( axs, data=df, x='Conditions', y='DTW',
                                            box_pairs=box_pairs,
                                            test='t-test_ind', text_format='star',
                                            loc='inside', verbose=2 )

        plt.tight_layout()
        fig.savefig( 'results/dists_%d.pdf' % i )
        plt.clf()



if __name__=='__main__':
    calc_all_comps()
    for i in range( 2 ):
        calc_all_dtw( cluster_id=i )
    plot_dists( n_clusters=2 )
