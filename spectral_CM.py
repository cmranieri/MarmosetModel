import os
import re
import GenericBG  # neural network designed through netpyne, to be optimized
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import scipy.stats
from scipy.signal import firwin, lfilter
import numpy as np
import pandas
from GA import set_genotype
from GA_utils import load_all_genotypes, get_clusters, get_ordered_cluster
from GA_utils import get_data, get_fitness, calc_power_band
from collections import defaultdict
from statannot import add_stat_annotation
import pickle


#This function simulate a set of genotypes located in folder "results". See GA_utils.py. Only works for CM.

# load_sims: if True, loads results from simulations previously performed, stored in results/sim_results.pickle 
load_sims = False

# dir_fft: directory containing the subdirectories 'healthy' and 'pd', each with the .npy fft files
mfr_h  = defaultdict( list )
mfr_pd = defaultdict( list )
new_mfr = True


def loadSimMFR( sim, mfr ):
    mfr['GPi'].append( sim.allSimData.popRates['GPi'] )
    mfr['GPe'].append( sim.allSimData.popRates['GPe'] )
    mfr['StrD1'].append( sim.allSimData.popRates['StrD1'] )
    mfr['StrD2'].append( sim.allSimData.popRates['StrD2'] )
    mfr['TH'].append( sim.allSimData.popRates['TH'] )
    mfr['STN'].append( sim.allSimData.popRates['STN'] )
    mfr['CTX_RS'].append( sim.allSimData.popRates['CTX_RS'] )
    mfr['CTX_FSI'].append( sim.allSimData.popRates['CTX_FSI'] )
    return mfr


def meanMFR( cluster_id ):
    global new_mfr
    if new_mfr:
        with open( 'results/mfr.csv', 'w' ) as f:
            f.write( 'Cluster,Condition,Region,MFR [Hz]\n' )
        new_mfr = False
    keys = [ 'StrD1', 'StrD2', 'GPe', 'GPi', 'TH', 'STN', 'CTX_RS', 'CTX_FSI' ]
    global mfr_h, mfr_pd
    conditions = [ 'healthy', 'pd' ]
    for i, mfr in enumerate( [ mfr_h, mfr_pd ] ):
        for key in keys:
            for val in mfr[ key ]:
                with open( 'results/mfr.csv', 'a' ) as f:
                    f.write( '%d,%s,%s,%0.2f\n' % ( cluster_id, conditions[i], key, val ) )
                print( '%d,%s,%s,%0.2f' % ( cluster_id, conditions[i], key, val ) )
            m = np.mean( mfr[ key ] )
            s = np.std( mfr[ key ] )
            print( '%s,%s,%0.2f \pm %0.2f' % ( conditions[i], key, m, s ) )


def simulate_CM(a, b, pd, genotype=None):
    global mfr_h, mfr_pd
    lfp_fft_list = list()
    power_list = list()
    N = 1
    for i in range( N ):
        net = GenericBG.Network( t_sim=2000, has_pd=pd )
        # Comment the line below to run the rat model
        net = set_genotype( genotype, net )
        sim = net.simulate( dt=0.1, lfp=True )
        if pd:
            mfr_pd = loadSimMFR( sim, mfr_pd )
        else:
            mfr_h  = loadSimMFR( sim, mfr_h )
        lfp_f, lfp_fft = net.extractLFP_SP()
        
        lfp_fft = np.transpose( lfp_fft, [1, 0] )
        lfp_fft[ 0, : ]  = 0.
        lfp_fft = lfp_fft / np.max( lfp_fft, axis=0 )
        lfp_fft = np.transpose( lfp_fft, [1, 0] )

        power = calc_power_band( lfp_fft, lfp_f, a, b )
        lfp_fft_list.append( lfp_fft )
        power_list.append( power )
    return np.mean( lfp_fft_list, axis=0 ), lfp_f, np.array( power_list )

   

def plot_psd( psd_h_list, psd_pd_list, lfp_f, area_names ):
    cluster_colors = [ 'orange', 'purple' ]
    sns.set( style='darkgrid' )
    fig, axs = plt.subplots( nrows=6, ncols=4, sharex=True, sharey='row', figsize=(8,5) )
    for c in range( 2 ):
        psd_h  = psd_h_list[  c ]
        psd_pd = psd_pd_list[ c ]
        print( 'shape',psd_h.shape )
        for i in range( psd_h.shape[0] ):
            if i<4:
                row = c
                row_cl = 2
            else:
                row = c + 3
                row_cl = 5
            R = psd_pd[i] / psd_h[i]
            threshold = np.percentile( psd_h[i,:50], 80 )
            R[ (psd_pd[i] < threshold) & (psd_h[i] < threshold) ] = 0.
            axs[row_cl, i%4].plot( lfp_f, R, color = cluster_colors[c] )
            #axs[row_cl, i%4].set_ylim( [ 0., 0.3 ] )
            axs[row_cl, 0].set_ylabel( '$R$' )
            axs[row, i%4].plot( lfp_f, psd_h[  i ], color='blue' )
            axs[row, i%4].plot( lfp_f, psd_pd[ i ], color='red' )
            if c == 0:
                axs[row, i%4].set_title( area_names[i] )
            elif c == 1:
                axs[row, i%4].set_ylim( 0, 1 )
            axs[row, i%4].set_xlim( 0, 50 )
            axs[row, i%4].set_xticks( [ 0, 10, 20, 30, 40, 50 ] )
            axs[row, i%4].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axs[row, 0].set_ylabel( 'c%d\n[a.u.]' % (c+1) )
    axs[0, 0].set_ylim( 0, 0.2 )
    axs[3, 0].set_ylim( 0, 0.3 )
    axs[5,0].set_xlabel( 'Frequency [$Hz$]' )
    axs[1,0].legend( ['Healthy', 'PD'] )
    axs[2,0].legend( ['c1', 'c2'] )
    fig.subplots_adjust( wspace=0.3 )
    fig.tight_layout()
    plt.savefig( 'results/CM_PSD.pdf' )
    plt.clf()


def plot_power( power_h_list, power_pd_list, area_names ):
    sns.set( style='darkgrid' )
    matplotlib.rcParams['axes.titlepad'] = 15
    fig, axs = plt.subplots( ncols=len(power_h_list), 
                             sharey = 'col',
                             nrows=power_h_list[0].shape[1], 
                             sharex=True, figsize=(3,9) )
    # for each cluster j
    for j in range( 2 ):
        power_h  = np.array( power_h_list[  j ], dtype='float32' )
        power_pd = np.array( power_pd_list[ j ], dtype='float32' )
        stat   = list()
        pvalue = list()
        for i in range( power_h.shape[1] ):
            lfp_stats =  [ [ 'H',  x ] for x in  power_h [ :, i ] ]
            lfp_stats += [ [ 'PD', x ] for x in  power_pd[ :, i ] ]
            #lfp_stats = np.array( [power_h[:,i], power_pd[:,i]] )
            #lfp_stats = lfp_stats.transpose()
            df = pandas.DataFrame( lfp_stats, columns=['Condition','Power'] )
            sns.boxplot( ax = axs[i,j], data=df, x='Condition', y='Power' )
            test_results = add_stat_annotation( axs[i,j], data=df, x='Condition', y='Power',
                                                box_pairs=[ ( 'H', 'PD' ) ],
                                                test='t-test_paired', text_format='star',
                                                loc='inside', verbose=2 )
            axs[i,j].set_xlabel('')
            axs[i,j].set_ylabel('')
            axs[i,0].set_ylabel( '%s\n[a.u.]' % area_names[i] )
            s, p = scipy.stats.ttest_ind( power_h[:,i], power_pd[:,i] )
            stat.append( s )
            pvalue.append( p )
    axs[0,0].set_title( 'Cluster 1' )
    axs[0,1].set_title( 'Cluster 2' )
    fig.subplots_adjust( wspace=0.3 )
    fig.tight_layout()
    plt.savefig( 'results/CM_power.pdf' )
    plt.clf()


def simulate_all_CM( genotypes, cluster_id ):
    psd_h    = list()
    psd_pd   = list()
    power_h  = list()
    power_pd = list()
    for k, genotype in enumerate( genotypes ):
        avg_fft_healthy, lfp_f, sims_power_healthy = simulate_CM( 13, 31, False, genotype )
        avg_fft_PD, lfp_f, sims_power_PD = simulate_CM( 13, 31, True, genotype )
        psd_h.append(    avg_fft_healthy )
        psd_pd.append(   avg_fft_PD )
        power_h  += list( sims_power_healthy )
        power_pd += list( sims_power_PD )
    power_h  = np.array( power_h )
    power_pd = np.array( power_pd )
    psd_h_mean  = np.mean( psd_h, axis=0 )
    psd_pd_mean = np.mean( psd_pd, axis=0 )
    meanMFR( cluster_id )
    return psd_h_mean, psd_pd_mean, lfp_f, power_h, power_pd


def simulate_all_clusters( genotypes, clusters ):
    N_samps = 50
    psd_h_list    = list()
    psd_pd_list   = list()
    power_h_list  = list()
    power_pd_list = list()
    for i in range( max( clusters ) + 1 ):
        ordered_cluster = get_ordered_cluster( genotypes, clusters, i )
        psd_h, psd_pd, lfp_f, power_h, power_pd = simulate_all_CM( ordered_cluster[ :N_samps ], i )
        psd_h_list.append(    psd_h )
        psd_pd_list.append(   psd_pd )
        power_h_list.append(  power_h )
        power_pd_list.append( power_pd )
    with open( 'results/sim_resuts.pickle', 'wb' ) as f:
        pickle.dump( { 'psd_h_list': psd_h_list,
                       'psd_pd_list': psd_pd_list,
                       'power_h_list': power_h_list,
                       'power_pd_list': power_pd_list,
                       'lfp_f': lfp_f }, f )
    return psd_h_list, psd_pd_list, power_h_list, power_pd_list, lfp_f
        

def load_all_clusters():
    with open( 'results/sim_resuts.pickle', 'rb' ) as f:
        res = pickle.load( f )
    psd_h_list  = res[ 'psd_h_list' ]
    psd_pd_list = res[ 'psd_pd_list' ]
    power_h_list  = res[ 'power_h_list' ]
    power_pd_list = res[ 'power_pd_list' ]
    lfp_f = res[ 'lfp_f' ]
    return psd_h_list, psd_pd_list, power_h_list, power_pd_list, lfp_f


def make_plots( psd_h, psd_pd, power_h, power_pd, lfp_f ):
    area_names=["StrD1", "StrD2", "TH", "GPi", "GPe", "CtxRS", "CtxFSI", "STN"]
    plot_psd( psd_h, psd_pd, lfp_f, area_names )
    plot_power( power_h, power_pd, area_names )


if __name__ == '__main__':
    genotypes, _,_,_ = load_all_genotypes()
    clusters  = get_clusters( genotypes, False )
    if not load_sims:
        psd_h, psd_pd, power_h, power_pd, lfp_f = simulate_all_clusters( genotypes, clusters )
    else:
        psd_h, psd_pd, power_h, power_pd, lfp_f = load_all_clusters()
    make_plots( psd_h, psd_pd, power_h, power_pd, lfp_f )
