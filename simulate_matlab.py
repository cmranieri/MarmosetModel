import numpy as np
import scipy.io as sio
import GenericBG
from GA import set_genotype
from GA_utils import load_all_genotypes, get_clusters, get_ordered_cluster

N_individuals = 5
N_repeats = 10

genotypes, _,_,_ = load_all_genotypes()
clusters  = get_clusters( genotypes, False )
ordered_cluster = get_ordered_cluster( genotypes, clusters, 0 )
subset = ordered_cluster[ :N_individuals ]
for i, genotype in enumerate( subset ):
    for j in range( N_repeats ):
        for has_pd in [ True, False ]:
            net = GenericBG.Network( t_sim = 100*1000,
                                     has_pd = has_pd,
                                     seed = j )
            net = set_genotype( genotype, net )
            sim = net.simulate( dt = 0.05, lfp=False )
            out_dict  = net.extractSpikes()
            out_dict[ 'freq_disp' ] = net.extractMFR()
            out_dict[ 'gsngi' ]     = net.get_gsngi()
            out_dict[ 'Striat_APs_dr' ]   = out_dict.pop( 'dStr_APs' )
            out_dict[ 'Striat_APs_indr' ] = out_dict.pop( 'iStr_APs' )
            pd_lbl = 'healthy'
            if has_pd:
                pd_lbl = 'pd'
            sio.savemat( 'matlab/sim_i%d_%s_r%d.mat' % (i, pd_lbl, j), out_dict )
