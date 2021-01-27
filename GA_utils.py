import numpy as np
import os
from collections import defaultdict
import re
from bisect import bisect_left
from scipy.integrate import simps
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt


# Integrates freq_data in the interval [a,b]
def calc_power_band(lfp_fft, lfp_f, a, b):
    res = list()
    a_ = bisect_left(lfp_f, a)
    b_ = bisect_left(lfp_f, b)
    # Integrate for each channel
    for ch in lfp_fft:
        res.append( simps( ch[a_:b_], lfp_f[a_:b_] ) )
    return np.array(res)


def get_data( filename ):
    with open( filename ) as f:
        filerows = list()
        new_row = ''
        for row in f.readlines():
            new_row += row
            if ']' in row:
                new_row = re.sub( '[\n,\[\]]','',new_row )
                new_row = new_row.split()
                filerows.append( new_row )
                new_row = ''
    filerows = np.array( filerows, dtype=np.float32 )
    return filerows


def build_dict( data ):
    data_dict = defaultdict( list )
    for row in data:
        data_dict[ int( row[0] ) ].append( row[1:] )
    return data_dict


def get_fitness( data ):
    data_dict = build_dict( data )
    best = list(); mean = list();
    best_genotype = list()
    for gen in sorted( data_dict.keys() ):
        all_fits = np.array( data_dict[ gen ] )[ :, 1 ]
        best.append( all_fits[ 0 ] )
        mean.append ( np.mean( all_fits ) )
        best_genotype = np.array(data_dict[ gen ])[ 0 ]
    return best_genotype, best, mean


def load_all_genotypes():
    means  = list()
    bests = list()
    bests_genotypes = list()
    count = 0
    for filename in os.listdir( 'results' ):
        if not re.match( 'inspyred\-individuals\-file.*\.csv', filename ): continue
        print(filename)
        data = get_data( os.path.join( 'results', filename ) )
        best_genotype, best, mean = get_fitness( data )
        bests_genotypes.append( best_genotype )
        means.append(  mean )
        bests.append( best )
        count += 1
    bests_genotypes = np.array( bests_genotypes )
    bests_genotypes = bests_genotypes[ (-bests_genotypes[:,1]).argsort() ]
    fits_bestg = bests_genotypes[ :, 1 ]
    bests_genotypes = bests_genotypes[ :, 2: ]
    
    return bests_genotypes, fits_bestg, bests, means


def get_clusters( genotypes, default=False, n_clusters=2 ):
    if default:
        clusters = default_clusters
    else:
        kmeans = KMeans( n_clusters=n_clusters, random_state=20, max_iter=3000, tol=1e-12 ).fit( genotypes )
        clusters = kmeans.labels_
    return clusters


def get_ordered_cluster( genotypes, clusters, cluster_id ):
    sil = silhouette_samples( genotypes, clusters )
    ordered_idx = np.argsort( - sil )
    idx = np.where( clusters[ ordered_idx ] == cluster_id )[0]
    return genotypes[ ordered_idx ][ idx ]


def get_targets_animal( pd = True, 
                        dir_fft = '/home/caetano/datasets/lfp/fft' ):
    cwd = os.getcwd()
    if pd: os.chdir( os.path.join( dir_fft, 'pd' ) )
    else: os.chdir( os.path.join( dir_fft, 'healthy' ) )
    
    lfp_f = np.load( '../lfp_f.npy' )
    data = list()
    # Load all models from the "results" directory
    for filename in os.listdir():
        if not re.match( '.*\.npy', filename ): continue
        # [ b, f, t ]
        trial    = np.load( filename )
        # [ [ t, f ] ]
        data += list( np.transpose( trial, [ 0, 2, 1 ] ) )
    # [ f, t, b ]
    data = np.transpose( data, [ 2, 1, 0 ] )
    fft = np.zeros( data.shape[:-1] )
    
    coefs = list()
    for b_fft in np.transpose( data, [2,0,1] ):

        denominator = calc_power_band( b_fft, lfp_f, 0, 51 )
        alpha = calc_power_band( b_fft, lfp_f, 8, 14 )
        beta  = calc_power_band( b_fft, lfp_f, 13, 31 )
        betaH = calc_power_band( b_fft, lfp_f, 30, 51 )
        coefs.append( ( alpha + beta + betaH ) / denominator )
    print( 'COEFS', np.nanmean(coefs, axis=0) )
    os.chdir( cwd )
    return np.nanmean( coefs, axis=0 )
    

if __name__ == '__main__':
    get_targets_animal()
    #genotypes,_,_,_ = load_all_genotypes()
    #get_clusters( genotypes )

