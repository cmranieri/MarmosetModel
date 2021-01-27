import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import GenericBG
from GA import set_genotype
from GA_utils import load_all_genotypes, get_clusters, get_ordered_cluster
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler


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



genotypes, _,_,_ = load_all_genotypes()
clusters  = get_clusters( genotypes, False )
# Cluster 1
ordered_cluster = get_ordered_cluster( genotypes, clusters, cluster_id=0 )
net = GenericBG.Network( t_sim=2000, has_pd=True )
# Genotype with highest silhouette within cluster 1
net = set_genotype( ordered_cluster[ 1 ], net )
sim = net.simulate( dt=0.1, lfp=True )
sample = net.extractLFP_raw()
print( sample.shape )

matplotlib.rcParams['xtick.labelsize']='large'
matplotlib.rcParams['ytick.labelsize']='large'
fig, axs = plt.subplots( 8, 1, figsize=(9,5) )
titles = [ 'StrD1', 'StrD2', 'GPe', 'GPi', 'TH', 'RS', 'FSI', 'STN' ]
for i in range( 8 ):
    sample_ch = sample[ i ]

    sample_ch = butter_bandpass_filter( sample_ch, 8, 50, 1000 )
    sample_ch = np.reshape( sample_ch, [-1,1] )
    scaler = StandardScaler()
    sample_ch = scaler.fit_transform( sample_ch )

    axs[i].plot( sample_ch )
    axs[i].set_ylabel( '%s\n[a.u.]' % titles[i], fontsize='large' )
    axs[i].set_xlim( 0, len(sample[i]) )
    axs[i].set_ylim( np.min( sample_ch ), np.max( sample_ch ) )
    axs[i].set_xticks( [] )
    axs[i].set_yticks( [] )
    #axs[i].ticklabel_format( style='sci', scilimits=(0,0) )
axs[7].set_xticks( [ 0, 500, 1000, 1500, 2000 ] )
axs[7].set_xlabel( 'Time ($ms$)', fontsize='large' )
fig.tight_layout()
plt.subplots_adjust( hspace = 1.5 )
plt.savefig( 'results/sample.pdf' )
plt.close()





