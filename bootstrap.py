import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

data_dir = '/home/caetano/datasets/lfp/bootstrap'


def load_file( fname ):
    data = list()
    with open( os.path.join(data_dir, fname), 'r' ) as f:
        for row in f.readlines():
            data.append( row.split(',') )
        data = np.array( data, dtype=np.float32 )
    return data

h_dict = dict()
pd_dict = dict()

#sns.set( style='darkgrid' )
area_names=["M1", "PUT", "GPe", "GPi", "VL", "STN", "VPL", ""]
fig, axs = plt.subplots( nrows=3, ncols=4, sharex=True, sharey='row', figsize=(9,3) )
for fname in sorted( os.listdir( data_dir ) ):
    data = load_file( fname )
    mtc = re.match( '(H|PD)_(\w+)\.csv', fname )
    condition = mtc.groups()[0]
    area = mtc.groups()[1]
    idx = np.where( [ area == area_names[i] for i in range(7) ] )[0][0]
    if condition == 'PD':
        pd_dict[area] = data
        color = 'red'
    elif condition == 'H':
       h_dict[area] = data
       color = 'blue'
    axs[ idx//4, idx%4 ].fill_between( data[:,0], data[:,2], data[:,3], edgecolor=color, facecolor=color, alpha=0.4 )
    axs[ idx//4, idx%4 ].plot( data[:,0], data[:,1], linewidth=0.7, color=color )
    axs[ idx//4, idx%4 ].set_title( area )
    axs[ idx//4, idx%4 ].set_xlim( (0,50) )
    axs[ idx//4, idx%4 ].set_ylim( (0,0.8) )
    axs[ idx//4, 0 ].set_ylabel( 'PSD\n[a.u.]' )
axs[ 0, 0 ].legend( ['Healthy', 'PD'] )

for i, area in enumerate( area_names[:4] ):
    pd = pd_dict[ area ]
    h  = h_dict[ area ]
    R = pd[:,1] / h[:,1]
    threshold = np.percentile( h[:50], 50 )
    R[ (pd[:,1] < threshold) & (h[:,1] < threshold) ] = 0.
    axs[ 2, i ].plot( h[:,0], R, linewidth=1., color='purple' )
    axs[ 2, i ].set_title( area )
axs[ 2, 0 ].set_ylabel('$R$')
axs[ 2, 0 ].set_xlabel('Frequency (Hz)')

fig.tight_layout()
fig.savefig('results/bootstrap_spectral_animal.pdf')




