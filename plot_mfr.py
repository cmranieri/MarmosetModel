import pandas
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from statannot import add_stat_annotation

data = pandas.read_csv( 'results/mfr.csv' )
print( data )

sns.set( style='darkgrid' )
matplotlib.rcParams[ 'axes.titlesize' ] = 'large'
matplotlib.rcParams[ 'axes.labelsize' ] = 'large'
# For each cluster i
for i in range( 2 ):
    fig, axs = plt.subplots( ncols=1, figsize=(6,3) )
    axs.set_title( 'Cluster %d' % (i+1) )
    df = data[ data['Cluster']==i ] 
    sns.barplot( ax=axs, x='Region', y='MFR [Hz]', hue='Condition', data=df )
    box_pairs = [((x, 'healthy'), (x, 'pd')) for x in df['Region'].unique()]
    print( box_pairs )
    test_results = add_stat_annotation( axs, data=df, x='Region', y='MFR [Hz]', hue='Condition',
                                        box_pairs=box_pairs,
                                        test='t-test_paired', text_format='star',
                                        loc='inside', verbose=2 )
    plt.tight_layout()
    plt.savefig( 'results/mfr_%d.pdf' % i )
    plt.clf()
