import numpy as np
import os
import re
from math import pi
from GA import transform_cand, inv_transform
from GA_utils import get_data, get_fitness, load_all_genotypes
from GA_utils import get_clusters, get_ordered_cluster
import dynamics
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#import chart_studio.plotly as py
import scipy.io as sio
import seaborn as sns
import pandas
import pickle
from spectral_CM import simulate_all_clusters, make_plots


colors = [ 'rgba(31, 119, 180,1)', 'rgba(255, 127, 14,1)',
           'rgba(44, 160, 44,1)', 'rgba(214, 39, 40,1)',
           'rgba(148, 103, 189,1)', 'rgba(140, 86, 75,1)',
           'rgba(227, 119, 194,1)', 'rgba(127, 127, 127,1)',
           'rgba(188, 189, 34,1)', 'rgba(23, 190, 207,1)' ]


colors_t = [ 'rgba(31, 119, 180,0.5)', 'rgba(255, 127, 14,0.5)',
             'rgba(44, 160, 44,0.5)', 'rgba(214, 39, 40,0.5)',
             'rgba(148, 103, 189,0.5)', 'rgba(140, 86, 75,0.5)',
             'rgba(227, 119, 194,0.5)', 'rgba(127, 127, 127,0.5)',
             'rgba(188, 189, 34,0.5)', 'rgba(23, 190, 207,0.5)' ]


#default_clusters = np.array( [ 5,  9,  5,  4,  9,  9,  1,  1,  9,  1,  9,  2,  1,  4,  9,
#                               7,  1,  4,  1,  4,  6,  6,  9,  1,  4,  4,  4,  5,  6,  5, 
#                               3,  7,  4,  1,  5,  6,  5,  8,  4,  4, 11,  5,  3,  3,  8,
#                               2,  9,  1,  7,  5,  9,  2,  3,  2,  3,  9,  2,  3,  8,  9,
#                               5,  9,  2,  3,  0,  3,  7,  0,  6,  1,  3,  8,  6,  3,  9, 
#                               4,  1,  3, 11,  0,  4, 11,  1,  1,  9,  8,  6,  6, 11, 11,
#                               10,  3,  1,  6, 11, 10, 11, 10,  1,  3] )



def plot_genotype( values, ax, values2=None ):
    values = list( values )
    N = len( values )
    x_as = [n / float(N) * 2 * pi for n in range(N)]
    # Because our chart will be circular we need to append a copy of the first
    # value of each list at the end of each list with data
    values += values[:1]
    x_as += x_as[:1]
    # Set color of axes
    plt.rc('axes', linewidth=0.5, edgecolor="#888888")
    # Create polar plot
    # Set clockwise rotation. That is:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    # Set position of y-labels
    ax.set_rlabel_position(0)
    # Set color and linestyle of grid
    ax.xaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)
    ax.yaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)
    # Set number of radial axes and remove labels
    ax.set_xticks(x_as[:-1])
    # Set ticks
    ax.set_xticklabels( [ '' for i in range(N) ] )
    ax.set_yticks( [0.0,0.25,0.5,0.75,1.0], minor=True )
    ax.set_yticklabels( ['', '', '', '',''] )
    # Plot data
    ax.set_ylim(0, 1.0)
    ax.plot(x_as, np.array(values), label='Marmoset', color='blue', linewidth=1, linestyle='solid', zorder=3)
    # Fill area
    ax.fill(x_as, values, facecolor='blue', alpha=0.5)
    if values2 is not None:
        values2_ = values2 + values2[:1]
        ax.plot(x_as, np.array(values2_), label='Rat', color='green', linewidth=1, linestyle='solid', zorder=3)
        ax.fill(x_as, values2_, facecolor='green', alpha=0.5)


def plot_dendrogram(model, **kwargs):
    # Children of hierarchical clustering
    children = model.children_
    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])
    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.savefig('results/dendrogram.pdf')


def plot_rat_genotype( bests_genotypes, clusters, clusters_id=[0,1] ):
    x = [ 1.2*1e-3, 3.*1e-3, 3.*1e-3, 5.*1e-3, 10.*1e-3, 1.0, 10.,10.,10.,10.,10.,10.,10.,10. ]
    rat_genotype = inv_transform( x )
    fig, axs = plt.subplots( nrows=len(clusters_id),
                             ncols=1,
                             figsize=(3,5),
                             subplot_kw = dict(projection='polar') )
    for cluster_id in clusters_id:
        ordered_cluster = get_ordered_cluster( bests_genotypes, clusters, cluster_id )
        #plot_genotype( ordered_cluster[ 0 ], axs[ cluster_id ], values2 = rat_genotype )
        plot_genotype( np.mean(ordered_cluster, axis=0), axs[ cluster_id ], values2 = rat_genotype )
    axs[ 0 ].legend( fontsize='x-large' )
    fig.tight_layout()
    plt.savefig( 'results/rat_gen.pdf' )

 


def plot_all_genotypes( bests_genotypes, fits_bestg, n_considered=2, minlen=0, use_default_clusters=False ):
    clusters = get_clusters( bests_genotypes, default=use_default_clusters )
    clusters_len = [ list(clusters).count(i) for i in range( max(clusters)+1 ) ]
    clusters_best = list()
    if minlen is not None:
        n_considered   = sum( [ x >= minlen for x in clusters_len ] )
    else:
        minlen = sorted( clusters_len, reverse=True )[ n_considered-1 ]
    fig, axs = plt.subplots( nrows=n_considered,
                             ncols=4,
                             figsize=(10,5),
                             subplot_kw = dict(projection='polar') )
    row_id = 0
    for cluster_id, cluster_len in enumerate( clusters_len ):
        if cluster_len >= minlen:
            pos = np.where( clusters == cluster_id )[0]
            clusters_best.append( bests_genotypes[ pos[0] ] )
            print( 'Cluster %d:'%cluster_id, pos, len(pos) )
            cmedian = np.median( fits_bestg[ list(pos) ] )
            cmax    = np.max( fits_bestg[ list(pos) ] ) 
            cmin    = np.min( fits_bestg[ list(pos) ] )
            ordered_cluster = get_ordered_cluster( bests_genotypes, clusters, cluster_id )
            for i in range(4):
                #genotype = bests_genotypes[ pos[i] ]
                genotype = ordered_cluster[ i ]
                plot_genotype( list(genotype), axs[row_id,i] )
            axs[ row_id, 0 ].set_ylabel( 'Best: %0.2f\nWorst: %0.2f\nMedian: %0.2f' % ( cmax, cmin, cmedian ),
                                         size='xx-large', labelpad=30 )
            row_id += 1
    cat = [ str(k+1) for k in range(14) ]
    axs[0,0].set_xticklabels( cat, fontsize='xx-large' )
    plt.subplots_adjust( wspace=0.7, hspace=0.7 )
    fig.tight_layout()
    plt.savefig( 'results/genotypes.pdf' )
    return clusters_best, clusters


def build_comps_dataframe( comps1, comps2, cluster_id ):
    df = pandas.DataFrame()
    df[ 'condition' ] = [ 'H' for i in range( len( comps1[0][0] ) ) ] +\
                        [ 'PD' for i in range( len( comps2[0][0] ) ) ] 
    df[ 'Z1' ] = np.concatenate( [ comps1[ cluster_id ][ 0 ], comps2[ cluster_id ][ 0 ] ] )
    df[ 'Z2' ] = np.concatenate( [ comps1[ cluster_id ][ 1 ], comps2[ cluster_id ][ 1 ] ] )
    return df


def plot_dynamics_kde( df, out_name ):
    sns.set( style='darkgrid' )
    sns.set_context( 'paper', rc={ 'font.size':20, 'axes.titlesize':20, 'axes.labelsize':20,
                                   'xtick.labelsize':20, 'ytick.labelsize':20 } )
    healthy = df.loc[ df.condition == 'H' ]
    pd      = df.loc[ df.condition == 'PD' ]
    g = sns.JointGrid( x = 'Z1', y = 'Z2', data = df )
    sns.kdeplot( healthy.Z1, healthy.Z2, cmap="Blues",
                 shade=False, shade_lowest=False, ax=g.ax_joint )
    sns.kdeplot( pd.Z1, pd.Z2, cmap="Reds",
                 shade=False, shade_lowest=False, ax=g.ax_joint )
    
    sns.distplot( healthy.Z1, kde=True, hist=False, color="r", ax=g.ax_marg_x )
    sns.distplot( pd.Z1, kde=True, hist=False, color="b", ax=g.ax_marg_x )
    sns.distplot( healthy.Z2, kde=True, hist=False, color="r", ax=g.ax_marg_y, vertical=True )
    sns.distplot( pd.Z2, kde=True, hist=False, color="b", ax=g.ax_marg_y, vertical=True )
    g.savefig( out_name )


def plot_dynamics( clusters_best ):
    comps1 = [ dynamics.get_spk_comps( x, pd=False  ) for x in clusters_best ]
    comps2 = [ dynamics.get_spk_comps( x, pd=True ) for x in clusters_best ]

    df = build_comps_dataframe( comps1, comps2, cluster_id=0 )
    plot_dynamics_kde( df, 'results/cluster1.pdf' )
    df = build_comps_dataframe( comps1, comps2, cluster_id=1 )
    plot_dynamics_kde( df, 'results/cluster2.pdf' )

    fig = make_subplots( rows=2, cols=2,
                         subplot_titles = ['Cluster 1', 'Cluster 2'],
                         specs=[[{'type':'scene'}]*2]*2,
                         vertical_spacing = 0,
                         horizontal_spacing = 0.05 )
    for i in range( len(comps1) ):
        trace = go.Scatter3d( x=comps1[i][0], y=comps1[i][1], z=comps1[i][2],
                              marker = {'color':'blue'},
                              line = {'color':'blue'},
                              name='Healthy',
                              showlegend= i==0 )
        fig.add_trace( trace, row=1+i//2, col=1+i%2 )
        trace = go.Scatter3d( x=comps2[i][0], y=comps2[i][1], z=comps2[i][2],
                              name='PD',
                              marker = {'color':'red'},
                              line = {'color':'red'},
                              showlegend= i==0 )
        fig.add_trace( trace, row=1+i//2, col=1+i%2 )
    fig.update_traces( marker={'size':5}, line={'width':7} )
    for scene in [fig.layout.scene, fig.layout.scene2, fig.layout.scene3, fig.layout.scene4]:
        scene.update( dict(
                xaxis = { 'range':[-0.2,0.2], 'tickvals':[-0.2,0,0.2], 'title':'Z<sub>1</sub>', 'title_font':{'size':16} },
                yaxis = { 'range':[-0.2,0.2], 'tickvals':[-0.2,0], 'title':'Z<sub>2</sub>', 'title_font':{'size':16} },
                zaxis = { 'range':[-0.2,0.2], 'tickvals':[0,0.2], 'title':'Z<sub>3</sub>', 'title_font':{'size':16} },
                aspectmode = 'cube' ) )
    fig.update_xaxes( tickfont = {'size':14} )
    fig.update_yaxes( tickfont = {'size':14} )
    fig.layout.update( 
        height = 1000,
        width  = 1000,
        margin = { 'pad':10 },
        font   = { 'size':14 },
        legend = { 'font':{'size': 16} } )
    fig.show()


def get_histogram( bests_genotypes ):
    bests = np.transpose( bests_genotypes, [1,0] )
    hists = list()
    for param in bests:
        hist = np.histogram( param, bins = np.linspace( min(param), max(param), 15 ) )
        hists.append( hist )
    return hists


def plot_params( bests_genotypes, fname='results/params.pdf' ):
    plt.clf()
    sns.set( style='darkgrid' )
    matplotlib.rcParams['xtick.labelsize']='large'
    matplotlib.rcParams['ytick.labelsize']='large'

    norm_bests_genotypes = np.array( [ transform_cand(x) for x in bests_genotypes ] )
    c1 = get_ordered_cluster( bests_genotypes, clusters, 0 )
    c2 = get_ordered_cluster( bests_genotypes, clusters, 1 )

    param_names = [ '$I_{TH}$',
                    '$I_{GPe}$',
                    '$I_{GPi}$',
                    '$g_{STN\_KCa}$',
                    '$g_{GP\_AHP}$',
                    '$g_{syn\_ctx\_str}$',
                    '$n_{GPe}$',
                    '$n_{GPi}$',
                    '$n_{TH}$',
                    '$n_{StrD1}$',
                    '$n_{StrD2}$',
                    '$n_{CTX\_RS}$',
                    '$n_{CTX\_FSI}$',
                    '$n_{STN}$']

    sns.set( style='darkgrid' )
    fig, axs = plt.subplots( nrows=2, ncols=7, figsize=(10,4) )
    for i in range(14):
        cluster_ids = np.array( [0]*len(c1) + [1]*len(c2) )
        data = np.concatenate( [ c1[:,i], c2[:,i] ] )
        data = np.array( [ data, cluster_ids ], dtype=np.float64 )
        data = np.transpose( data, [1,0] )
        df0 = pandas.DataFrame( data, columns=[ param_names[i], 'Cluster' ] )
        sns.violinplot( x='Cluster', y=param_names[i], ax=axs[i//7,i%7], data=df0 )
        axs[i//7, i%7].ticklabel_format( style='sci', scilimits=(0,30), axis='y' )
        axs[i//7, i%7].set_xticklabels( ['c1', 'c2'] )
        axs[i//7, i%7].set_xlabel('')
        axs[i//7, i%7].set_ylabel('')
        axs[i//7, i%7].set_title(param_names[i]+'\n')
        #data = [ c1[:,i], c2[:,i] ]
        #parts = axs[i//7, i%7].violinplot( [ c1[:,i], c2[:,i] ], showmedians=True )
    fig.tight_layout()
    fig.savefig( fname )
    plt.clf()


def plot_fitness_evo( bests, medians ):
    sns.set( style='darkgrid' )
    fig, axs = plt.subplots( nrows=1, ncols=2, figsize=(8,4), sharex=True, sharey=True )
    bests_mean = np.mean( bests, axis=0 )
    bests_std = np.std( bests, axis=0 )
    medians_mean = np.mean( medians, axis=0 )
    medians_std = np.std( medians, axis=0 )
    x = np.arange( len( bests_mean ) )
    axs[0].set_title( 'Best fitness', fontsize='large' )
    axs[0].set_xlabel( 'Generation', fontsize='large' )
    axs[0].set_ylabel( 'Fitness', fontsize='large' )
    axs[0].errorbar( x, bests_mean, yerr=bests_std )
    axs[0].tick_params( labelsize='large' )
    axs[1].set_title( 'Average fitness', fontsize='large' )
    axs[1].errorbar( x, medians_mean, yerr=medians_std )
    axs[1].tick_params( labelsize='large' )
    fig.tight_layout()
    plt.savefig( 'results/fitness.pdf' )


def plot_fitness_evo_box( bests ):
    plt.clf()
    sns.set( style='darkgrid' )
    bests = np.array( bests )
    df = pandas.DataFrame( bests )
    sns.boxplot( data = df )
    plt.xticks( [0, 41], ['1','40'] )
    plt.savefig( 'results/evo_box.pdf' )


def plot_fitness_dist( bests ):
    plt.clf()
    sns.set( style='darkgrid' )
    sns.distplot( bests )
    plt.xlabel( 'Fitness $f(M)$', fontsize='x-large' )
    plt.tick_params( labelsize='large' )
    plt.savefig( 'results/fitness_dist.pdf' )


def plot_fitness_evo_dist( bests, means ):
    plt.clf()
    sns.set( style='darkgrid' )
    fig, axs = plt.subplots( nrows=2, ncols=2, figsize=(10,8) )

    # Distplot - bests
    sns.distplot( bests, ax=axs[0,1] )
    axs[0,1].set_xlabel( 'Fitness $f(M)$' )
    axs[0,1].set_ylabel( 'Density' )
    # Distplot - means
    sns.distplot( means, ax=axs[1,1] )
    axs[1,1].set_xlabel( 'Fitness $f(M)$\n\n(b)' )
    axs[1,1].set_ylabel( 'Density' )

    # Boxplot - bests
    bests = np.array( bests )
    df = pandas.DataFrame( bests )
    sns.boxplot( ax=axs[0,0], data = df )
    axs[0,0].set_xticks( [0, 10, 20, 30, 40] )
    axs[0,0].set_xticklabels( ['0', '10', '20', '30', '40'] )
    axs[0,0].set_xlabel( 'Generation' )
    axs[0,0].set_ylabel( 'Fitness $f(M)$' )
    axs[0,0].set_title( 'Best individuals' )
    axs[0,1].set_title( 'Best individuals' )
    # Boxplot - means
    means = np.array( means )
    df = pandas.DataFrame( means )
    sns.boxplot( ax=axs[1,0], data = df )
    axs[1,0].set_xticks( [0, 10, 20, 30, 40] )
    axs[1,0].set_xticklabels( ['0', '10', '20', '30', '40'] )
    axs[1,0].set_xlabel( 'Generation\n\n(a)' )
    axs[1,0].set_ylabel( 'Fitness $f(M)$' )
    axs[1,0].set_title( 'Mean individuals' )
    axs[1,1].set_title( 'Mean individuals' )

    fig.tight_layout()
    plt.savefig( 'results/fitness_evo_dist.pdf' )


if __name__=='__main__':
    flag_genotypes = True
    flag_dynamics  = True
    flag_fitness   = True
    flag_params    = True
    flag_spectral  = True

    bests_genotypes, fits_bestg, bests, means = load_all_genotypes()

    if flag_genotypes:
        clusters_best, clusters = plot_all_genotypes( bests_genotypes, fits_bestg, 
                                                      use_default_clusters=False )
        plot_rat_genotype( bests_genotypes, clusters )
        if flag_spectral:
            psd_h, psd_pd, power_h, power_pd, lfp_f = simulate_all_clusters( bests_genotypes, clusters )
            make_plots( psd_h, psd_pd, power_h, power_pd, lfp_f )
        if flag_dynamics:
            clusters_best = [ get_ordered_cluster(bests_genotypes, clusters, i)[0] for i in range(2) ]
            plot_dynamics( clusters_best )
            dynamics.calc_all_comps()
            for i in range( 2 ):
                dynamics.calc_all_dtw( cluster_id=i )
            dynamics.plot_dists( n_clusters=2 )

    if flag_params:
        plot_params( bests_genotypes, 'results/params2.pdf' )
    if flag_fitness: 
        plot_fitness_evo_dist( bests, means )


        

