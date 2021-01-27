import os, re
import numpy as np
import scipy.io as sio
import quantities as pq
from collections import defaultdict
from neo.core import Block, Segment, ChannelIndex, Unit, SpikeTrain, AnalogSignal
from neo.io import NeoMatlabIO, PickleIO

REGEX_HEMISPHERES    = '(\D+)?\d.*([LR])'
REGEX_NO_HEMISPHERES = '(\D+)\d' 
SPK_ONLY = False
PICKLE   = True
MATLAB   = False
#T_STOP = 4000


def create_block( name ):
    block = Block( name = name )
    spk_dict = defaultdict( list )
    lfp_dict = defaultdict( list )
    return block, ( spk_dict , lfp_dict )


def _area_from_match( match, hemispheres = True ):
    if hemispheres:
        return '_'.join( match.groups() ).upper().replace( 'O' , '' )
    else:
        return match.groups()[0].upper()


def belongs_to( field_name, area_name ):
    pattern = re.compile( REGEX_HEMISPHERES )
    match   = pattern.match( field_name )
    if match is not None:
        true_area_name = _area_from_match( match, True )
    else:
        pattern = re.compile( REGEX_NO_HEMISPHERES )
        true_area_name = _area_from_match( pattern.match( field_name ), False )
    return area_name == true_area_name


def _create_areas_list( names ):
    areas_list = list()
    for name in names:
        pattern = re.compile( REGEX_HEMISPHERES )
        match = pattern.match( name )
        if match is not None:
            areas_list.append( _area_from_match( match, True ) )
        else:
            pattern = re.compile( REGEX_NO_HEMISPHERES )
            areas_list.append( _area_from_match( pattern.match( name ), False ) )
    return np.array( areas_list )


def _read_new_spikes( data ):
    neurons     = [ neuron for neuron in data['neurons'][0] if len( neuron.dtype ) != 0 ]
    neurons     = [ neurons[i] for i in range( len(neurons) )
                               if len( neurons[i]['timestamps'] ) != 0 ]
    spikes      = [ neuron['timestamps'][0,0][0] for neuron in neurons ]
    field_names = [ neuron['name'][0,0][0] for neuron in neurons ]
    return spikes, field_names


def _read_bia_spikes( data ):
    spikes = data[ 'spikes' ][ 0,0 ]
    field_names = spikes.dtype.fields.keys()
    return spikes, field_names


def _read_bia_lfp( data ):
    samples = list()
    for sample in data[ 'LFP' ]:
        samples.append( np.array( sample[0] ) )
    lfp_array = np.array( samples )

    field_names = np.empty( ( len( lfp_array ) ), dtype=list )
    for i, name in enumerate( data[ 'areaNames' ][0] ):
        for ch_index in np.array( data[ 'channelsLeft' ] [ 0, i ] ):
            field_names[ ch_index - 1 ] = str( name[0] ) + '_L'
        for ch_index in np.array( data[ 'channelsRight' ][ 0, i ] ):
            field_names[ ch_index - 1 ] = str( name[0] ) + '_R'
    return samples, field_names



def _read_spikes_1( data ):
    field_names = [key for key in data.keys() if not key.startswith('__')]
    spikes = list()
    for key in field_names:
        spk = np.array( data[ key ] )
        spikes.append( spk )
    spikes = np.array( spikes )
    return spikes, field_names


def _read_spikes_2( data ):
    spike_ts    = data[ 'data' ][ 0,0 ][ 'spike_ts' ]
    spikes      = [ spk for spk in spike_ts[ : , 0 ] ]
    field_names = [ name[ 0 ].replace( '\x00' , '' )
                    for name in spike_ts[ : , 1 ] ]
    return spikes, field_names


def _read_lfp_2( data ):
    lfp_samples = data[ 'data' ][ 0,0 ][ 'lfp_samples' ]
    samples     = [ sample[ :,0 ] for sample in lfp_samples[ :,0 ] ]
    field_names = [ name[0][0] for name in data[ 'data' ][ 0,0 ][ 'lfp_id' ] ]
    return samples, field_names


def _read_lfp_ed( data ):
    field_names = [ key for key in data.keys()
                    if not re.match('.*_|Start|Stop|AllFile|FP64', key) ]
    samples     = [ data[key] for key in field_names ]
    field_names = [ re.sub('VL\D', 'VL', name) for name in field_names ]
    return samples, field_names



def convert_spikes( segment, spikes_data ):
    spikes, spike_names, spk_dict = spikes_data
    for spk_seq, spk_name in zip( spikes, spike_names ):
        spk = spk_seq.squeeze()
        if spk.shape == (): spk = np.array( [ spk ] )
        spk_train = SpikeTrain( times  = spk,
                                t_stop = np.max( spk_seq ),
                                units  = 's',
                                name   = spk_name.upper() )
        segment.spiketrains.append ( spk_train )
        spk_dict[ spk_name.upper() ].append( spk_train )
    return segment, spk_dict


def fix_lfp_timesteps( lfp_dict ):
    lengths = [ len(lfp_dict[key]) for key in lfp_dict ]
    keys    = [ key for key in lfp_dict ]
    print(lengths)
    print(keys)
    print(len(lengths), len(keys))
    while max(lengths) != min(lengths):
        del lfp_dict[ keys[ np.argmax( lengths ) ] ]
        print( 'removed signal' )
    return lfp_dict


def convert_lfp( segment, lfp_data ):
    lfp_samples, lfp_names, lfp_dict = lfp_data
    for lfp_sample, lfp_name in zip( lfp_samples, lfp_names ):
        analog_signal = AnalogSignal( signal = lfp_sample,
                                      units = 'V',
                                      sampling_rate = pq.Quantity( 1000, 'Hz' ),
                                      name = 'LFP_' + lfp_name.upper() )
        segment.analogsignals.append( analog_signal )
        lfp_dict[ lfp_name.upper() ].append( analog_signal )
    return segment, lfp_dict


def _create_segment( block, seg_name, spikes_data, lfp_data ):
    spikes, spike_names, spk_dict = spikes_data
    lfp_samples, lfp_names, lfp_dict = lfp_data
    segment = Segment( name = seg_name )
    if spikes != None:
        segment, spk_dict = convert_spikes( segment, spikes_data )
    if lfp_samples != None:
        segment, lfp_dict = convert_lfp( segment, lfp_data )
    block.segments.append( segment )
    return block, segment, ( spk_dict , lfp_dict )


def _create_channel_indexes( block, spk_dict ):
    area_names_list = _create_areas_list( spk_dict.keys() )
    for area_name in set( area_names_list ):
        index = np.where( area_names_list == area_name )
        ch = ChannelIndex( index, name = area_name )

        for field_name in spk_dict.keys():
            if belongs_to( field_name, area_name ):
                unit = Unit( name = field_name )
                for spk_train in spk_dict[ field_name ]:
                    unit.spiketrains.append( spk_train )
                ch.units.append( unit )
        block.channel_indexes.append( ch )
    return ch


def read_spikes_from_data( data, flag ):
    if flag in [ 'bia' ]:
        spikes, field_names = _read_bia_spikes( data )
    elif flag in [ 'deco_1' , 'kadu1' ]:
        spikes, field_names = _read_spikes_1( data )
    elif flag in [ 'deco_2', 'dede' , 'pele', 'zeca_1' ]:
        spikes, field_names = _read_spikes_2( data )
    elif flag in [ 'paty', 'zeca_2', 'ed', 'kadu2' ]:
        spikes, field_names = ( None , None )
    elif flag in ['new']:
        spikes, field_names = _read_new_spikes( data )
    return spikes, field_names


def read_lfp_from_data( data, flag, spk_only = False ):
    if spk_only or flag in [ 'deco_1', 'kadu1', 'new' ]:
        samples, field_names = ( None , None )
    elif flag in [ 'bia' ]:
        samples, field_names = _read_bia_lfp( data )
    elif flag in [ 'deco_2', 'dede' , 'pele', 'paty', 'zeca_1', 'zeca_2' ]:
        samples, field_names = _read_lfp_2( data )
    elif flag in [ 'ed', 'kadu2' ]:
        samples, field_names = _read_lfp_ed( data )
    return samples, field_names


def _process_generic( block, dataset_paths, flag, dicts ):
    spk_dict, lfp_dict = dicts
    for path in dataset_paths:
        path1, path2 = ( path, None )
        if isinstance( path, tuple ):
            path1, path2 = path
        filename = path1.split( '/' )[ -1 ]
        print( 'Processing %s...' % filename )
        print(path1)
        data_spk = sio.loadmat( path1 )
        if path2 is not None:
            data_lfp = sio.loadmat( path2 )
        else:
            data_lfp = data_spk
        spikes, spike_names    = read_spikes_from_data( data_spk, flag )
        lfp_samples, lfp_names = read_lfp_from_data( data_lfp, flag, SPK_ONLY )
        spikes_data = ( spikes, spike_names, spk_dict )
        lfp_data    = ( lfp_samples, lfp_names, lfp_dict )
        block, segment, dicts = _create_segment( block       = block,
                                                 seg_name    = filename,
                                                 spikes_data = spikes_data,
                                                 lfp_data    = lfp_data )
        spk_dict, lfp_dict = dicts
    #if spikes != None:
    #    ch = _create_channel_indexes( block, spk_dict )
    return block 


def process_ed():
    step = 1
    blocks = list()
    dataset_paths = [ os.path.join( '../datasets/Ed',d,x )
                      for d in  os.listdir( '../datasets/Ed' )
                      if  os.path.isdir( os.path.join( '../datasets/Ed', d ) )
                      for x in os.listdir( os.path.join('../datasets/Ed',d) ) ]
    for i in range( len(dataset_paths) // 3 ):
        block, dicts = create_block( 'block_ed_%d' % i )
        paths = dataset_paths[ i*step: i*step+step ]
        block = _process_generic( block, paths, 'ed', dicts )
        save_block( block, 'ed_%d' % i )
    return None


def process_bia():
    block, dicts = create_block( 'block_bia' )
    dataset_paths = [ '../datasets/Bia/dataset_Bia_%d.mat' % (i+1) 
                      for i in range(7) ]
    block = _process_generic( block, dataset_paths, 'bia', dicts )
    return block


def process_deco():
    block, dicts = create_block( 'block_deco' )
    dataset_paths_1 = [ '../datasets/Deco/Deco_20120928_mrg_spike.mat',
                        ( '../datasets/Deco/Deco_20121029_baseline_spike.mat',
                          '../datasets/Deco/Deco_20121029_baseline_LFP.mat' ) ]
    base_path = '../datasets/Deco/Deco_08112012'
    dataset_paths_2 = [ os.path.join( base_path, filename )
                        for filename in os.listdir( base_path ) ]
    block = _process_generic( block, dataset_paths_1, 'deco_1', dicts )
    block = _process_generic( block, dataset_paths_2, 'deco_2', dicts )
    return block


def process_kadu():
    block, dicts = create_block( 'block_kadu' )
    dataset_paths = [ '../datasets/Kadu/Kadu_20120928_spike.mat' ]
    block = _process_generic( block, dataset_paths, 'kadu1', dicts )
    dataset_paths = [ '../datasets/Kadu/Kadu_20120928_LFP.mat',
                      '../datasets/Kadu/Kadu_20121102_LFP.mat' ]
    block = _process_generic( block, dataset_paths, 'kadu2', dicts )
    return block


def process_dede():
    block, dicts = create_block( 'block_dede_0' )
    dataset_paths = [ '../datasets/Dede_28062012/DATA_Dede_0%d.mat' % (i+1) 
                      for i in range(9) ]
    block = _process_generic( block, dataset_paths, 'dede', dicts )
    save_block( block, 'dede_0' )

    block, dicts = create_block( 'block_dede_1' )
    dataset_paths = [ '../datasets/DATA_Dede_23052012_Box.mat' ]
    dataset_paths += [ '../datasets/DATA_Dede_06062012.mat' ]
    block = _process_generic( block, dataset_paths, 'dede', dicts )
    save_block( block, 'dede_1' )
    return None


def process_dede_new():
    block, dicts  = create_block( 'block_dede' )
    dataset_paths = [ '../datasets/sorting/neurons/cvt/Dede_06062012_neurons_edited.mat',
                      '../datasets/sorting/neurons/cvt/Dede_23052012_Box_neurons_edited.mat',
                      '../datasets/sorting/neurons/cvt/Dede_28062012_01_neurons_edited.mat' ]
    block = _process_generic( block, dataset_paths, 'new', dicts )
    return block


def process_pele_new():
    block, dicts  = create_block( 'block_pele' )
    dataset_paths = [ '../datasets/sorting/neurons/cvt/Pele_05042013_neurons_edited.mat',
                      '../datasets/sorting/neurons/cvt/Pele_11042013_Baseline_neurons_edited.mat',
                      '../datasets/sorting/neurons/cvt/Pele_27022013_neurons_edited.mat',
                      '../datasets/sorting/neurons/cvt/Pele_baseline_09042013_neurons_edited.mat' ]
    block = _process_generic( block, dataset_paths, 'new', dicts )
    return block


def process_zeca_new():
    block, dicts = create_block( 'block_zeca' )
    dataset_paths = [ '../datasets/sorting/neurons/cvt/Zeca_11042013_Baseline_neurons_edited.mat' ]
    block = _process_generic( block, dataset_paths, 'new', dicts )
    return block


def process_pele_1():
    block, spk_dict = create_block( 'block_pele_1' )
    dataset_paths = list()
    dataset_paths.append( '../datasets/Pele_09042013/DATA_Pele_baseline_09042013.mat' )
    dataset_paths.append( '../datasets/Pele_09042013/DATA_Pele_SCS_09042013.mat' )
    dataset_paths.append( '../datasets/Pele_09042013/DATA_Pele_SCS_09042013b.mat' )
    dataset_paths.append( '../datasets/Pele_09042013/DATA_Pele_SCS_09042013c.mat' )
    block = _process_generic( block, dataset_paths, 'pele', spk_dict )
    return block


def process_pele_2():
    block, spk_dict = create_block( 'block_pele_2' )
    dataset_paths = list()
    dataset_paths.append( '../datasets/Pele_11042013/DATA_Pele_11042013_Baseline.mat' )
    dataset_paths.append( '../datasets/Pele_11042013/DATA_Pele_11042013_SCS.mat' )
    dataset_paths.append( '../datasets/Pele_11042013/DATA_Pele_11042013_SCS_Mashmallow.mat' )
    block = _process_generic( block, dataset_paths, 'pele', spk_dict )
    return block



def process_pele_3():
    block, spk_dict = create_block( 'block_pele_3' )
    dataset_paths = list()
    dataset_paths.append( '../datasets/Pele_20042013/DATA_Pele_baseline_6HAfterAMPT.mat' )
    #dataset_paths.append( '../datasets/Pele_20042013/DATA_Pele_baseline_6HAfterAMPT_WithMarshmallow.mat' )
    #dataset_paths.append( '../datasets/Pele_20042013/DATA_Pele_6HAfterAMPT_SCS.mat' )
    block = _process_generic( block, dataset_paths, 'pele', spk_dict )
    return block


def process_pele_4():
    block, spk_dict = create_block( 'block_pele_3' )
    dataset_paths = list()
    dataset_paths.append( '../datasets/DATA_Pele_05042013.mat' )
    dataset_paths.append( '../datasets/DATA_Pele_27022013.mat' )
    block = _process_generic( block, dataset_paths, 'pele', spk_dict )
    return block


def process_paty():
    block, spk_dict = create_block( 'block_paty' )
    dataset_paths = [ '../datasets/DATA_Paty_20150703.mat' ]
    block = _process_generic( block, dataset_paths, 'paty', spk_dict )
    return block


def process_zeca():
    block, spk_dict = create_block( 'block_zeca' )
    #dataset_paths_1 = list()
    dataset_paths_2 = list()
    #dataset_paths_1.append( '../datasets/Zeca/DATA_Zeca_11042013_Baseline.mat' )
    #dataset_paths_1.append( '../datasets/Zeca/DATA_Zeca_11042013_SCS.mat' )
    #dataset_paths_2.append( '../datasets/Zeca/DATA_Zeca_05042013.mat' )
    dataset_paths_2.append( '../datasets/Zeca/DATA_FP_Zeca_24042013_Baseline.mat' )
    #block = _process_generic( block, dataset_paths_1, 'zeca_1', spk_dict )
    block = _process_generic( block, dataset_paths_2, 'zeca_2', spk_dict )
    return block



def save_block( block, name ):
    out_dir = '../datasets/'
    if PICKLE:
        pickle_io = PickleIO( os.path.join( out_dir, 'python/%s.pickle' % name ) )
        pickle_io.write_block( block )
    if MATLAB:
        neomatlab_io = NeoMatlabIO( os.path.join( out_dir, 'matlab/%s.mat' % name ) )
        neomatlab_io.write_block( block )



if __name__ == '__main__':
    """
    block = process_bia()
    save_block( block, 'bia' )
    block = process_deco()
    save_block( block, 'deco' )
   
    block = process_kadu()
    save_block( block, 'kadu' )
    
    process_dede()
    
    block = process_pele_1()
    save_block( block, 'pele1' )
    block = process_pele_2()
    save_block( block, 'pele2' )
    """
    block = process_pele_3()
    save_block( block, 'pele3' )
    """
    block = process_pele_4()
    save_block( block, 'pele4' )
    
    block = process_paty()
    save_block( block, 'paty' )
    """

    block = process_zeca()
    save_block( block, 'zeca' )
    #process_ed()
    #block = process_dede_new()
    #save_block( block, 'dede_new' )
    #block = process_pele_new()
    #save_block( block, 'pele_new' )
    #block = process_zeca_new()
    #save_block( block, 'zeca_new' )
