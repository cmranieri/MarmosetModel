# _ -*- coding: cp1252 -*-
import random
from netpyne import specs, sim
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import argrelmax
import pylab
from GA_params import GA_params


#TODO This is the Kumaravelu model class. Verify if all possible configurations are flexible, i.e., can be configured when a new object is instantiated

# Configuration 1
# region AP ML DV
# M1 10 6.5 14.4
# GPi 8 3.5 7.8
# GPe 8 5.2 8.8
# Put 8.5 6.5 11.5
# VL 5.5 3.7 10.5
# VPL 4.5 4.3 9.2
# STN 5.5 3.7 7.6

# Configuration 2
# M1 10 6.5 14.4
# S1 8 5.2 15.6
# Put 8.5 6.5 11.5
# VL 5.5 3.7 10.5
# VPL 4.5 4.3 9.2
# STN 5.5 3.7 7.6


###################### Health / Parkinson ###################
# RS-> StrD1 connections
# GPe-< GPe connections
# Str.mod
#############################################################

class Network:
    class Spikes:
        def __init__(self):
            self.times = []

    def __init__( self,
                  has_pd     = False,
                  dbs        = False,
                  t_sim      = 1000,
                  n_channels = 1,
                  seed       = 1 ):
        random.seed( seed )
        self.electrodesPos = [ [5000, 4900, 4000],  # StrD1
                               [5000, 4900, 4000],  # StrD2
                               [1000, 2600, 1800],  # TH
                               [4500, 1200, 1000],  # GPi
                               [4500, 2200, 2700],  # GPe
                               [6500, 7800, 4000],  # CtxRS
                               [6500, 7800, 4000],  # CtxFSI
                               [2000, 1200, 1200] ] # STN
        self.nelec = 1

        self.pd         = has_pd
        self.dbs        = dbs
        self.t_sim      = t_sim
        self.n_channels = n_channels

        self.netParams = self.buildNetParams()
        self.buildPopulationParameters()
        self.buildCellRules()
        self.buildSynMechParams()
        self.buildCellConnRules()
        self.buildStimParams()


    def set_genotype( self, genotype ):
        ga_params = GA_params()
        genotype  = ga_params.transform( genotype )
        self.buildPopulationParameters( n_gpe   = int( genotype[6] ),
                                        n_gpi   = int( genotype[7] ),
                                        n_th    = int( genotype[8] ),
                                        n_strd1 = int( genotype[9] ),
                                        n_strd2 = int( genotype[10] ),
                                        n_rs    = int( genotype[11] ),
                                        n_fsi   = int( genotype[12] ),
                                        n_stn   = int( genotype[13] ) )
        self.buildCellConnRules()
        #net.buildCellConnRules( stn_gpe = int( genotype[8] ),
        #                        gpe_gpe = int( genotype[9] ),
        #                        stn_gpi = int( genotype[10] ),
        #                        gpe_gpi = int( genotype[11] ),
        #                        strd2_strd2 = int( genotype[12] ),
        #                        strd1_strd1 = int( genotype[13] ),
        #                        rs_fsi = int( genotype[14] ),
        #                        fsi_rs = int( genotype[15] ) )
        self.buildStimParams( amp_th  = genotype[0],
                              amp_gpe = genotype[1],
                              amp_gpi = genotype[2])
        for ch in range( self.n_channels ):
            self.netParams.cellParams['STN_%d'%ch]['secs']['soma']['mechs']['STN']['gkcabar'] = genotype[3]
            self.netParams.cellParams['GPe_%d'%ch]['secs']['soma']['mechs']['GP']['gahp']     = genotype[4]
            self.netParams.cellParams['GPi_%d'%ch]['secs']['soma']['mechs']['GP']['gahp']     = genotype[4]
        self.strConnRules( gsynmod = genotype[5] )
        return



    def buildNetParams(self):
        return specs.NetParams()  # object of class NetParams to store the network parameters


    def buildPopulationParameters( self,
                                   n_strd1 = 10,
                                   n_strd2 = 10,
                                   n_th    = 10,
                                   n_gpi   = 10,
                                   n_gpe   = 10,
                                   n_rs    = 10,
                                   n_fsi   = 10,
                                   n_stn   = 10 ):

        self.netParams.sizeX = 7500  # x-dimension (horizontal length) size in um
        self.netParams.sizeY = 8800  # y-dimension (vertical height or cortical depth) size in um
        self.netParams.sizeZ = 5000  # z-dimension (horizontal length) size in um

        # volume occupied by each population can be customized (xRange, yRange and zRange) in um
        # xRange or xnormRange - Range of neuron positions in x-axis (horizontal length), specified 2-element list [min, max].
        # zRange or znormRange - Range of neuron positions in z-axis (horizontal depth)
        # establishing 2000 um as a standard coordinate span

        for ch in range( self.n_channels ):
            self.netParams.popParams['StrD1_%d'%ch] = {'cellModel': 'StrD1',
                                                 'cellType': 'StrD1',
                                                 'numCells': n_strd1,
                                                 'xRange': [4000, 6000],
                                                 'yRange': [3900, 5900],
                                                 'zRange': [3000, 5000]}
            self.netParams.popParams['StrD2_%d'%ch] = {'cellModel': 'StrD2',
                                                 'cellType': 'StrD2',
                                                 'numCells': n_strd2,
                                                 'xRange': [4000, 6000],
                                                 'yRange': [3900, 5900],
                                                 'zRange': [3000, 5000]}
            # considering VPL coordinates
            self.netParams.popParams['TH_%d'%ch] = {'cellModel': 'TH',
                                              'cellType': 'Thal',
                                              'numCells': n_th,
                                              'xRange': [0, 2000],
                                              'yRange': [1600, 3600],
                                              'zRange': [800, 2800]}
            self.netParams.popParams['GPi_%d'%ch] = {'cellModel': 'GPi',
                                               'cellType': 'GPi',
                                               'numCells': n_gpi,
                                               'xRange': [3500, 5500],
                                               'yRange': [200, 2200],
                                               'zRange': [0, 2000]}
            self.netParams.popParams['GPe_%d'%ch] = {'cellModel': 'GPe',
                                               'cellType': 'GPe',
                                               'numCells': n_gpe,
                                               'xRange': [3500, 5500],
                                               'yRange': [1200, 3200],
                                               'zRange': [1700, 3700]}
            # considering M1
            self.netParams.popParams['CTX_RS_%d'%ch] = {'cellModel': 'CTX_RS',
                                                  'cellType': 'CTX_RS',
                                                  'numCells': n_rs,
                                                  'xRange': [5500, 7500],
                                                  'yRange': [6800, 8800],
                                                  'zRange': [3000, 5000]}
            self.netParams.popParams['CTX_FSI_%d'%ch] = {'cellModel': 'CTX_FSI',
                                                   'cellType': 'CTX_FSI',
                                                   'numCells': n_fsi,
                                                   'xRange': [5500, 7500],
                                                   'yRange': [6800, 8800],
                                                   'zRange': [3000, 5000]}
            self.netParams.popParams['STN_%d'%ch] = {'cellModel': 'STN',
                                               'cellType': 'STN',
                                               'numCells': n_stn,
                                               'xRange': [1000, 3000],
                                               'yRange': [0, 2000],
                                               'zRange': [200, 2200]}

    def buildCellRules(self, **args):
        self.rsCellRules(**args)
        self.fsiCellRules(**args)
        self.strD1CellRules(**args)
        self.strD2CellRules(**args)
        self.thCellRules(**args)
        self.gpiCellRules(**args)
        self.gpeCellRules(**args)
        self.stnCellRules(**args)

    def rsCellRules(self):
        cellRule = {'conds': {'cellModel': 'CTX_RS', 'cellType': 'CTX_RS'}, 'secs': {}}
        cellRule['secs']['soma'] = {'geom': {}, 'pointps': {}}
        cellRule['secs']['soma']['geom'] = {'diam': 5.642,
                                            'L': 5.642,
                                            'Ra': 1,
                                            'nseg': 1,
                                            'cm': 1}
        cellRule['secs']['soma']['pointps']['Izhi'] = {'mod': 'Izhi2003b',
                                                       'a': 0.02,
                                                       'b': 0.2,
                                                       'c': -65,
                                                       'd': 8,
                                                       'f': 5,
                                                       'g': 140,
                                                       'thresh': 30}
        cellRule['secs']['soma']['vinit'] = -65
        cellRule['secs']['soma']['threshold'] = 30
        for ch in range( self.n_channels ):
            self.netParams.cellParams['CTX_RS_%d'%ch] = cellRule

    def fsiCellRules(self):
        cellRule = {'conds': {'cellModel': 'CTX_FSI', 'cellType': 'CTX_FSI'}, 'secs': {}}
        cellRule['secs']['soma'] = {'geom': {}, 'pointps': {}}
        cellRule['secs']['soma']['geom'] = {'diam': 5.642,
                                            'L': 5.642,
                                            'Ra': 1,
                                            'nseg': 1,
                                            'cm': 1}
        cellRule['secs']['soma']['pointps']['Izhi'] = {'mod': 'Izhi2003b',
                                                       'a': 0.1,
                                                       'b': 0.2,
                                                       'c': -65,
                                                       'd': 2,
                                                       'f': 5,
                                                       'g': 140,
                                                       'thresh': 30}
        cellRule['secs']['soma']['vinit'] = -65
        cellRule['secs']['soma']['threshold'] = 30
        for ch in range( self.n_channels ):
            self.netParams.cellParams['CTX_FSI_%d'%ch] = cellRule

    def strD1CellRules(self):
        cellRule = {'conds': {'cellModel': 'StrD1', 'cellType': 'StrD1'}, 'secs': {}}
        cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}
        cellRule['secs']['soma']['geom'] = {'diam': 5.642,
                                            'L': 5.642,
                                            'Ra': 1,
                                            'nseg': 1}
        cellRule['secs']['soma']['mechs']['Str'] = {'gmbar': (2.6e-3 - self.pd * 1.1e-3)}
        cellRule['secs']['soma']['vinit'] = random.gauss(-63.8, 5)
        cellRule['secs']['soma']['threshold'] = -10
        for ch in range( self.n_channels ):
            self.netParams.cellParams['StrD1_%d'%ch] = cellRule

    def strD2CellRules(self):
        cellRule = {'conds': {'cellModel': 'StrD2', 'cellType': 'StrD2'}, 'secs': {}}
        cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}
        cellRule['secs']['soma']['geom'] = {'diam': 5.642,
                                            'L': 5.642,
                                            'Ra': 1,
                                            'nseg': 1}
        cellRule['secs']['soma']['mechs']['Str'] = {'gmbar': (2.6e-3 - self.pd * 1.1e-3)}
        cellRule['secs']['soma']['vinit'] = random.gauss(-63.8, 5)
        cellRule['secs']['soma']['threshold'] = -10
        for ch in range( self.n_channels ):
            self.netParams.cellParams['StrD2_%d'%ch] = cellRule

    def thCellRules(self):
        cellRule = {'conds': {'cellModel': 'TH', 'cellType': 'Thal'}, 'secs': {}}
        cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}
        cellRule['secs']['soma']['geom'] = {'diam': 5.642,
                                            'L': 5.642,
                                            'Ra': 1,
                                            'nseg': 1}
        cellRule['secs']['soma']['mechs']['thalamus'] = {}
        cellRule['secs']['soma']['vinit'] = random.gauss(-62, 5)
        cellRule['secs']['soma']['threshold'] = -10
        for ch in range( self.n_channels ):
            self.netParams.cellParams['TH_%d'%ch] = cellRule

    def gpiCellRules(self, gahp=10e-3):
        cellRule = {'conds': {'cellModel': 'GPi', 'cellType': 'GPi'}, 'secs': {}}
        cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}
        cellRule['secs']['soma']['geom'] = {'diam': 5.642,
                                            'L': 5.642,
                                            'Ra': 1,
                                            'nseg': 1}
        cellRule['secs']['soma']['mechs']['GP'] = {'gahp': gahp}
        # cellRule['secs']['GPi']['mechs']['GP'] = {}
        cellRule['secs']['soma']['vinit'] = random.gauss(-62, 5)
        cellRule['secs']['soma']['threshold'] = -10
        for ch in range( self.n_channels ):
            self.netParams.cellParams['GPi_%d'%ch] = cellRule

    def gpeCellRules(self, gahp=10e-3):
        cellRule = {'conds': {'cellModel': 'GPe', 'cellType': 'GPe'}, 'secs': {}}
        cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}
        cellRule['secs']['soma']['geom'] = {'diam': 5.642,
                                            'L': 5.642,
                                            'Ra': 1,
                                            'nseg': 1}
        cellRule['secs']['soma']['mechs']['GP'] = {'gahp': gahp}
        # cellRule['secs']['GPe']['mechs']['GP'] = {}
        cellRule['secs']['soma']['vinit'] = random.gauss(-62, 5)
        cellRule['secs']['soma']['threshold'] = -10
        for ch in range( self.n_channels ):
            self.netParams.cellParams['GPe_%d'%ch] = cellRule

    def stnCellRules(self, gkcabar=1e-3):
        cellRule = {'conds': {'cellModel': 'STN', 'cellType': 'STN'}, 'secs': {}}
        cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}}
        cellRule['secs']['soma']['geom'] = {'diam': 5.642,
                                            'L': 5.642,
                                            'Ra': 1,
                                            'nseg': 1}
        cellRule['secs']['soma']['mechs']['STN'] = {'dbs': self.dbs,
                                                    'gkcabar': gkcabar}
        cellRule['secs']['soma']['vinit'] = random.gauss(-62, 5)
        cellRule['secs']['soma']['threshold'] = -10
        for ch in range( self.n_channels ):
            self.netParams.cellParams['STN_%d'%ch] = cellRule

    def buildSynMechParams(self):
        # TH
        self.netParams.synMechParams['Igith'] = {'mod': 'Exp2Syn',
                                                 'tau1': 5,
                                                 'tau2': 5,
                                                 'e': -85}  # gpi -<th
        # GPe
        self.netParams.synMechParams['Insge,ampa'] = {'mod': 'Exp2Syn',
                                                      'tau1': 0.4,
                                                      'tau2': 2.5,
                                                      'e': 0}  # stn -> gpe
        self.netParams.synMechParams['Insge,nmda'] = {'mod': 'Exp2Syn',
                                                      'tau1': 2,
                                                      'tau2': 67,
                                                      'e': 0}  # stn -> gpe
        self.netParams.synMechParams['Igege'] = {'mod': 'Exp2Syn',
                                                 'tau1': 5,
                                                 'tau2': 5,
                                                 'e': -85}  # gpe -< gpe
        self.netParams.synMechParams['Istrgpe'] = {'mod': 'Exp2Syn',
                                                   'tau1': 5,
                                                   'tau2': 5,
                                                   'e': -85}  # D2 -> gpe
        # GPi
        self.netParams.synMechParams['Igegi'] = {'mod': 'Exp2Syn',
                                                 'tau1': 5,
                                                 'tau2': 5,
                                                 'e': -85}  # gpe -< gp
        self.netParams.synMechParams['Isngi'] = {'mod': 'Exp2Syn',
                                                 'tau1': 5,
                                                 'tau2': 5,
                                                 'e': 0}  # stn -> gpi
        self.netParams.synMechParams['Istrgpi'] = {'mod': 'Exp2Syn',
                                                   'tau1': 5,
                                                   'tau2': 5,
                                                   'e': -85}  # D1 -> gpi
        # STN
        self.netParams.synMechParams['Igesn'] = {'mod': 'Exp2Syn',
                                                 'tau1': 0.4,
                                                 'tau2': 7.7,
                                                 'e': -85}  # gpe -< stn
        self.netParams.synMechParams['Icosn,ampa'] = {'mod': 'Exp2Syn',
                                                      'tau1': 0.5,
                                                      'tau2': 2.49,
                                                      'e': 0}  # ctx -> gpe
        self.netParams.synMechParams['Icosn,nmda'] = {'mod': 'Exp2Syn',
                                                      'tau1': 2,
                                                      'tau2': 90,
                                                      'e': 0}  # ctx -> gpe
        # Str
        self.netParams.synMechParams['Igabadr'] = {'mod': 'Exp2Syn',
                                                   'tau1': 0.1,
                                                   'tau2': 13,
                                                   'e': -80}  # str -< str
        self.netParams.synMechParams['Igabaindr'] = {'mod': 'Exp2Syn',
                                                     'tau1': 0.1,
                                                     'tau2': 13,
                                                     'e': -80}  # str -< str
        self.netParams.synMechParams['Icostr'] = {'mod': 'Exp2Syn',
                                                  'tau1': 5,
                                                  'tau2': 5,
                                                  'e': 0}  # ctx -> str
        # CTX
        self.netParams.synMechParams['Iei'] = {'mod': 'Exp2Syn',
                                               'tau1': 5,
                                               'tau2': 5,
                                               'e': 0}  # rs->fsi
        self.netParams.synMechParams['Iie'] = {'mod': 'Exp2Syn',
                                               'tau1': 5,
                                               'tau2': 5,
                                               'e': -85}  # fsi<-rs
        self.netParams.synMechParams['Ithco'] = {'mod': 'Exp2Syn',
                                                 'tau1': 5,
                                                 'tau2': 5,
                                                 'e': 0}  # th->rs

    def buildCellConnRules(self, **args):
        self.thConnRules(**args)
        self.gpeConnRules(**args)
        self.gpiConnRules(**args)
        self.stnConnRules(**args)
        self.strConnRules(**args)
        self.ctxConnRules(**args)

    def thConnRules(self, **args):
        # GPi-> Th connections
        n_th  = self.netParams.popParams['TH_0']['numCells']
        n_gpi = self.netParams.popParams['GPi_0']['numCells']
        n_neurons = max( n_th, n_gpi )
        for ch in range( self.n_channels ):
            self.netParams.connParams['GPi->th_%d'%ch] = {
                'preConds': {'pop': 'GPi_%d'%ch}, 'postConds': {'pop': 'TH_%d'%ch},  # GPi-> th
                'connList': [[i%n_gpi, i%n_th] for i in range(n_neurons)],
                'weight': 0.0336e-3,  # synaptic weight (conductance)
                'delay': 5,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Igith'}  # target synaptic mechanism

    def gpeConnRules(self,
                     stn_gpe=2,
                     gpe_gpe=2,
                     **args):
        # STN->GPe connections
        # Two aleatory GPe cells (index i) receive synapse from cells i and i - 1
        n_stn = self.netParams.popParams['STN_0']['numCells']
        n_gpe = self.netParams.popParams['GPe_0']['numCells']
        n_neurons = max( n_stn, n_gpe )
        aux = random.sample(range(n_neurons), stn_gpe)
        connList = [[(x - c)%n_stn, x%n_gpe] for x in aux for c in [1, 0]]
        weight = [random.uniform(0, 0.3) * 0.43e-3 for k in range(len(connList))]
        for ch in range( self.n_channels ):
            self.netParams.connParams['STN->GPe_%d'%ch] = {
                'preConds': {'pop': 'STN_%d'%ch}, 'postConds': {'pop': 'GPe_%d'%ch},  # STN-> GPe
                'connList': connList,  # AMPA
                'weight': weight,  # synaptic weight (conductance)
                'delay': 2,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Insge,ampa'}  # target synaptic mechanism
        # STN->GPe connections
        # Two aleatory GPe cells (index i) receive synapse from cells i and i - 1
        aux = random.sample(range(n_neurons), stn_gpe)
        connList = [[(x - c)%n_stn, x%n_gpe] for x in aux for c in [1, 0]]
        weight = [random.uniform(0, 0.002) * 0.43e-3 for k in range(len(connList))]
        for ch in range( self.n_channels ):
            self.netParams.connParams['STN->GPe2_%d'%ch] = {
                'preConds': {'pop': 'STN_%d'%ch}, 'postConds': {'pop': 'GPe_%d'%ch},  # STN-> GPe
                'connList': connList,  # NMDA
                'weight': weight,  # synaptic weight (conductance)
                'delay': 2,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Insge,nmda'}  # target synaptic mechanism

        # GPe-< GPe connections
        n_neurons = self.netParams.popParams['GPe_0']['numCells']
        connList = [[(idx + ncn) % n_neurons, idx] for ncn in range(1, gpe_gpe + 1, 2)
                    for idx in range(n_neurons)] + \
                   [[idx, (idx + ncn) % n_neurons] for ncn in range(2, gpe_gpe + 1, 2)
                    for idx in range(n_neurons)]
        # connList = [[2,1],[3,2],[4,3],[5,4],[6,5],[7,6],[8,7],[9,8],[0,9],[1,0],
        #            [8,0],[9,1],[0,2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,8],[7,9]]
        weight = [(0.25 + 0.75 * self.pd) * random.uniform(0, 1) * 0.3e-3 \
                  for k in range(len(connList))]
        for ch in range( self.n_channels ):
            self.netParams.connParams['GPe->GPe_%d'%ch] = {
                'preConds': {'pop': 'GPe_%d'%ch}, 'postConds': {'pop': 'GPe_%d'%ch},  # GPe-< GPe
                'connList': connList,
                'weight': weight,  # synaptic weight (conductance)
                'delay': 1,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Igege'}  # target synaptic mechanism

        # StrD2>GPe connections
        n_strd2 = self.netParams.popParams['StrD2_0']['numCells']
        n_gpe   = self.netParams.popParams['GPe_0']['numCells']
        for ch in range( self.n_channels ):
            self.netParams.connParams['StrD2->GPe_%d'%ch] = {
                'preConds': {'pop': 'StrD2_%d'%ch}, 'postConds': {'pop': 'GPe_%d'%ch},  # StrD2-> GPe
                'connList': [[j, i] for i in range(n_gpe)
                                    for j in range(n_strd2)],
                'weight': 0.15e-3,  # synaptic weight (conductance)
                'delay': 5,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Istrgpe'}  # target synaptic mechanism

    def gpiConnRules(self,
                     stn_gpi=5,
                     gpe_gpi=2,
                     **args):
        # STN-> GPi connections
        # Five aleatory GPi cells (index i) receive synapse from cells i and i - 1
        n_stn = self.netParams.popParams['STN_0']['numCells']
        n_gpi = self.netParams.popParams['GPi_0']['numCells']
        n_neurons = max( n_stn, n_gpi )
        aux = random.sample(range(n_neurons), stn_gpi)

        # PSTH
        self.gsngi = np.zeros(10)
        for k in range(0,10):
            if (k == aux[0] or k == aux[1] or k == aux[2] or k == aux[3] or k == aux[4]):
                self.gsngi[k] = 1
            else:
                self.gsngi[k] = 0

        connList = [[(x - c)%n_stn, x%n_gpi] for x in aux for c in [1, 0]]
        for ch in range( self.n_channels ):
            self.netParams.connParams['STN->GPi_%d'%ch] = {
                'preConds': {'pop': 'STN_%d'%ch}, 'postConds': {'pop': 'GPi_%d'%ch},
                'connList': connList,
                'weight': 0.0645e-3,  # synaptic weight (conductance)
                'delay': 1.5,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Isngi'}  # target synaptic mechanism

        # GPe-< GPi connections 
        n_gpe = self.netParams.popParams['GPe_0']['numCells']
        n_gpi = self.netParams.popParams['GPi_0']['numCells']
        n_neurons = max( n_gpe, n_gpi )
        for ch in range( self.n_channels ):
            self.netParams.connParams['GPe->GPi_%d'%ch] = {
                'preConds': {'pop': 'GPe_%d'%ch}, 'postConds': {'pop': 'GPi_%d'%ch},
                'connList':
                    [[idx%n_gpe, (idx + ncn) % n_gpi] for ncn in range(2, gpe_gpi + 1, 2)
                     for idx in range(n_neurons)] + \
                    [[(idx + ncn) % n_gpe, idx%n_gpi] for ncn in range(1, gpe_gpi + 1, 2)
                     for idx in range(n_neurons)],
                # [ [ idx, (idx+2) % n_neurons ] for idx in range( n_neurons ) ] + \
                # [ [ (idx+1) % n_neurons, idx ] for idx in range( n_neurons ) ],
                'weight': 0.15e-3,  # synaptic weight (conductance)
                'delay': 3,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Igegi'}  # target synaptic mechanism

        # StrD1>GPi connections
        n_strd1 = self.netParams.popParams['StrD1_0']['numCells']
        n_gpi   = self.netParams.popParams['GPi_0']['numCells']
        for ch in range( self.n_channels ):
            self.netParams.connParams['StrD1->GPe_%d'%ch] = {
                'preConds': {'pop': 'StrD1_%d'%ch}, 'postConds': {'pop': 'GPi_%d'%ch},  # StrD1-> GPi
                'connList': [[j, i] for i in range(n_gpi)
                             for j in range(n_strd1)],
                'weight': 0.15e-3,  # synaptic weight (conductance)
                'delay': 4,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Istrgpi'}  # target synaptic mechanism

    def stnConnRules(self, **args):
        # GPe-> STN connections 
        n_gpe = self.netParams.popParams['GPe_0']['numCells']
        n_stn = self.netParams.popParams['STN_0']['numCells']
        n_neurons = max( n_gpe, n_stn )
        for ch in range( self.n_channels ):
            self.netParams.connParams['GPe->STN_%d'%ch] = {
                'preConds': {'pop': 'GPe_%d'%ch}, 'postConds': {'pop': 'STN_%d'%ch},  # GPe-< STN
                'connList': [[(i + c) % n_gpe, i%n_stn] for c in [1, 0] for i in range(n_neurons)],
                'weight': 0.15e-3,  # synaptic weight (conductance)
                'delay': 4,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Igesn'}  # target synaptic mechanism

        # CTX-> STN connections
        n_ctxrs = self.netParams.popParams['CTX_RS_0']['numCells']
        n_stn   = self.netParams.popParams['STN_0']['numCells']
        n_neurons = max( n_ctxrs, n_stn )
        connList = [[(i + c) % n_ctxrs, i%n_stn] for c in [1, 0] for i in range(n_neurons)]
        weight = [random.uniform(0, 0.3) * 0.43e-3 for k in range(len(connList))]
        for ch in range( self.n_channels ):
            self.netParams.connParams['CTX->STN_%d'%ch] = {
                'preConds': {'pop': 'CTX_RS_%d'%ch}, 'postConds': {'pop': 'STN_%d'%ch},  # CTX-> STN
                'connList': connList,
                'weight': weight,  # synaptic weight (conductance)
                'delay': 5.9,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Icosn,ampa'}  # target synaptic mechanism
        # CTX-> STN2 
        connList = [[(i + c) % n_ctxrs, i%n_stn] for c in [1, 0] for i in range(n_neurons)]
        weight = [random.uniform(0, 0.003) * 0.43e-3 for k in range(len(connList))]
        for ch in range( self.n_channels ):
            self.netParams.connParams['CTX->STN2_%d'%ch] = {
                'preConds': {'pop': 'CTX_RS_%d'%ch}, 'postConds': {'pop': 'STN_%d'%ch},  # CTX-> STN
                'connList': connList,
                'weight': weight,  # synaptic weight (conductance)
                'delay': 5.9,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Icosn,nmda'}  # target synaptic mechanism

    def strConnRules(self,
                     strd2_strd2=4,
                     strd1_strd1=3,
                     gsynmod=1,
                     **args):
        # StrD2-< StrD2 connections
        # Each StrD2 cell receive synapse from 4 aleatory StrD2 cell (except from itself)
        n_neurons = self.netParams.popParams['StrD2_0']['numCells']
        connList = [[x, i] for i in range(n_neurons)
                           for x in random.sample([k for k in range(n_neurons) if k != i],
                                                  strd2_strd2)]
        for ch in range( self.n_channels ):
            self.netParams.connParams['StrD2->StrD2_%d'%ch] = {
                'preConds': {'pop': 'StrD2_%d'%ch}, 'postConds': {'pop': 'StrD2_%d'%ch},  # StrD2-< StrD2
                'connList': connList,
                'weight': 0.1 / 4 * 0.5e-3,  # synaptic weight (conductance) -> mudar essa maluquisse
                'delay': 0,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Igabaindr'}  # target synaptic mechanism

        # StrD1-< StrD1 connections
        # Each StrD1 cell receive synapse from 3 aleatory StrD1 cell (except from itself)
        n_neurons = self.netParams.popParams['StrD1_0']['numCells']
        connList = [[x, i] for i in range(n_neurons)
                           for x in random.sample([k for k in range(n_neurons) if k != i],
                                                  strd1_strd1)]
        for ch in range( self.n_channels ):
            self.netParams.connParams['StrD1->StrD1_%d'%ch] = {
                'preConds': {'pop': 'StrD1_%d'%ch}, 'postConds': {'pop': 'StrD1_%d'%ch},  # StrD1-< StrD1
                'connList': connList,
                'weight': 0.1 / 3 * 0.5e-3,  # synaptic weight (conductance) -> mudar aqui tb
                'delay': 0,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Igabadr'}  # target synaptic mechanism

        # RS-> StrD1 connections 
        n_ctxrs = self.netParams.popParams['CTX_RS_0']['numCells']
        n_strd1 = self.netParams.popParams['StrD1_0']['numCells']
        n_neurons = max( n_ctxrs, n_strd1 )
        for ch in range( self.n_channels ):
            self.netParams.connParams['RS->StrD1_%d'%ch] = {
                'preConds': {'pop': 'CTX_RS_%d'%ch}, 'postConds': {'pop': 'StrD1_%d'%ch},  # RS-> StrD1
                'connList': [[i%n_ctxrs, i%n_strd1] for i in range(n_neurons)],
                'weight': (0.07 - 0.044 * self.pd) * 0.43e-3 * gsynmod,  # synaptic weight (conductance)
                'delay': 5.1,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Icostr'}  # target synaptic mechanism

        # RS-> StrD2 connections 
        n_ctxrs = self.netParams.popParams['CTX_RS_0']['numCells']
        n_strd2 = self.netParams.popParams['StrD2_0']['numCells']
        n_neurons = max( n_ctxrs, n_strd2 )
        for ch in range( self.n_channels ):
            self.netParams.connParams['RS->StrD2_%d'%ch] = {
                'preConds': {'pop': 'CTX_RS_%d'%ch}, 'postConds': {'pop': 'StrD2_%d'%ch},  # RS-> StrD2 
                'connList': [[i%n_ctxrs, i%n_strd2] for i in range(n_neurons)],
                'weight': 0.07 * 0.43e-3 * gsynmod,  # synaptic weight (conductance)
                'delay': 5.1,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Icostr'}  # target synaptic mechanism

    def ctxConnRules(self,
                     rs_fsi=4,
                     fsi_rs=4,
                     **args):
        # RS -> FSI connections
        # Each FSI cell receive synapse from 4 aleatory RS cells
        n_rs  = self.netParams.popParams['CTX_RS_0']['numCells']
        n_fsi = self.netParams.popParams['CTX_FSI_0']['numCells']
        connList = [[x, i] for i in range(n_fsi)
                    for x in random.sample([k for k in range(n_rs) if k != i],
                                           rs_fsi)]
        for ch in range( self.n_channels ):
            self.netParams.connParams['ctx_rs->ctx_fsi_%d'%ch] = {
                'preConds': {'pop': 'CTX_RS_%d'%ch}, 'postConds': {'pop': 'CTX_FSI_%d'%ch},  # ctx_rs -> ctx_fsi
                'connList': connList,
                'weight': 0.043e-3,  # synaptic weight (conductance)
                'delay': 1,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Iei'}  # target synaptic mechanism

        # FSI -> RS connections
        # Each RS cell receive synapse from 4 aleatory FSI cells
        connList = [[x, i] for i in range(n_rs)
                    for x in random.sample([k for k in range(n_fsi) if k != i],
                                           fsi_rs)]
        for ch in range( self.n_channels ):
            self.netParams.connParams['ctx_fsi->ctx_rs_%d'%ch] = {
                'preConds': {'pop': 'CTX_FSI_%d'%ch}, 'postConds': {'pop': 'CTX_RS_%d'%ch},  # ctx_fsi -< ctx_rs
                'connList': connList,
                'weight': 0.083e-3,  # synaptic weight (conductance)
                'delay': 1,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Iie'}  # target synaptic mechanism

        # Th -> RS connections
        n_th    = self.netParams.popParams['TH_0']['numCells']
        n_ctxrs = self.netParams.popParams['CTX_RS_0']['numCells']
        n_neurons = max( n_th, n_ctxrs )
        for ch in range( self.n_channels ):
            self.netParams.connParams['th->ctx_rs_%d'%ch] = {
                'preConds': {'pop': 'TH_%d'%ch}, 'postConds': {'pop': 'CTX_RS_%d'%ch},  # th -> ctx_rs
                'connList': [[i%n_th, i%n_ctxrs] for i in range(n_neurons)],
                'weight': 0.0645e-3,  # synaptic weight (conductance)
                'delay': 5,  # transmission delay (ms)
                'loc': 1,  # location of synapse
                'synMech': 'Ithco'}  # target synaptic mechanism

    def buildStimParams(self,
                        amp_th=1.2e-3, amp_gpe=3e-3,
                        amp_gpi=3e-3, amp_stn=0,
                        amp_fs=0, amp_rs=0,
                        amp_dstr=0, amp_istr=0):
        bin_fs = 0;
        bin_rs = 0;
        bin_gpe = 0;
        bin_gpi = 0;
        bin_stn = 0;
        bin_dstr = 0;
        bin_istr = 0;
        bin_th = 0;

        for ch in range( self.n_channels ):
            # FS receve a constante 3 density current or 1 during cortical stimulation
            self.netParams.stimSourceParams['Input_FS_%d'%ch] = {'type': 'IClamp',
                                                           'delay': 0,
                                                           'dur': self.t_sim,
                                                           'amp': bin_fs * -1}
            self.netParams.stimTargetParams['Input_FS->FS_%d'%ch] = {'source': 'Input_FS_%d'%ch,
                                                               'conds': {'pop': 'CTX_FSI_%d'%ch},
                                                               'sec': 'soma',
                                                               'loc': 0}

            # RS receve a constante 3 density current or 1 during cortical stimulation
            self.netParams.stimSourceParams['Input_RS_%d'%ch] = {'type': 'IClamp',
                                                           'delay': 0,
                                                           'dur': self.t_sim,
                                                           'amp': bin_rs * -1 + amp_rs}
            self.netParams.stimTargetParams['Input_RS->RS_%d'%ch] = {'source': 'Input_RS_%d'%ch,
                                                               'conds': {'pop': 'CTX_RS_%d'%ch},
                                                               'sec': 'soma',
                                                               'loc': 0}

            # GPe receve a constante 3 density current or 1 during cortical stimulation
            self.netParams.stimSourceParams['Input_GPe_%d'%ch] = {'type': 'IClamp',
                                                            'delay': 0,
                                                            'dur': self.t_sim,
                                                            'amp': bin_gpe * -1 + amp_gpe}
            self.netParams.stimTargetParams['Input_GPe->GPe_%d'%ch] = {'source': 'Input_GPe_%d'%ch,
                                                                 'conds': {'pop': 'GPe_%d'%ch},
                                                                 'sec': 'soma',
                                                                 'loc': 0}

            # GPi receve a constante 3 density current
            self.netParams.stimSourceParams['Input_GPi_%d'%ch] = {'type': 'IClamp',
                                                            'delay': 0, 'dur': self.t_sim,
                                                            'amp': bin_gpi * -1 + amp_gpi}
            self.netParams.stimTargetParams['Input_GPi->GPi_%d'%ch] = {'source': 'Input_GPi_%d'%ch,
                                                                 'conds': {'pop': 'GPi_%d'%ch},
                                                                 'sec': 'soma',
                                                                 'loc': 0}

            # STN receve a constante 3 density current or 1 during cortical stimulation
            self.netParams.stimSourceParams['Input_STN_%d'%ch] = {'type': 'IClamp',
                                                            'delay': 0,
                                                            'dur': self.t_sim,
                                                            'amp': bin_stn * -1 + amp_stn}
            self.netParams.stimTargetParams['Input_STN->STN_%d'%ch] = {'source': 'Input_STN_%d'%ch,
                                                                 'conds': {'pop': 'STN_%d'%ch},
                                                                 'sec': 'soma',
                                                                 'loc': 0}

            # dStr receve a constante 3 density current
            self.netParams.stimSourceParams['Input_StrD1_%d'%ch] = {'type': 'IClamp',
                                                              'delay': 0,
                                                              'dur': self.t_sim,
                                                              'amp': bin_dstr * -1 + amp_dstr}
            self.netParams.stimTargetParams['Input_StrD1->StrD1_%d'%ch] = {'source': 'Input_StrD1_%d'%ch,
                                                                     'conds': {'pop': 'StrD1_%d'%ch},
                                                                     'sec': 'soma',
                                                                     'loc': 0}

            # iStr receve a constante 3 density current
            self.netParams.stimSourceParams['Input_StrD2_%d'%ch] = {'type': 'IClamp',
                                                              'delay': 0, 'dur': self.t_sim,
                                                              'amp': bin_istr * -1 + amp_istr}
            self.netParams.stimTargetParams['Input_StrD2->StrD2_%d'%ch] = {'source': 'Input_StrD2_%d'%ch,
                                                                     'conds': {'pop': 'StrD2_%d'%ch},
                                                                     'sec': 'soma',
                                                                     'loc': 0}

            # Thalamus receve a constante 1.2 density current
            self.netParams.stimSourceParams['Input_th_%d'%ch] = {'type': 'IClamp',
                                                           'delay': 0,
                                                           'dur': self.t_sim,
                                                           'amp': bin_th * -1 + amp_th}
            self.netParams.stimTargetParams['Input_th->TH_%d'%ch] = {'source': 'Input_th_%d'%ch,
                                                               'conds': {'pop': 'TH_%d'%ch},
                                                               'sec': 'soma',
                                                               'loc': 0}

    def extractLFP_SP(self):
        lfp = sim.allSimData['LFP']
        # [ f, t ]
        lfp = np.transpose(lfp, [1, 0])

        # calculate LFP using Welch method
        lfp_f, lfp_dimensions = signal.welch( lfp[0], 1000, nperseg=1024, detrend=False )
        lfp_fft = np.zeros(( len(self.electrodesPos)//self.nelec, lfp_dimensions.shape[0] ))
        for i in range( 0, lfp.shape[0], self.nelec ):
            reg_fft = list()
            for j in range( self.nelec ):
                reg_fft.append( signal.welch( lfp[i+j], 1000, nperseg=1024, detrend=False ) )
            lfp_f, lfp_fft[i//self.nelec, :] = np.mean( reg_fft, axis=0 )
        return lfp_f, lfp_fft


    def extractLFP_raw(self):
        lfp = sim.allSimData['LFP']
        # [ f, t ]
        lfp = np.transpose(lfp, [1, 0])
        return lfp


    def extractSpikes(self):
        spikes = self.Spikes
        spk_dict = dict()
        n_strd1 = self.netParams.popParams['StrD1_0']['numCells']
        n_strd2 = self.netParams.popParams['StrD2_0']['numCells'] 
        n_th  = self.netParams.popParams['TH_0']['numCells'] 
        n_gpi = self.netParams.popParams['GPi_0']['numCells'] 
        n_gpe = self.netParams.popParams['GPe_0']['numCells'] 
        n_cor_rs  = self.netParams.popParams['CTX_RS_0']['numCells'] 
        n_cor_fsi = self.netParams.popParams['CTX_FSI_0']['numCells'] 
        n_stn = self.netParams.popParams['STN_0']['numCells'] 

        c_strd1    = n_strd1
        c_strd2    = c_strd1   + n_strd2
        c_th       = c_strd2   + n_th
        c_gpi      = c_th      + n_gpi
        c_gpe      = c_gpi     + n_gpe
        c_cor_rs   = c_gpe     + n_cor_rs
        c_cor_fsi  = c_cor_rs  + n_cor_fsi
        c_stn      = c_cor_fsi + n_stn
        
        for ch in range( self.n_channels ):
            spk_dict['dStr_APs_%d'%ch]    = [spikes() for k in range( n_strd1 )]
            spk_dict['iStr_APs_%d'%ch]    = [spikes() for k in range( n_strd2 )]
            spk_dict['TH_APs_%d'%ch]      = [spikes() for k in range( n_th    )]
            spk_dict['GPi_APs_%d'%ch]     = [spikes() for k in range( n_gpi   )]
            spk_dict['GPe_APs_%d'%ch]     = [spikes() for k in range( n_gpe   )]
            spk_dict['Cor_RS_APs_%d'%ch]  = [spikes() for k in range( n_cor_rs  )] 
            spk_dict['Cor_FSI_APs_%d'%ch] = [spikes() for k in range( n_cor_fsi )]
            spk_dict['STN_APs_%d'%ch]     = [spikes() for k in range( n_stn     )]

        for ch in range( self.n_channels ):
            for i in range( len(sim.allSimData['spkt']) ):
                strd1_a = ch * c_stn          
                strd2_a = ch * c_stn + c_strd1   
                th_a    = ch * c_stn + c_strd2   
                gpi_a   = ch * c_stn + c_th      
                gpe_a   = ch * c_stn + c_gpi     
                rs_a    = ch * c_stn + c_gpe     
                fsi_a   = ch * c_stn + c_cor_rs  
                stn_a   = ch * c_stn + c_cor_fsi 
                strd1_b = strd1_a + n_strd1           
                strd2_b = strd2_a + n_strd2
                th_b    = th_a    + n_th
                gpi_b   = gpi_a   + n_gpi
                gpe_b   = gpe_a   + n_gpe
                rs_b    = rs_a    + n_cor_rs
                fsi_b   = fsi_a   + n_cor_fsi
                stn_b   = stn_a   + n_stn

                if ( sim.allSimData['spkid'][i] >= strd1_a and sim.allSimData['spkid'][i] < strd1_b ):
                    spk_dict['dStr_APs_%d'%ch][int(sim.allSimData['spkid'][i] - strd1_a)].times += [sim.allSimData['spkt'][i]]
                
                elif ( sim.allSimData['spkid'][i] >= strd2_a and sim.allSimData['spkid'][i] < strd2_b ):
                    spk_dict['iStr_APs_%d'%ch][int(sim.allSimData['spkid'][i] - strd2_a)].times += [sim.allSimData['spkt'][i]]

                elif ( sim.allSimData['spkid'][i] >= th_a and sim.allSimData['spkid'][i] < th_b ):
                    spk_dict['TH_APs_%d'%ch][int(sim.allSimData['spkid'][i] - th_a)].times += [sim.allSimData['spkt'][i]]
               
                elif ( sim.allSimData['spkid'][i] >= gpi_a and sim.allSimData['spkid'][i] < gpi_b ):
                    spk_dict['GPi_APs_%d'%ch][int(sim.allSimData['spkid'][i] - gpi_a)].times += [sim.allSimData['spkt'][i]]
                
                elif ( sim.allSimData['spkid'][i] >= gpe_a and sim.allSimData['spkid'][i] < gpe_b ):
                    spk_dict['GPe_APs_%d'%ch][int(sim.allSimData['spkid'][i] - gpe_a)].times += [sim.allSimData['spkt'][i]]

                elif ( sim.allSimData['spkid'][i] >= rs_a and sim.allSimData['spkid'][i] < rs_b ):
                    spk_dict['Cor_RS_APs_%d'%ch][int(sim.allSimData['spkid'][i] - rs_a)].times += [sim.allSimData['spkt'][i]]

                elif ( sim.allSimData['spkid'][i] >= fsi_a and sim.allSimData['spkid'][i] < fsi_b ):
                    spk_dict['Cor_FSI_APs_%d'%ch][int(sim.allSimData['spkid'][i] - fsi_a)].times += [sim.allSimData['spkt'][i]]

                elif ( sim.allSimData['spkid'][i] >= stn_a and sim.allSimData['spkid'][i] < stn_b ):
                    spk_dict['STN_APs_%d'%ch][int(sim.allSimData['spkid'][i] - stn_a)].times += [sim.allSimData['spkt'][i]]

        return spk_dict


    def extractMFR(self):
        mfr = [sim.allSimData.popRates['CTX_FSI_0'],
               sim.allSimData.popRates['CTX_RS_0'],
               sim.allSimData.popRates['GPe_0'],
               sim.allSimData.popRates['GPi_0'],
               sim.allSimData.popRates['STN_0'],
               sim.allSimData.popRates['StrD1_0'],
               sim.allSimData.popRates['StrD2_0'],
               sim.allSimData.popRates['TH_0']]
        for i in range(0, 8):
            mfr[i] = round(mfr[i], 2)
        return mfr


    def get_gsngi(self):
        return self.gsngi


    def buildSimConfig(self, dt=0.1, lfp=False, recordStep=1, seeds=None):
        # Simulation parameters
        simConfig = specs.SimConfig()
        simConfig.duration = self.t_sim  # Duration of the simulation, in ms
        simConfig.dt = dt  # Internal integration timestep to use
        simConfig.verbose = False  # Show detailed messages
        simConfig.printPopAvgRates = True
        if seeds is not None:
            simConfig.seeds = seeds

        # Recording
        print( 'recordStep', recordStep )
        simConfig.recordStep = recordStep  # Step size in ms to save data (eg. V traces, LFP, etc)
        simConfig.recordCells = ['allCells']
        simConfig.recordSpikesGids = True

        # lfp and plot
        if lfp:
            simConfig.analysis['plotRaster'] = True
            simConfig.recordLFP = self.electrodesPos 
            simConfig.saveLFPCells = True
            simConfig.analysis['plotLFP'] = {'electrodes': ['all'],
                                              'includeAxon': False,
                                              #'timeRange': [0, 2000],
                                              'plots': ['timeSeries', 'locations', 'PSD'],
                                              # 'plots': ['locations'],
                                              'showFig': True}
        return simConfig


    def simulate(self, dt=0.1, lfp=False, seeds=None):
        simConfig = self.buildSimConfig(dt=dt, lfp=lfp, seeds=seeds)
        sim.createSimulateAnalyze(netParams=self.netParams, simConfig=simConfig)
        # out_dict = self.extractSpikes()
        # print( sim.allSimData['LFPCells'] )
        return sim


if __name__ == '__main__':
    network = Network( n_channels=1, seed=1 )
    network.simulate()
