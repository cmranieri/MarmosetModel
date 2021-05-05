# _ -*- coding: cp1252 -*-
import random
from netpyne import specs, sim
from scipy import signal
from scipy.signal import firwin, lfilter
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import argrelmax
import pylab


#random.seed( 2 )

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


electrodesPos = [ [5000, 4900, 4000],  # StrD1
                  [5000, 4900, 4000],  # StrD2
                  [1000, 2600, 1800],  # TH
                  [4500, 1200, 1000],  # GPi
                  [4500, 2200, 2700],  # GPe
                  [6500, 7800, 4000],  # CtxRS
                  [6500, 7800, 4000],  # CtxFSI
                  [2000, 1200, 1200] ] # STN

nelec = 1
#electrodesPos = [ *list( np.random.normal( [5000, 4900, 4000], 500, (nelec,3) ) ),  # StrD1
#                  *list( np.random.normal( [5000, 4900, 4000], 500, (nelec,3) ) ),  # StrD2
#                  *list( np.random.normal( [1000, 2600, 1800], 500, (nelec,3) ) ),  # TH
#                  *list( np.random.normal( [4500, 1200, 1000], 500, (nelec,3) ) ),  # GPi
#                  *list( np.random.normal( [4500, 2200, 2700], 500, (nelec,3) ) ),  # GPe
#                  *list( np.random.normal( [6500, 7800, 4000], 500, (nelec,3) ) ),  # CtxRS
#                  *list( np.random.normal( [6500, 7800, 4000], 500, (nelec,3) ) ),  # CtxFSI
#                  *list( np.random.normal( [2000, 1200, 1200], 500, (nelec,3) ) ) ] # STN

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
                  has_pd=0,
                  it_num=1,
                  dbs=0,
                  t_sim=1000,
                  n_neurons=10,
                  seed = 1 ):
        random.seed( seed )
        self.pd = has_pd
        self.it_num = it_num
        self.dbs = dbs
        self.t_sim = t_sim

        self.netParams = self.buildNetParams()
        self.buildPopulationParameters()
        self.buildCellRules()
        self.buildSynMechParams()
        self.buildCellConnRules()
        self.buildStimParams()


    def buildNetParams(self):
        return specs.NetParams()  # object of class NetParams to store the network parameters


    def buildPopulationParameters(self,
                                  n_strd1=10, n_strd2=10,
                                  n_th=10, n_gpi=10,
                                  n_gpe=10, n_rs=10,
                                  n_fsi=10, n_stn=10):

        self.netParams.sizeX = 7500  # x-dimension (horizontal length) size in um
        self.netParams.sizeY = 8800  # y-dimension (vertical height or cortical depth) size in um
        self.netParams.sizeZ = 5000  # z-dimension (horizontal length) size in um

        # volume occupied by each population can be customized (xRange, yRange and zRange) in um
        # xRange or xnormRange - Range of neuron positions in x-axis (horizontal length), specified 2-element list [min, max].
        # zRange or znormRange - Range of neuron positions in z-axis (horizontal depth)
        # establishing 2000 um as a standard coordinate span

        self.netParams.popParams['StrD1'] = {'cellModel': 'StrD1',
                                             'cellType': 'StrD1',
                                             'numCells': n_strd1,
                                             'xRange': [4000, 6000],
                                             'yRange': [3900, 5900],
                                             'zRange': [3000, 5000]}
        self.netParams.popParams['StrD2'] = {'cellModel': 'StrD2',
                                             'cellType': 'StrD2',
                                             'numCells': n_strd2,
                                             'xRange': [4000, 6000],
                                             'yRange': [3900, 5900],
                                             'zRange': [3000, 5000]}
        # considering VPL coordinates
        self.netParams.popParams['TH'] = {'cellModel': 'TH',
                                          'cellType': 'Thal',
                                          'numCells': n_th,
                                          'xRange': [0, 2000],
                                          'yRange': [1600, 3600],
                                          'zRange': [800, 2800]}
        self.netParams.popParams['GPi'] = {'cellModel': 'GPi',
                                           'cellType': 'GPi',
                                           'numCells': n_gpi,
                                           'xRange': [3500, 5500],
                                           'yRange': [200, 2200],
                                           'zRange': [0, 2000]}
        self.netParams.popParams['GPe'] = {'cellModel': 'GPe',
                                           'cellType': 'GPe',
                                           'numCells': n_gpe,
                                           'xRange': [3500, 5500],
                                           'yRange': [1200, 3200],
                                           'zRange': [1700, 3700]}
        # considering M1
        self.netParams.popParams['CTX_RS'] = {'cellModel': 'CTX_RS',
                                              'cellType': 'CTX_RS',
                                              'numCells': n_rs,
                                              'xRange': [5500, 7500],
                                              'yRange': [6800, 8800],
                                              'zRange': [3000, 5000]}
        self.netParams.popParams['CTX_FSI'] = {'cellModel': 'CTX_FSI',
                                               'cellType': 'CTX_FSI',
                                               'numCells': n_fsi,
                                               'xRange': [5500, 7500],
                                               'yRange': [6800, 8800],
                                               'zRange': [3000, 5000]}
        self.netParams.popParams['STN'] = {'cellModel': 'STN',
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
        self.netParams.cellParams['CTX_RS'] = cellRule

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
        self.netParams.cellParams['CTX_FSI'] = cellRule

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
        self.netParams.cellParams['StrD1'] = cellRule

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
        self.netParams.cellParams['StrD2'] = cellRule

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
        self.netParams.cellParams['TH'] = cellRule

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
        self.netParams.cellParams['GPi'] = cellRule

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
        self.netParams.cellParams['GPe'] = cellRule

    def stnCellRules(self, gkcabar=5e-3):
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
        self.netParams.cellParams['STN'] = cellRule

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
        n_th = self.netParams.popParams['TH']['numCells']
        n_gpi = self.netParams.popParams['GPi']['numCells']
        n_neurons = max( n_th, n_gpi )
        self.netParams.connParams['GPi->th'] = {
            'preConds': {'pop': 'GPi'}, 'postConds': {'pop': 'TH'},  # GPi-> th
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
        n_stn = self.netParams.popParams['STN']['numCells']
        n_gpe = self.netParams.popParams['GPe']['numCells']
        n_neurons = max( n_stn, n_gpe )
        aux = random.sample(range(n_neurons), stn_gpe)
        connList = [[(x - c)%n_stn, x%n_gpe] for x in aux for c in [1, 0]]
        weight = [random.uniform(0, 0.3) * 0.43e-3 for k in range(len(connList))]
        self.netParams.connParams['STN->GPe'] = {
            'preConds': {'pop': 'STN'}, 'postConds': {'pop': 'GPe'},  # STN-> GPe
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
        self.netParams.connParams['STN->GPe2'] = {
            'preConds': {'pop': 'STN'}, 'postConds': {'pop': 'GPe'},  # STN-> GPe
            'connList': connList,  # NMDA
            'weight': weight,  # synaptic weight (conductance)
            'delay': 2,  # transmission delay (ms)
            'loc': 1,  # location of synapse
            'synMech': 'Insge,nmda'}  # target synaptic mechanism

        # GPe-< GPe connections
        n_neurons = self.netParams.popParams['GPe']['numCells']
        connList = [[(idx + ncn) % n_neurons, idx] for ncn in range(1, gpe_gpe + 1, 2)
                    for idx in range(n_neurons)] + \
                   [[idx, (idx + ncn) % n_neurons] for ncn in range(2, gpe_gpe + 1, 2)
                    for idx in range(n_neurons)]
        # connList = [[2,1],[3,2],[4,3],[5,4],[6,5],[7,6],[8,7],[9,8],[0,9],[1,0],
        #            [8,0],[9,1],[0,2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,8],[7,9]]
        weight = [(0.25 + 0.75 * self.pd) * random.uniform(0, 1) * 0.3e-3 \
                  for k in range(len(connList))]
        self.netParams.connParams['GPe->GPe'] = {
            'preConds': {'pop': 'GPe'}, 'postConds': {'pop': 'GPe'},  # GPe-< GPe
            'connList': connList,
            'weight': weight,  # synaptic weight (conductance)
            'delay': 1,  # transmission delay (ms)
            'loc': 1,  # location of synapse
            'synMech': 'Igege'}  # target synaptic mechanism

        # StrD2>GPe connections
        n_strd2 = self.netParams.popParams['StrD2']['numCells']
        n_gpe = self.netParams.popParams['GPe']['numCells']
        self.netParams.connParams['StrD2->GPe'] = {
            'preConds': {'pop': 'StrD2'}, 'postConds': {'pop': 'GPe'},  # StrD2-> GPe
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
        n_stn = self.netParams.popParams['STN']['numCells']
        n_gpi = self.netParams.popParams['GPi']['numCells']
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
        self.netParams.connParams['STN->GPi'] = {
            'preConds': {'pop': 'STN'}, 'postConds': {'pop': 'GPi'},
            'connList': connList,
            'weight': 0.0645e-3,  # synaptic weight (conductance)
            'delay': 1.5,  # transmission delay (ms)
            'loc': 1,  # location of synapse
            'synMech': 'Isngi'}  # target synaptic mechanism

        # GPe-< GPi connections 
        n_gpe = self.netParams.popParams['GPe']['numCells']
        n_gpi = self.netParams.popParams['GPi']['numCells']
        n_neurons = max( n_gpe, n_gpi )
        self.netParams.connParams['GPe->GPi'] = {
            'preConds': {'pop': 'GPe'}, 'postConds': {'pop': 'GPi'},
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
        n_strd1 = self.netParams.popParams['StrD1']['numCells']
        n_gpi = self.netParams.popParams['GPi']['numCells']
        self.netParams.connParams['StrD1->GPe'] = {
            'preConds': {'pop': 'StrD1'}, 'postConds': {'pop': 'GPi'},  # StrD1-> GPi
            'connList': [[j, i] for i in range(n_gpi)
                         for j in range(n_strd1)],
            'weight': 0.15e-3,  # synaptic weight (conductance)
            'delay': 4,  # transmission delay (ms)
            'loc': 1,  # location of synapse
            'synMech': 'Istrgpi'}  # target synaptic mechanism

    def stnConnRules(self, **args):
        # GPe-> STN connections 
        n_gpe = self.netParams.popParams['GPe']['numCells']
        n_stn = self.netParams.popParams['STN']['numCells']
        n_neurons = max( n_gpe, n_stn )
        self.netParams.connParams['GPe->STN'] = {
            'preConds': {'pop': 'GPe'}, 'postConds': {'pop': 'STN'},  # GPe-< STN
            'connList': [[(i + c) % n_gpe, i%n_stn] for c in [1, 0] for i in range(n_neurons)],
            'weight': 0.15e-3,  # synaptic weight (conductance)
            'delay': 4,  # transmission delay (ms)
            'loc': 1,  # location of synapse
            'synMech': 'Igesn'}  # target synaptic mechanism

        # CTX-> STN connections
        n_ctxrs = self.netParams.popParams['CTX_RS']['numCells']
        n_stn = self.netParams.popParams['STN']['numCells']
        n_neurons = max( n_ctxrs, n_stn )
        connList = [[(i + c) % n_ctxrs, i%n_stn] for c in [1, 0] for i in range(n_neurons)]
        weight = [random.uniform(0, 0.3) * 0.43e-3 for k in range(len(connList))]
        self.netParams.connParams['CTX->STN'] = {
            'preConds': {'pop': 'CTX_RS'}, 'postConds': {'pop': 'STN'},  # CTX-> STN
            'connList': connList,
            'weight': weight,  # synaptic weight (conductance)
            'delay': 5.9,  # transmission delay (ms)
            'loc': 1,  # location of synapse
            'synMech': 'Icosn,ampa'}  # target synaptic mechanism
        # CTX-> STN2 
        connList = [[(i + c) % n_ctxrs, i%n_stn] for c in [1, 0] for i in range(n_neurons)]
        weight = [random.uniform(0, 0.003) * 0.43e-3 for k in range(len(connList))]
        self.netParams.connParams['CTX->STN2'] = {
            'preConds': {'pop': 'CTX_RS'}, 'postConds': {'pop': 'STN'},  # CTX-> STN
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
        n_neurons = self.netParams.popParams['StrD2']['numCells']
        connList = [[x, i] for i in range(n_neurons)
                           for x in random.sample([k for k in range(n_neurons) if k != i],
                                                  strd2_strd2)]
        self.netParams.connParams['StrD2-> StrD2'] = {
            'preConds': {'pop': 'StrD2'}, 'postConds': {'pop': 'StrD2'},  # StrD2-< StrD2
            'connList': connList,
            'weight': 0.1 / 4 * 0.5e-3,  # synaptic weight (conductance) -> mudar essa maluquisse
            'delay': 0,  # transmission delay (ms)
            'loc': 1,  # location of synapse
            'synMech': 'Igabaindr'}  # target synaptic mechanism

        # StrD1-< StrD1 connections
        # Each StrD1 cell receive synapse from 3 aleatory StrD1 cell (except from itself)
        n_neurons = self.netParams.popParams['StrD1']['numCells']
        connList = [[x, i] for i in range(n_neurons)
                           for x in random.sample([k for k in range(n_neurons) if k != i],
                                                  strd1_strd1)]
        self.netParams.connParams['StrD1-> StrD1'] = {
            'preConds': {'pop': 'StrD1'}, 'postConds': {'pop': 'StrD1'},  # StrD1-< StrD1
            'connList': connList,
            'weight': 0.1 / 3 * 0.5e-3,  # synaptic weight (conductance) -> mudar aqui tb
            'delay': 0,  # transmission delay (ms)
            'loc': 1,  # location of synapse
            'synMech': 'Igabadr'}  # target synaptic mechanism

        # RS-> StrD1 connections 
        n_ctxrs = self.netParams.popParams['CTX_RS']['numCells']
        n_strd1 = self.netParams.popParams['StrD1']['numCells']
        n_neurons = max( n_ctxrs, n_strd1 )
        self.netParams.connParams['RS->StrD1'] = {
            'preConds': {'pop': 'CTX_RS'}, 'postConds': {'pop': 'StrD1'},  # RS-> StrD1
            'connList': [[i%n_ctxrs, i%n_strd1] for i in range(n_neurons)],
            'weight': (0.07 - 0.044 * self.pd) * 0.43e-3 * gsynmod,  # synaptic weight (conductance)
            'delay': 5.1,  # transmission delay (ms)
            'loc': 1,  # location of synapse
            'synMech': 'Icostr'}  # target synaptic mechanism

        # RS-> StrD2 connections 
        n_ctxrs = self.netParams.popParams['CTX_RS']['numCells']
        n_strd2 = self.netParams.popParams['StrD2']['numCells']
        n_neurons = max( n_ctxrs, n_strd2 )
        self.netParams.connParams['RS->StrD2'] = {
            'preConds': {'pop': 'CTX_RS'}, 'postConds': {'pop': 'StrD2'},  # RS-> StrD2 
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
        n_rs = self.netParams.popParams['CTX_RS']['numCells']
        n_fsi = self.netParams.popParams['CTX_FSI']['numCells']
        connList = [[x, i] for i in range(n_fsi)
                    for x in random.sample([k for k in range(n_rs) if k != i],
                                           rs_fsi)]
        self.netParams.connParams['ctx_rs->ctx_fsi'] = {
            'preConds': {'pop': 'CTX_RS'}, 'postConds': {'pop': 'CTX_FSI'},  # ctx_rs -> ctx_fsi
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
        self.netParams.connParams['ctx_fsi->ctx_rs'] = {
            'preConds': {'pop': 'CTX_FSI'}, 'postConds': {'pop': 'CTX_RS'},  # ctx_fsi -< ctx_rs
            'connList': connList,
            'weight': 0.083e-3,  # synaptic weight (conductance)
            'delay': 1,  # transmission delay (ms)
            'loc': 1,  # location of synapse
            'synMech': 'Iie'}  # target synaptic mechanism

        # Th -> RS connections
        n_th = self.netParams.popParams['TH']['numCells']
        n_ctxrs = self.netParams.popParams['CTX_RS']['numCells']
        n_neurons = max( n_th, n_ctxrs )
        self.netParams.connParams['th->ctx_rs'] = {
            'preConds': {'pop': 'TH'}, 'postConds': {'pop': 'CTX_RS'},  # th -> ctx_rs
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

        # FS receve a constante 3 density current or 1 during cortical stimulation
        self.netParams.stimSourceParams['Input_FS'] = {'type': 'IClamp',
                                                       'delay': 0,
                                                       'dur': self.t_sim,
                                                       'amp': bin_fs * -1}
        self.netParams.stimTargetParams['Input_FS->FS'] = {'source': 'Input_FS',
                                                           'conds': {'pop': 'CTX_FSI'},
                                                           'sec': 'soma',
                                                           'loc': 0}

        # RS receve a constante 3 density current or 1 during cortical stimulation
        self.netParams.stimSourceParams['Input_RS'] = {'type': 'IClamp',
                                                       'delay': 0,
                                                       'dur': self.t_sim,
                                                       'amp': bin_rs * -1 + amp_rs}
        self.netParams.stimTargetParams['Input_RS->RS'] = {'source': 'Input_RS',
                                                           'conds': {'pop': 'CTX_RS'},
                                                           'sec': 'soma',
                                                           'loc': 0}

        # GPe receve a constante 3 density current or 1 during cortical stimulation
        self.netParams.stimSourceParams['Input_GPe'] = {'type': 'IClamp',
                                                        'delay': 0,
                                                        'dur': self.t_sim,
                                                        'amp': bin_gpe * -1 + amp_gpe}
        self.netParams.stimTargetParams['Input_GPe->GPe'] = {'source': 'Input_GPe',
                                                             'conds': {'pop': 'GPe'},
                                                             'sec': 'soma',
                                                             'loc': 0}

        # GPi receve a constante 3 density current
        self.netParams.stimSourceParams['Input_GPi'] = {'type': 'IClamp',
                                                        'delay': 0, 'dur': self.t_sim,
                                                        'amp': bin_gpi * -1 + amp_gpi}
        self.netParams.stimTargetParams['Input_GPi->GPi'] = {'source': 'Input_GPi',
                                                             'conds': {'pop': 'GPi'},
                                                             'sec': 'soma',
                                                             'loc': 0}

        # STN receve a constante 3 density current or 1 during cortical stimulation
        self.netParams.stimSourceParams['Input_STN'] = {'type': 'IClamp',
                                                        'delay': 0,
                                                        'dur': self.t_sim,
                                                        'amp': bin_stn * -1 + amp_stn}
        self.netParams.stimTargetParams['Input_STN->STN'] = {'source': 'Input_STN',
                                                             'conds': {'pop': 'STN'},
                                                             'sec': 'soma',
                                                             'loc': 0}

        # dStr receve a constante 3 density current
        self.netParams.stimSourceParams['Input_StrD1'] = {'type': 'IClamp',
                                                          'delay': 0,
                                                          'dur': self.t_sim,
                                                          'amp': bin_dstr * -1 + amp_dstr}
        self.netParams.stimTargetParams['Input_StrD1->StrD1'] = {'source': 'Input_StrD1',
                                                                 'conds': {'pop': 'StrD1'},
                                                                 'sec': 'soma',
                                                                 'loc': 0}

        # iStr receve a constante 3 density current
        self.netParams.stimSourceParams['Input_StrD2'] = {'type': 'IClamp',
                                                          'delay': 0, 'dur': self.t_sim,
                                                          'amp': bin_istr * -1 + amp_istr}
        self.netParams.stimTargetParams['Input_StrD2->StrD2'] = {'source': 'Input_StrD2',
                                                                 'conds': {'pop': 'StrD2'},
                                                                 'sec': 'soma',
                                                                 'loc': 0}

        # Thalamus receve a constante 1.2 density current
        self.netParams.stimSourceParams['Input_th'] = {'type': 'IClamp',
                                                       'delay': 0,
                                                       'dur': self.t_sim,
                                                       'amp': bin_th * -1 + amp_th}
        self.netParams.stimTargetParams['Input_th->TH'] = {'source': 'Input_th',
                                                           'conds': {'pop': 'TH'},
                                                           'sec': 'soma',
                                                           'loc': 0}


    def filter_LFP( self, lfp ):
        h = firwin( 3, [0.5, 250], fs=1000 )
        filtered = lfilter( h, 1, lfp )
        return filtered


    def extractLFP_SP(self):
        lfp = sim.allSimData['LFP']
        # [ f, t ]
        lfp = np.transpose(lfp, [1, 0])

        # calculate SP using Welch method
        lfp_f, lfp_dimensions = signal.welch( lfp[0], 1000, nperseg=1024, detrend=False )
        lfp_fft = np.zeros(( len(electrodesPos)//nelec, lfp_dimensions.shape[0] ))
        for i in range( 0, lfp.shape[0], nelec ):
            reg_fft = list()
            for j in range( nelec ):
                s = self.filter_LFP( lfp[i+j] )
                reg_fft.append( signal.welch( s, 1000, nperseg=1024, detrend=False ) )
            lfp_f, lfp_fft[i//nelec, :] = np.mean( reg_fft, axis=0 )
        return lfp_f, lfp_fft


    def extractLFP_raw(self):
        lfp = sim.allSimData['LFP']
        # [ f, t ]
        lfp = np.transpose(lfp, [1, 0])
        return lfp


    def extractSpikes(self):
        spikes = self.Spikes
        spk_dict = dict()
        n_strd1 = self.netParams.popParams['StrD1']['numCells']
        n_strd2 = self.netParams.popParams['StrD2']['numCells']
        n_th  = self.netParams.popParams['TH']['numCells']
        n_gpi = self.netParams.popParams['GPi']['numCells']
        n_gpe = self.netParams.popParams['GPe']['numCells']
        n_cor_rs  = self.netParams.popParams['CTX_RS']['numCells']
        n_cor_fsi = self.netParams.popParams['CTX_FSI']['numCells']
        n_stn = self.netParams.popParams['STN']['numCells']

        c_strd1    = n_strd1
        c_strd2    = c_strd1   + n_strd2
        c_th       = c_strd2   + n_th
        c_gpi      = c_th      + n_gpi
        c_gpe      = c_gpi     + n_gpe
        c_cor_rs   = c_gpe     + n_cor_rs
        c_cor_fsi  = c_cor_rs  + n_cor_fsi
        c_stn      = c_cor_fsi + n_stn
        
        spk_dict['dStr_APs'] = [spikes() for k in range( n_strd1 ) ]
        spk_dict['iStr_APs'] = [spikes() for k in range( n_strd2 )]
        spk_dict['TH_APs']   = [spikes() for k in range( n_th )]
        spk_dict['GPi_APs']  = [spikes() for k in range( n_gpi )]
        spk_dict['GPe_APs']  = [spikes() for k in range( n_gpe )]
        spk_dict['Cor_RS_APs']  = [spikes() for k in range( n_cor_rs )]
        spk_dict['Cor_FSI_APs'] = [spikes() for k in range( n_cor_fsi )]
        spk_dict['STN_APs']  = [spikes() for k in range( n_stn )]

        for i in range(0, len(sim.allSimData.spkt)):
            if (sim.allSimData.spkid[i] >= 0 and sim.allSimData.spkid[i] < c_strd1):
                spk_dict['dStr_APs'][int(sim.allSimData.spkid[i])].times = spk_dict['dStr_APs'][
                                                                   int(sim.allSimData.spkid[i])].times + [
                                                                   sim.allSimData.spkt[i]]
            elif (sim.allSimData.spkid[i] >= c_strd1 and sim.allSimData.spkid[i] < c_strd2):
                spk_dict['iStr_APs'][int(sim.allSimData.spkid[i] - c_strd1)].times = spk_dict['iStr_APs'][
                                                                        int(sim.allSimData.spkid[i] - c_strd1)].times + [
                                                                        sim.allSimData.spkt[i]]
            elif (sim.allSimData.spkid[i] >= c_strd2 and sim.allSimData.spkid[i] < c_th):
                spk_dict['TH_APs'][int(sim.allSimData.spkid[i] - c_strd2)].times = spk_dict['TH_APs'][
                                                                      int(sim.allSimData.spkid[i] - c_strd2)].times + [
                                                                      sim.allSimData.spkt[i]]
            elif (sim.allSimData.spkid[i] >= c_th and sim.allSimData.spkid[i] < c_gpi):
                spk_dict['GPi_APs'][int(sim.allSimData.spkid[i] - c_th)].times = spk_dict['GPi_APs'][
                                                                       int(sim.allSimData.spkid[i] - c_th)].times + [
                                                                       sim.allSimData.spkt[i]]
            elif (sim.allSimData.spkid[i] >= c_gpi and sim.allSimData.spkid[i] < c_gpe):
                spk_dict['GPe_APs'][int(sim.allSimData.spkid[i] - c_gpi)].times = spk_dict['GPe_APs'][
                                                                       int(sim.allSimData.spkid[i] - c_gpi)].times + [
                                                                       sim.allSimData.spkt[i]]
            elif (sim.allSimData.spkid[i] >= c_gpe and sim.allSimData.spkid[i] < c_cor_rs):
                spk_dict['Cor_RS_APs'][int(sim.allSimData.spkid[i] - c_gpe)].times = spk_dict['Cor_RS_APs'][
                                                                       int(sim.allSimData.spkid[i] - c_gpe)].times + [
                                                                       sim.allSimData.spkt[i]]
            elif (sim.allSimData.spkid[i] >= c_cor_rs and sim.allSimData.spkid[i] < c_cor_fsi):
                spk_dict['Cor_FSI_APs'][int(sim.allSimData.spkid[i] - c_cor_rs)].times = spk_dict['Cor_FSI_APs'][
                                                                       int(sim.allSimData.spkid[i] - c_cor_rs)].times + [
                                                                       sim.allSimData.spkt[i]]
            elif (sim.allSimData.spkid[i] >= c_cor_fsi and sim.allSimData.spkid[i] < c_stn):
                spk_dict['STN_APs'][int(sim.allSimData.spkid[i] - c_cor_fsi)].times = spk_dict['STN_APs'][
                                                                       int(sim.allSimData.spkid[i] - c_cor_fsi)].times + [
                                                                       sim.allSimData.spkt[i]]
        return spk_dict


    def extractMFR(self):
        mfr = [sim.allSimData.popRates['CTX_FSI'],
               sim.allSimData.popRates['CTX_RS'],
               sim.allSimData.popRates['GPe'],
               sim.allSimData.popRates['GPi'],
               sim.allSimData.popRates['STN'],
               sim.allSimData.popRates['StrD1'],
               sim.allSimData.popRates['StrD2'],
               sim.allSimData.popRates['TH']]
        for i in range(0, 8):
            mfr[i] = round(mfr[i], 2)
        return mfr


    def get_gsngi(self):
        return self.gsngi


    def buildSimConfig(self, dt=0.1, lfp=False, seeds=None):
        # Simulation parameters
        simConfig = specs.SimConfig()
        simConfig.duration = self.t_sim  # Duration of the simulation, in ms
        simConfig.dt = dt  # Internal integration timestep to use
        simConfig.verbose = False  # Show detailed messages
        simConfig.printPopAvgRates = True
        if seeds is not None:
            simConfig.seeds = seeds

        # Recording
        simConfig.recordStep = 1  # Step size in ms to save data (eg. V traces, LFP, etc)
        simConfig.recordCells = ['allCells']
        simConfig.recordSpikesGids = True

        # lfp and plot
        if lfp:
            simConfig.analysis['plotRaster'] = False
            simConfig.recordLFP = electrodesPos 
            simConfig.saveLFPCells = True
            #simConfig.analysis['plotLFP'] = {'electrodes': ['all'],
            #                                  'includeAxon': False,
            #                                  'timeRange': [0, 2000],
            #                                  'plots': ['timeSeries', 'locations', 'PSD'],
            #                                  # 'plots': ['locations'],
            #                                  'showFig': True}
        return simConfig


    def simulate(self, dt=0.1, lfp=False, seeds=None):
        simConfig = self.buildSimConfig(dt=dt, lfp=lfp, seeds=seeds)
        sim.createSimulateAnalyze(netParams=self.netParams, simConfig=simConfig)
        # out_dict = self.extractSpikes()
        # print( sim.allSimData['LFPCells'] )
        return sim

if __name__=='__main__':
    network = Network()
    network.simulate()
