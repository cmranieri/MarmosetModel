import pylab              # scientific computing and plotting
import matplotlib
from random import Random # pseudorandom number generation
from inspyred import ec   # evolutionary algorithm
import GenericBG # neural network designed through netpyne, to be optimized
from netpyne import sim   # neural network design and simulation
import numpy as np
import inspyred
import time
from scipy import stats
from GA_utils import calc_power_band


mode   = 'lfp'
has_pd = True
A = 1.0

minParamValues = list()
maxParamValues = list()
# Additional stimulations
# [ Th, GPe, GPi ]
#Leave e-3 term to the moment of phenotype mapping to ensure genotype range is the same for all params=
minParamValues += list( np.array([1.2, 3., 3.]) * 1e-3 * (1 - 0.5) )
maxParamValues += list( np.array([1.2, 3., 3.]) * 1e-3 * (1 + 0.5) )

# Conductances
# [ gkcabar, gahp ]
minParamValues += list( np.array([5., 10.]) * 1e-3 * (1 - 0.5) )
maxParamValues += list( np.array([5., 10.]) * 1e-3 * (1 + 0.5) )

# Conductances modulator from cortex to str
# multiplies gcostr and gsyn
minParamValues += [0.8]
maxParamValues += [1.2]

# Number of cells in each region
minParamValues += [10.]*8
maxParamValues += [30.]*8

minParamValues = np.array( minParamValues )
maxParamValues = np.array( maxParamValues )


def transform_cand( cand ):
    cand = np.array( cand )
    cand = cand * ( maxParamValues - minParamValues )
    cand = cand + minParamValues
    return list( cand )


def inv_transform( cand ):
    cand = np.array( cand )
    cand = cand - minParamValues
    cand = cand / ( maxParamValues - minParamValues )
    return list( cand )


def fitnessD( x, target ):
    d = min( A, ( abs( A * (x - target) ) / target ) )
    return d


def computeFitness( ref_dict, target_dict ):
    n = len( ref_dict )
    fit = 0.0
    for key in ref_dict.keys():
        fit += fitnessD( ref_dict[key] , target_dict[key] )
    print( 'fitness:' , n - fit )
    return n - fit


def loadSimMFR( sim ):
    mfr = dict()
    mfr['GPi']     = sim.allSimData.popRates['GPi']
    mfr['GPe']     = sim.allSimData.popRates['GPe']
    mfr['StrD1']   = sim.allSimData.popRates['StrD1']
    mfr['StrD2']   = sim.allSimData.popRates['StrD2']
    mfr['TH']      = sim.allSimData.popRates['TH']
    mfr['STN']     = sim.allSimData.popRates['STN']
    mfr['CTX_RS']  = sim.allSimData.popRates['CTX_RS']
    mfr['CTX_FSI'] = sim.allSimData.popRates['CTX_FSI']
    return mfr


# CHECK THIS FUNCTION
def loadSimLFPcoef( net ):
    lfp_f, lfp_fft = net.extractLFP_SP()
    denominator = calc_power_band( lfp_fft, lfp_f, 0, 51 )
    alpha = calc_power_band( lfp_fft, lfp_f, 8, 14 )
    beta  = calc_power_band( lfp_fft, lfp_f, 13, 31 )
    betaH = calc_power_band( lfp_fft, lfp_f, 30, 51 )
    coefs = ( alpha + beta + betaH ) / denominator

    lfp_coef = dict()
    area_names = ["StrD1", "StrD2", "TH", "GPi", "GPe", "CtxRS", "CtxFSI", "STN"]
    for i in range( len(coefs) ):
        lfp_coef[ area_names[i] ] = ( coefs[i] )
    return lfp_coef


# TARGETS NEED TO BE CHANGED FOR THE HEALTHY CONDITION.
def loadHealthyLFPcoef():
    lfp_coef = dict()
    area_names = ["StrD1", "StrD2", "TH", "GPi", "GPe", "CtxRS", "CtxFSI", "STN"]
    # ["M1", "PUT", "GPe", "GPi", "VL", "STN", "VPL", ""]
    coefs = [ 0.3922243,  0.44226615, 0.42052627, 0.46059581, 0.38124447, 0.36704531, 0.3721786 ]
    coefs = [ coefs[1], coefs[1], (coefs[4]+coefs[6])/2, coefs[3], coefs[2], coefs[0], coefs[0], coefs[5] ]
    for i in range( len(coefs) ):
        lfp_coef[ area_names[i] ] = ( coefs[i] )
    return lfp_coef


def loadPDLFPcoef():
    lfp_coef = dict()
    area_names = ["StrD1", "StrD2", "TH", "GPi", "GPe", "CtxRS", "CtxFSI", "STN"]
    # ["M1", "PUT", "GPe", "GPi", "VL", "STN", "VPL", ""]
    coefs = [ 0.3922243,  0.44226615, 0.42052627, 0.46059581, 0.38124447, 0.36704531, 0.3721786 ]
    coefs = [ coefs[1], coefs[1], (coefs[4]+coefs[6])/2, coefs[3], coefs[2], coefs[0], coefs[0], coefs[5] ]
    for i in range( len(coefs) ):
        lfp_coef[ area_names[i] ] = ( coefs[i] )
    return lfp_coef


def loadRatMFR():
    targetMfr = dict()
    targetMfr['GPi']     = 22.95
    targetMfr['GPe']     = 43.20
    targetMfr['StrD1']   = 1e-3
    targetMfr['StrD2']   = 1e-3
    targetMfr['TH']      = 27.35 
    targetMfr['STN']     = 8.65
    targetMfr['CTX_RS']  = 2.95
    targetMfr['CTX_FSI'] = 1.85
    return targetMfr

def loadHealthyMFR():
    targetMfr = dict()
    targetMfr['GPi']     = 75.0 # from literature
    targetMfr['GPe']     = 55.0 # from literature
    targetMfr['StrD1']   = 3.34
    targetMfr['StrD2']   = 3.34
    targetMfr['TH']      = 10.04 
    targetMfr['STN']     = 25.08
    targetMfr['CTX_RS']  = 5.38
    targetMfr['CTX_FSI'] = 10.76 # double of CTX_RS
    return targetMfr


def loadPDMFR():
    targetMfr = dict()
    targetMfr['GPi']     = 75.0 # from literature
    targetMfr['GPe']     = 55.0 # from literature
    targetMfr['StrD1']   = 18.92
    targetMfr['StrD2']   = 18.92
    targetMfr['TH']      = 12.58 
    targetMfr['STN']     = 18.47
    targetMfr['CTX_RS']  = 15.37
    targetMfr['CTX_FSI'] = 30.74 # double of CTX_RS
    return targetMfr
    

# design parameter generator function, used in the ec evolve function --> final_pop = my_ec.evolve(generator=generate_netparams,...)
def generate_netparams(random, args):
    size = args.get('num_inputs')
    myclip_a = 0
    myclip_b = 1
    my_mean = 0.5
    my_std = 0.8

    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    initialParams = stats.truncnorm.rvs( a, b, my_mean, my_std, size=size )
    return initialParams


def set_genotype( cand, net ):
    cand = transform_cand( cand )
    net.buildPopulationParameters( n_gpe   = int( cand[6] ),
                                   n_gpi   = int( cand[7] ),
                                   n_th    = int( cand[8] ),
                                   n_strd1 = int( cand[9] ),
                                   n_strd2 = int( cand[10] ),
                                   n_rs    = int( cand[11] ),
                                   n_fsi   = int( cand[12] ),
                                   n_stn   = int( cand[13] ) )
    net.buildCellConnRules()
    #net.buildCellConnRules( stn_gpe = int( cand[8] ),
    #                        gpe_gpe = int( cand[9] ),
    #                        stn_gpi = int( cand[10] ),
    #                        gpe_gpi = int( cand[11] ),
    #                        strd2_strd2 = int( cand[12] ),
    #                        strd1_strd1 = int( cand[13] ),
    #                        rs_fsi = int( cand[14] ),
    #                        fsi_rs = int( cand[15] ) )
    net.buildStimParams( amp_th  = cand[0],
                         amp_gpe = cand[1],
                         amp_gpi = cand[2])
    net.netParams.cellParams['STN']['secs']['soma']['mechs']['SubTN']['gkcabar'] = cand[3] 
    net.netParams.cellParams['GPe']['secs']['soma']['mechs']['GP']['gahp']     = cand[4] 
    net.netParams.cellParams['GPi']['secs']['soma']['mechs']['GP']['gahp']     = cand[4] 
    net.strConnRules( gsynmod = cand[5] )
    return net


# design fitness function, used in the ec evolve function --> final_pop = my_ec.evolve(...,evaluator=evaluate_netparams,...)
def evaluate_netparams(candidates, args):
    global net
    fitnessCandidates = list()

    for icand,cand in enumerate(candidates):
        # modify network params based on this candidate params (genes)
        net = set_genotype( cand, net )
        simConfig = net.buildSimConfig( lfp = (mode=='lfp') )
        sim.createSimulate( netParams = net.netParams,
                            simConfig = simConfig )
        if mode == 'mfr':
            ref_dict    = loadSimMFR( sim )
            if has_pd:
                target_dict = loadPDMFR()
            else:
                target_dict = loadHealthyMFR()
        elif mode == 'lfp':
            ref_dict    = loadSimLFPcoef( net )
            if has_pd:
                target_dict = loadPDLFPcoef()
            else:
                target_dict = loadHealthyLFPcoef()

        # add to list of fitness for each candidate
        fitness =  computeFitness( ref_dict, target_dict )

        fitnessCandidates.append( fitness )
    return fitnessCandidates


def runGA():
    # create random seed for evolutionary computation algorithm --> my_ec = ec.EvolutionaryComputation(rand)
    rand = Random()
    rand.seed( None )
    
    # create fresh network
    global net
    net = GenericBG.Network( t_sim = 2000, has_pd = has_pd )
    net.simulate( lfp = (mode == 'lfp') )

    # instantiate evolutionary computation algorithm with random seed
    my_ec = ec.DEA(rand) #check if we should use ec.EvolutionaryComputation instead

    # establish parameters for the evolutionary computation algorithm, additional documentation can be found @ pythonhosted.org/inspyred/reference.html
    my_ec.selector = ec.selectors.tournament_selection  # tournament sampling of individuals from population (<num_selected> individuals are chosen based on best fitness performance in tournament)

    #toggle variators
    my_ec.variator = [ec.variators.uniform_crossover,   # biased coin flip to determine whether 'mom' or 'dad' element is passed to offspring design
                      ec.variators.gaussian_mutation]   # gaussian mutation which makes use of bounder function as specified in --> my_ec.evolve(...,bounder=ec.BOunder(minParamValues, maxParamValues),...)

    my_ec.replacer = ec.replacers.generational_replacement    # existing generation is replaced by offspring, with elitism (<num_elites> existing individuals will survive if they have better fitness than offspring)

    my_ec.terminator = ec.terminators.evaluation_termination  # termination dictated by number of evaluations that have been run

    #toggle observers
    my_ec.observer = [ ec.observers.stats_observer,  # print evolutionary computation statistics
    #                   ec.observers.plot_observer,   # plot output of the evolutionary computation as graph
                       ec.observers.best_observer,   # print the best individual in the population to screen
                       ec.observers.file_observer ]

    #call evolution iterator
    final_pop = my_ec.evolve(generator=generate_netparams,  # assign design parameter generator to iterator parameter generator
                          evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
                          mp_evaluator=evaluate_netparams,     # assign fitness function to iterator evaluator
                          #mp_procs = 8,                     #leave this commented to use all available processors
                          pop_size=200,                      # each generation of parameter sets will consist of 10 individuals
                          maximize=True,                   # False: best fitness corresponds to minimum value
                          bounder=ec.Bounder(0, 1), # boundaries for parameter set ([probability, weight, delay])
                          max_evaluations=1000,              # evolutionary algorithm termination at 50 evaluations
                          num_selected=20,                  # number of generated parameter sets to be selected for next generation
                          mutation_rate=0.1,                # rate of mutation
                          num_inputs=14,                    # len([probability, weight, delay])
                          gaussian_stdev=0.3,
                          num_elites=1)                     # 1 existing individual will survive to next generation if it has better fitness than an individual selected by the tournament selection

    final_pop.sort(reverse=True) 
    bestCand = final_pop[0].candidate
    global bestFitness
    bestFitness = final_pop[0].fitness 
    return bestCand


if __name__=='__main__':
    for trial in range( 93 ):
        bestCand = runGA()
        net = set_genotype( bestCand, net )
        simConfig = net.buildSimConfig( lfp = (mode=='lfp') )
        sim.createSimulate( netParams=net.netParams, simConfig=simConfig )
        mfr_dict = loadSimMFR( sim )
        
        #global bestFitness
        #with open( 'out.txt', 'a' ) as f:
        #    f.write( str(bestFitness) + '\n' )
        #    f.write( str(transform_cand(bestCand)) + '\n' )
        #    f.write( str(mfr_dict) + '\n\n')
        #with open( 'best_gen_%d.txt' % trial, 'w' ) as f:
        #    f.write( str(bestCand) )
        print( np.array( bestCand ) )
        print( transform_cand( bestCand ) )

    #The format of each line of the statistics file is as follows:
    #generation number, population size, worst, best, median, average, standard deviation

    #The format of each line of the individuals file is as follows:
    #generation number, individual number, fitness, string representation of candidate
