from GenericBG import Network
from netpyne import sim

class MarmosetBG( Network ):
    def __init__( self ):
        super( MarmosetBG, self ).__init__()
        self.set_marmoset()


    def set_marmoset( self ):
        genotype = [ 0.47148627, 0.         , 1.        , 0.24748603, 1.        , 0.0930143,
                     1.        ,  0.01593558, 0.9980333 , 0.19937477, 1.        , 0.48306456,
                     0.48590738,  0.14889275 ]
        self.set_genotype( genotype )


if __name__=='__main__':
    network = MarmosetBG()
    network.simulate()
    spikes_dict = network.extractSpikes()
    print('SPIKES')
    for key in spikes_dict.keys():
        print(key)
        print( [s.times for s in spikes_dict[key]] )
    mfr = network.extractMFR()
    print(mfr)