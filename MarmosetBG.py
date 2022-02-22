from GenericBG import Network
from GA import set_genotype
from netpyne import sim

class MarmosetBG( Network ):
    def __init__( self ):
        super( MarmosetBG, self ).__init__()
        self.set_marmoset()


    def set_marmoset( self ):
        genotype = [ 0.47148627, 0.         , 1.        , 0.24748603, 1.        , 0.0930143,
                     1.        ,  0.01593558, 0.9980333 , 0.19937477, 1.        , 0.48306456,
                     0.48590738,  0.14889275 ]
        set_genotype( genotype, self )


if __name__=='__main__':
    network = MarmosetBG()
    network.simulate()