import numpy as np

class GA_params:
    def __init__( self ):
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

        self.minParamValues = np.array( minParamValues )
        self.maxParamValues = np.array( maxParamValues )

    def transform( self, cand ):
        cand = np.array( cand )
        cand = cand * ( self.maxParamValues - self.minParamValues )
        cand = cand + self.minParamValues
        return list( cand )


    def inv_transform( self, cand ):
        cand = np.array( cand )
        cand = cand - self.minParamValues
        cand = cand / ( self.maxParamValues - self.minParamValues )
        return list( cand )






