''' Self organising map (SOM) class. Fit an N-dimensional manifold to the input
 feature vector. A Gaussian neighbourhood function is used. The shape of the SOM
 grid is regular rectangular by default but a set of generic x,y,z,..., N
 co-ordinate pairs can be given defining the neuron locations.
 If generic points are given then the array shape should be Nneuron x N where
 Nneuron is the total number of neurons and N is the number of dimensions/features

 requirements - numpy

 Author - James Beckwith
 '''

import numpy

class SOM:

    # initialisation
    def __init__(self, Input, somShape=[8,8], neighbourSize=10, learningRate=1, learningDecay=100, neighbourDecay=100, numEpochs=10000, flag1DGeneric = False):

        # determine input shape and size
        somShape = np.array(somShape)
        shape = np.shape(somShape)
        # are generic points given?
        if (len(shape)!=1) or (flag1flag1DGeneric==True):
            # generic points are given, get x, y, ... from somShape variable
            self.neuronLocs = somShape
        else:
            # dimensions given in somShape
            Ndimensions = len(somShape)
                
