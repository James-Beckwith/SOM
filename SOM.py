''' Self organising map (SOM) class. Fit an N-dimensional manifold to the input
 feature vector. A Gaussian neighbourhood function is used.
 Input should be in the form - Nexamples x Nfeatures.
 The shape of the SOM grid is regular rectangular by default but a set of
 generic x,y,z,..., N co-ordinate pairs can be given defining the neuron locations.
 If generic points are given then the array shape should be N x Nneuron where
 Nneuron is the total number of neurons and N is the number of dimensions of the
 SOM map

 requirements - numpy

 Author - James Beckwith
 '''

import numpy as np

class SOM:

    # initialisation
    def __init__(self, input, somShape=[8,8], neighbourSize=10.0, learningRate=1.0, learningDecay=1000.0, neighbourDecay=1000.0, numEpochs=10000, flag1DGeneric = False, weightInitSigma=0.01, weightInitMean=0.0):

        # determine input shape and size
        somShape = np.array(somShape)
        shape = np.shape(somShape)
        # are generic points given?
        if (len(shape)!=1) or (flag1DGeneric==True):
            if len(shape) > np.shape(np.array(input))[1]:
                raise error("Number of dimensions of SOM grid must be less than number of features")
            # generic points are given, get x, y, ... from somShape variable
            self.neuronLocs = somShape
            self.Nneuron = shape[1]
            self.NSOMDimensions = shape[0]
        else:
            # check if number of dimensions is less than number of features
            if len(shape) > np.shape(np.array(input))[1]:
                raise error("Number of dimensions of SOM grid must be less than number of features")
            # dimensions given in somShape
            self.NSOMDimensions = len(somShape)
            self.Nneuron = np.product(somShape)
            # define locations of each neuron and store as an Ndimension x Nneuron matrix
            self.neuronLocs = np.reshape(np.meshgrid(*[np.arange(0, x) for x in somShape]),[self.NSOMDimensions, self.Nneuron])

        # initialise other variables
        self.input = input
        self.learningRate = learningRate
        self.learningRate0 = learningRate
        self.neighbourSize = neighbourSize
        self.neighbourSize0 = neighbourSize
        self.learningDecay = learningDecay
        self.neighbourDecay = neighbourDecay
        self.numEpochs = numEpochs
        self.winningNeuron = []
        self.epoch = 0.0
        self.currentExample = []
        self.Nfeature = np.shape(input)[1]
        self.Nexample = np.shape(input)[0]

        # initialise weights
        self.weights = weightInitMean + weightInitSigma * np.random.randn(self.Nfeature, self.Nneuron)

    # function to competition adn adaptation
    def run(self):
        for i in range(self.numEpochs):
            # select a random example from the input
            self.currentExample = self.input[np.random.randint(self.Nneuron),:]
            # run competition
            self.winningNeuron = self.competition()
            #run adaptation
            self.adaptation()
            #update learning rate and neighoburhood size
            self.learningRate = self.learningRate0 * np.exp(-self.epoch / self.learningDecay)
            self.neighbourSize = self.neighbourSize0 * np.exp(-self.epoch / self.neighbourDecay)
            self.epoch += 1

        #final competition to find best matching neuron for each input example
        self.finalNeuron = np.zeros(self.Nexample)
        for i in range(self.Nexample):
            self.currentExample = self.input[i,:]
            self.finalNeuron[i] = self.competition()

    # run competition phase and decide winning neuron
    def competition(self):
        # find minimum distance between selected example features and SOM weights
        distances = np.zeros(self.Nneuron)
        for i in range(self.Nfeature):
            distances += (self.currentExample[i] - self.weights[i,:]) ** 2.0

        winningNeuron = np.argmin(distances)

        return winningNeuron

    def adaptation(self):
        # adapt weghts based on best winning neuron
        # determine distance between all neurons and wining neuron
        winningNeuronLoc = self.neuronLocs[:,self.winningNeuron]
        distances = np.zeros(self.Nneuron)
        for i in range(self.Nneuron):
            distances[i] = sum((winningNeuronLoc - self.neuronLocs[:,i]) ** 2.0) ** 0.5

        #define neighbourhood for contrbution for all neurons
        print(self.neighbourSize)
        neighbourhood = np.exp(-distances / (self.neighbourSize) ** 2.0)

        # define error as difference between selected example and winning neurons
        # weights
        error = (self.currentExample - self.weights[:,self.winningNeuron])

        # update weights
        for i in range(self.Nneuron):
            self.weights[:,i] += error * self.learningRate * neighbourhood[i]
