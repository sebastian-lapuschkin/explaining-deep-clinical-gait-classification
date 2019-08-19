from abc import abstractmethod
from .base import ModelArchitecture
from .training import *
import numpy
import importlib.util as imp
if imp.find_spec("cupy"): import cupy # import cupy, if available
from modules import * #import all NN modules
import helpers



######################################
# Abstract base class for CNNs.
# parameterized classes below.
######################################

class ConvolutionalArchitectureBase(ModelArchitecture):
    #Note: this class is abstract and provides the preprocessing for all CNN type models
    #Architectures need to be desingned in sublcasses

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_data(self, x_train, x_val, x_test,
                              y_train, y_val, y_test):
        """
        prepare data and labels as input to the model.
        convert input multi-dim arrays into vectors
        """

        #add additional 1-dim channel axis to training inputs
        data = (x_train[..., None], x_val[..., None], x_test[..., None], y_train, y_val, y_test)
        if self.use_gpu:
            #move data to GPU if GPU execution is desired/possible
            data = helpers.arrays_to_cupy(*data)
        else:
            #otherwise, make sure the data is available to the CPU
            data = helpers.arrays_to_numpy(*data)

        return data

    def assert_shapes(self, x_shape, y_shape):
        """ assert the shapes of input data for all fully connected models """
        assert len(x_shape) == 4, "Expected 4-dimensional shape tuple for MLP type models, but got x_shape={}".format(x_shape)
        assert len(y_shape) == 2, "Expected 2-dimensional shape tuple for MLP type models, but got y_shape={}".format(y_shape)


#################################
# CNN architecture specifications
#################################

class CnnA6(ConvolutionalArchitectureBase, NeuralNetworkTrainingDefault):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def build_model(self, x_shape, y_shape):
        #samples are expected in shape 101 x 6 x 1 ie x_shape should be N x 101 x 6 x 1
        self.assert_shapes(x_shape, y_shape)
        assert x_shape[1:] == (101, 6, 1)
        n_classes = y_shape[1]

        h1 = Convolution(filtersize=(6,6,1,32), stride=(1,1))   # h1: 96 x 1 x 32
        h2 = Convolution(filtersize=(6,1,32,32), stride=(1,1))  # h2 output: 91 x 1 x 32 = 2912
        h3 = Linear(2912, n_classes)
        self.model = Sequential([h1, Rect(), h2, Rect(), Flatten(), h3, SoftMax()])
        if not self.use_gpu:
            self.model.to_numpy()
        else:
            self.model.to_cupy()

class CnnA3(ConvolutionalArchitectureBase, NeuralNetworkTrainingDefault):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gpu = False

    def build_model(self, x_shape, y_shape):
        #samples are expected in shape 101 x 6 x 1 ie x_shape should be N x 101 x 6 x 1
        self.assert_shapes(x_shape, y_shape)
        assert x_shape[1:] == (101, 6, 1)
        n_classes = y_shape[1]

        h1 = Convolution(filtersize=(3,6,1,32), stride=(1,1))   # h1 output: 99 x 1 x 32
        h2 = Convolution(filtersize=(3,1,32,32), stride=(1,1))  # h2 output: 97 x 1 x 32 = 3104
        h3 = Linear(3104, n_classes)
        self.model = Sequential([h1, Rect(), h2, Rect(), Flatten(), h3, SoftMax()])
        if not self.use_gpu:
            self.model.to_numpy()
        else:
            self.model.to_cupy()

class CnnAshort(ConvolutionalArchitectureBase, NeuralNetworkTrainingDefault):
    #formerly known as CNN-A, which is shorter than CnnA6 by one conv layer
    #this could explain the performance discrepancy.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self, x_shape, y_shape):
        #samples are expected in shape 101 x 6 x 1 ie x_shape should be N x 101 x 6 x 1
        self.assert_shapes(x_shape, y_shape)
        assert x_shape[1:] == (101, 6, 1)
        n_classes = y_shape[1]

        h1 = Convolution(filtersize=(6,6,1,32), stride=(1,1))   # h1: 96 x 1 x 32 = 3072
        h2 = Linear(3072, n_classes)
        self.model = Sequential([h1, Rect(), Flatten(), h2, SoftMax()])
        if not self.use_gpu:
            self.model.to_numpy()
        else:
            self.model.to_cupy()

class CnnC3(ConvolutionalArchitectureBase, NeuralNetworkTrainingDefault):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self, x_shape, y_shape):
        #samples are expected in shape 101 x 6 x 1 ie x_shape should be N x 101 x 6 x 1
        self.assert_shapes(x_shape, y_shape)
        assert x_shape[1:] == (101, 6, 1)
        n_classes = y_shape[1]

        h1 = Convolution(filtersize=(3,3,1,32), stride=(1,1))  # h1 output: 99 x 4 x 32
        h2 = Convolution(filtersize=(3,3,32,32), stride=(1,1)) # h2 output: 97 x 2 x 32
        h3 = Convolution(filtersize=(2,2,32,32), stride=(1,1)) # h3 output: 96 x 1 x 32 = 3072
        h4 = Linear(3072,n_classes)
        self.model = Sequential([h1, Rect(), h2, Rect(), h3, Rect(), Flatten(), h4, SoftMax()])
        if not self.use_gpu:
            self.model.to_numpy()
        else:
            self.model.to_cupy()

class CnnC6(ConvolutionalArchitectureBase, NeuralNetworkTrainingDefault):
    #same as CnnAshort for 6-channel-input-data
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self, x_shape, y_shape):
        #samples are expected in shape 101 x 6 x 1 ie x_shape should be N x 101 x 6 x 1
        self.assert_shapes(x_shape, y_shape)
        assert x_shape[1:] == (101, 6, 1)
        n_classes = y_shape[1]

        h1 = Convolution(filtersize=(6,6,1,32), stride=(1,1)) # h1 output: 96 x 1 x 32 = 3072
        h2 = Linear(3072,n_classes)
        self.model = Sequential([h1, Rect(), Flatten(), h2, SoftMax()])
        if not self.use_gpu:
            self.model.to_numpy()
        else:
            self.model.to_cupy()