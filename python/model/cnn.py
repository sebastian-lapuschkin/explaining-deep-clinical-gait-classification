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
        data = (x_train, x_val, x_test, y_train, y_val, y_test)
        if self.use_gpu:
            #move data to GPU if GPU execution is desired/possible
            data = helpers.arrays_to_cupy(*data)
        else:
            #otherwise, make sure the data is available to the CPU
            data = helpers.arrays_to_numpy(*data)

        return data

    def assert_shapes(self, x_shape, y_shape):
        """ assert the shapes of input data for all fully connected models """
        assert len(x_shape) >= 3, "Expected at least 2-dimensional shape tuple for MLP type models, but got x_shape={}".format(x_shape)
        assert len(y_shape) == 2, "Expected 2-dimensional shape tuple for MLP type models, but got y_shape={}".format(y_shape)