from abc import abstractmethod
from .base import ModelArchitecture
from .training import *
import numpy
import importlib.util as imp
if imp.find_spec("cupy"): import cupy # import cupy, if available
from modules import * #import all NN modules
import helpers



######################################
# Abstract base class for MLPs.
# parameterized classes below.
######################################

class FullyConnectedArchitectureBase(ModelArchitecture):
    #Note: this class is abstract and provides the preprocessing for all MLP type models
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
        assert len(x_shape) >= 2, "Expected at least 2-dimensional shape tuple for MLP type models (Flatten layer takes care of the rest), but got x_shape={}".format(x_shape)
        assert len(y_shape) == 2, "Expected 2-dimensional shape tuple for MLP type models, but got y_shape={}".format(y_shape)


    def postprocess_relevance(self, *args, **kwargs):
        relevance = helpers.arrays_to_numpy(*args)
        return relevance





#################################
# MLP architecture specifications
#################################

class MlpLinear(FullyConnectedArchitectureBase, NeuralNetworkTrainingDefault):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gpu = False # GPU execution overhead is not worth it.

    def build_model(self, x_shape, y_shape):
        self.assert_shapes(x_shape, y_shape)
        n_dims = numpy.prod(x_shape[1:])
        n_classes = y_shape[1]

        self.model = Sequential([Flatten(), Linear(n_dims, n_classes)])
        if not self.use_gpu:
            self.model.to_numpy()
        else:
            self.model.to_cupy()


####################################################################
# MLP Template class for all RELU-architectures with 2 hidden layers
####################################################################

class Mlp2LayerTemplate(FullyConnectedArchitectureBase, NeuralNetworkTrainingDefault):
    # 2 hidden layers of X neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = None #define number of hidden units in implementing classes

    def build_model(self, x_shape, y_shape):
        self.assert_shapes(x_shape, y_shape)
        n_dims = numpy.prod(x_shape[1:])
        n_classes = y_shape[1]

        self.model = Sequential([
            Flatten(),
            Linear(n_dims, self.n_hidden), Rect(),
            Linear(self.n_hidden, n_classes), SoftMax()]
            )
        if not self.use_gpu:
            self.model.to_numpy()
        else:
            self.model.to_cupy()

################################################################
# MLP classes with 2 hidden layers and hidden unit specification
################################################################

class Mlp2Layer64Unit(Mlp2LayerTemplate):
    # 2 hidden layers of 64 neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = 64
        self.use_gpu = False #not worth using the gpu yet

class Mlp2Layer128Unit(Mlp2LayerTemplate):
    # 2 hidden layers of 512 neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = 128

class Mlp2Layer256Unit(Mlp2LayerTemplate):
    # 2 hidden layers of 256 neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = 256

class Mlp2Layer512Unit(Mlp2LayerTemplate):
    # 2 hidden layers of 256 neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = 512

class Mlp2Layer768Unit(Mlp2LayerTemplate):
    # 2 hidden layers of 256 neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = 768



####################################################################
# MLP Template class for all RELU-architectures with 3 hidden layers
####################################################################

class Mlp3LayerTemplate(FullyConnectedArchitectureBase, NeuralNetworkTrainingDefault):
    # 3 hidden layers of X neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = None #define number of hidden units in implementing classes

    def build_model(self, x_shape, y_shape):
        self.assert_shapes(x_shape, y_shape)
        n_dims = numpy.prod(x_shape[1:])
        n_classes = y_shape[1]

        self.model = Sequential([
            Flatten(),
            Linear(n_dims, self.n_hidden), Rect(),
            Linear(self.n_hidden, self.n_hidden), Rect(),
            Linear(self.n_hidden, n_classes), SoftMax()]
            )
        if not self.use_gpu:
            self.model.to_numpy()
        else:
            self.model.to_cupy()

################################################################
# MLP classes with 3 hidden layers and hidden unit specification
################################################################

class Mlp3Layer64Unit(Mlp3LayerTemplate):
    # 3 hidden layers of 64 neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = 64

class Mlp3Layer128Unit(Mlp3LayerTemplate):
    # 3 hidden layers of 512 neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = 128

class Mlp3Layer256Unit(Mlp3LayerTemplate):
    # 3 hidden layers of 256 neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = 256

class Mlp3Layer512Unit(Mlp3LayerTemplate):
    # 3 hidden layers of 256 neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = 512

class Mlp3Layer768Unit(Mlp3LayerTemplate):
    # 3 hidden layers of 256 neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = 768



##############################################################################################################
# MLP classes with 3 hidden layers and hidden unit specification, longer training, larger batches, more epochs
##############################################################################################################

# NOTE: inheriting the training procedure first effectively picks the training from NeuralNetworkTrainingIncreaseBatchSize
# since this class now provides the first match for a .train() method
class Mlp3Layer64UnitLongerTraining(NeuralNetworkTrainingIncreaseBatchSize, Mlp3Layer64Unit):
    pass

class Mlp3Layer128UnitLongerTraining(NeuralNetworkTrainingIncreaseBatchSize, Mlp3Layer128Unit):
    pass

class Mlp3Layer256UnitLongerTraining(NeuralNetworkTrainingIncreaseBatchSize, Mlp3Layer256Unit):
    pass

class Mlp3Layer512UnitLongerTraining(NeuralNetworkTrainingIncreaseBatchSize, Mlp3Layer512Unit):
    pass

class Mlp3Layer768UnitLongerTraining(NeuralNetworkTrainingIncreaseBatchSize, Mlp3Layer768Unit):
    pass


##############################################################################################################
# MLP classes with 3 hidden layers and hidden unit specification, longer training, larger batches, more epochs
##############################################################################################################

# NOTE: inheriting the training procedure first effectively picks the training from NeuralNetworkTrainingIncreaseBatchSize
# since this class now provides the first match for a .train() method
class Mlp3Layer64UnitLongerTrainingDecreaseBatchSize(NeuralNetworkTrainingDecreaseBatchSize, Mlp3Layer64Unit):
    pass

class Mlp3Layer128UnitLongerTrainingDecreaseBatchSize(NeuralNetworkTrainingDecreaseBatchSize, Mlp3Layer128Unit):
    pass

class Mlp3Layer256UnitLongerTrainingDecreaseBatchSize(NeuralNetworkTrainingDecreaseBatchSize, Mlp3Layer256Unit):
    pass

class Mlp3Layer512UnitLongerTrainingDecreaseBatchSize(NeuralNetworkTrainingDecreaseBatchSize, Mlp3Layer512Unit):
    pass

class Mlp3Layer768UnitLongerTrainingDecreaseBatchSize(NeuralNetworkTrainingDecreaseBatchSize, Mlp3Layer768Unit):
    pass