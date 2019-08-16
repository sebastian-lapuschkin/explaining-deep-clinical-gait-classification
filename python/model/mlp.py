from abc import abstractmethod
from .base import ModelArchitecture, ModelTraining
import numpy
import importlib.util as imp
if imp.find_spec("cupy"): import cupy # import cupy, if available
from modules import * #import all NN modules
import helpers



######################################
# Abstract base class for Linear SVMs.
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
        x_train = numpy.reshape(x_train, [x_train.shape[0], -1])
        x_val = numpy.reshape(x_val, [x_val.shape[0], -1])
        x_test = numpy.reshape(x_test, [x_test.shape[0], -1])

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
        assert len(x_shape) == 2, "Expected 2-dimensional shape tuple for MLP type models, but got x_shape={}".format(x_shape)
        assert len(y_shape) == 2, "Expected 2-dimensional shape tuple for MLP type models, but got y_shape={}".format(y_shape)


#####################################
# Training schemes for the MLP models
#####################################

class FullyConnectedTrainingDefault(ModelTraining):
    #this clas provides the until now default training scheme for MLPs
    def train_model(self, x_train, y_train, x_val, y_val):
        print('training {} model (3 epochs, default setting)'.format(self.__class__.__name__))
        self.model.train(x_train, y_train, Xval=x_val, Yval=y_val, batchsize=5, lrate=0.005)  # train the model
        self.model.train(x_train, y_train, Xval=x_val, Yval=y_val, batchsize=5, lrate=0.001)  # slower training once the model has converged somewhat
        self.model.train(x_train, y_train, Xval=x_val, Yval=y_val, batchsize=5, lrate=0.0005) # one last epoch

class FullyConnectedTrainingQuickTest(ModelTraining):
    #very short, rudimentary model training for testing
    def train_model(self, x_train, y_train, x_val, y_val):
        print('training {} model (quick test)'.format(self.__class__.__name__))
        self.model.train(x_train, y_train, iters=10)



#################################
# MLP architecture specifications
#################################

class MlpLinear(FullyConnectedArchitectureBase, FullyConnectedTrainingDefault):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gpu = False # GPU execution overhead is not worth it.

    def build_model(self, x_shape, y_shape):
        self.assert_shapes(x_shape, y_shape)
        n_dims = x_shape[1];    n_classes = y_shape[1]

        self.model = Sequential([Linear(n_dims, n_classes)])
        if not self.use_gpu:
            self.model.to_numpy()
        else:
            self.model.to_cupy()


####################################################################
# MLP Template class for all RELU-architectures with 2 hidden layers
####################################################################

class Mlp2LayerTemplate(FullyConnectedArchitectureBase, FullyConnectedTrainingDefault):
    # 2 hidden layers of X neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = None #define number of hidden units in implementing classes

    def build_model(self, x_shape, y_shape):
        self.assert_shapes(x_shape, y_shape)
        n_dims = x_shape[1]
        n_classes = y_shape[1]

        self.model = Sequential([
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
        self.use_gpu = False #not worth using the gpu yet

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

class Mlp3LayerTemplate(FullyConnectedArchitectureBase, FullyConnectedTrainingQuickTest):
    # 3 hidden layers of X neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = None #define number of hidden units in implementing classes

    def build_model(self, x_shape, y_shape):
        self.assert_shapes(x_shape, y_shape)
        n_dims = x_shape[1]
        n_classes = y_shape[1]

        self.model = Sequential([
            Linear(n_dims, self.n_hidden), Rect(),
            Linear(self.n_hidden, self.n_hidden), Rect(),
            Linear(self.n_hidden, n_classes), SoftMax()]
            )
        if not self.use_gpu:
            self.model.to_numpy()
        else:
            self.model.to_cupy()

################################################################
# MLP classes with 2 hidden layers and hidden unit specification
################################################################

class Mlp3Layer64Unit(Mlp3LayerTemplate):
    # 2 hidden layers of 64 neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = 64

class Mlp3Layer128Unit(Mlp3LayerTemplate):
    # 2 hidden layers of 512 neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = 128

class Mlp3Layer256Unit(Mlp3LayerTemplate):
    # 2 hidden layers of 256 neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = 256

class Mlp3Layer512Unit(Mlp3LayerTemplate):
    # 2 hidden layers of 256 neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = 512

class Mlp3Layer768Unit(Mlp3LayerTemplate):
    # 2 hidden layers of 256 neurons
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_hidden = 768

