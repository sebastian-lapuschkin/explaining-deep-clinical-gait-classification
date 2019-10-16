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

class Convolution2DArchitectureBase(ModelArchitecture):
    #Note: this class is abstract and provides the preprocessing for all CNN type models
    #Architectures need to be desingned in sublcasses

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_data(self, x_train, x_val, x_test,
                              y_train, y_val, y_test):
        """
        prepare data and labels as input to the model.
        add channel axis to multi-dim arrays
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

    def postprocess_relevance(self, *args):
        relevance = helpers.arrays_to_numpy(*args)
        #select previously added dummy axis explicitly
        return tuple([r[..., 0] for r in relevance])

    def assert_shapes(self, x_shape, y_shape):
        """ assert the shapes of input data for all fully connected models """
        assert len(x_shape) == 4, "Expected 4-dimensional shape tuple for 2d-CNN type models, but got x_shape={}".format(x_shape)
        assert len(y_shape) == 2, "Expected 2-dimensional shape tuple for 2d-CNN type models, but got y_shape={}".format(y_shape)


class Convolution1DArchitectureBase(ModelArchitecture):
    #base class for 1D-convolutions (ie reading the gait signal as a single time series)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape_processor = Flatten()
        self.input_shape_processor.to_numpy()

    def preprocess_data(self, x_train, x_val, x_test,
                              y_train, y_val, y_test):
        """
        prepare data and labels as input to the model.
        convert input multi-dim arrays into vectors, and add channel axis
        """

        # convert 2d data to 1d data then add a 1-dim spatial axis and a 1-dim channel axis to training inputs
        data = (self.input_shape_processor.forward(x_train)[..., None, None],
                self.input_shape_processor.forward(x_val)[..., None, None],
                self.input_shape_processor.forward(x_test)[..., None, None],
                y_train,
                y_val,
                y_test)

        if self.use_gpu:
            #move data to GPU if GPU execution is desired/possible
            data = helpers.arrays_to_cupy(*data)
        else:
            #otherwise, make sure the data is available to the CPU
            data = helpers.arrays_to_numpy(*data)

        return data


    def postprocess_relevance(self, *args):
        relevance = helpers.arrays_to_numpy(*args)
        #select previously added dummy axis explicitly for removal. then reshape to original signal again
        return tuple([self.input_shape_processor.backward(r[..., 0, 0]) for r in relevance])


    def assert_shapes(self, x_shape, y_shape):
        """ assert the shapes of input data for all fully connected models """
        assert len(x_shape) == 4, "Expected 4-dimensional shape tuple for 1d-CNN type models, but got x_shape={}".format(x_shape)
        assert len(y_shape) == 2, "Expected 2-dimensional shape tuple for 1d-CNN type models, but got y_shape={}".format(y_shape)


####################################
# 2D-CNN architecture specifications
####################################

class CnnA6(Convolution2DArchitectureBase, NeuralNetworkTrainingDefault):
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

class CnnA3(Convolution2DArchitectureBase, NeuralNetworkTrainingDefault):
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

class CnnAshort(Convolution2DArchitectureBase, NeuralNetworkTrainingDefault):
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

class CnnC3(Convolution2DArchitectureBase, NeuralNetworkTrainingDefault):
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

class CnnC6(Convolution2DArchitectureBase, NeuralNetworkTrainingDefault):
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


class CnnC3_3(Convolution2DArchitectureBase, NeuralNetworkTrainingDefault):
    # same as CnnC3, but with a stride of 3. in the vertical axis.
    # that means that left and and right body hemisphere are processed
    # non-overlappingly.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gpu = False

    def build_model(self, x_shape, y_shape):
        #samples are expected in shape 101 x 6 x 1 ie x_shape should be N x 101 x 6 x 1
        self.assert_shapes(x_shape, y_shape)
        assert x_shape[1:] == (101, 6, 1)
        n_classes = y_shape[1]

        h1 = Convolution(filtersize=(3,3,1,32), stride=(1,3))  # h1 output: 99 x 2 x 32 = 3072
        h2 = Convolution(filtersize=(3,2,32,32), stride=(1,1)) # h2 output: 97 x 1 x 32 = 3104
        h3 = Linear(3104,n_classes)
        self.model = Sequential([h1, Rect(), h2, Rect(), Flatten(), h3, SoftMax()])
        if not self.use_gpu:
            self.model.to_numpy()
        else:
            self.model.to_cupy()



####################################
# 1D-CNN architecture specifications
####################################

class Cnn1DC3(Convolution1DArchitectureBase, NeuralNetworkTrainingDefault):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self, x_shape, y_shape):
        #samples are expected in shape 606 x 1 ie x_shape should be N x 606 x 1
        self.assert_shapes(x_shape, y_shape)
        assert x_shape[1:] == (606, 1, 1)
        n_classes = y_shape[1]

        h1 = Convolution(filtersize=(3,1,1,16), stride=(1,1))  # h1 output: 604 x 1 x 16
        h2 = Convolution(filtersize=(3,1,16,24), stride=(1,1))  # h2 output: 602 x 1 x 24
        h3 = Convolution(filtersize=(4,1,24,48), stride=(2,1))  # h2 output: 301 x 1 x 48
        h4 = Convolution(filtersize=(4,1,48,48), stride=(3,1))  # h2 output: 99 x 1 x 48
        h5 = Linear(99*48, n_classes)
        self.model = Sequential([h1, Rect(),
                                 h2, Rect(),
                                 h3, Rect(),
                                 h4, Rect(),
                                 Flatten(),
                                 h5,
                                 SoftMax()])
        if not self.use_gpu:
            self.model.to_numpy()
        else:
            self.model.to_cupy()

class Cnn1DC6(Convolution1DArchitectureBase, NeuralNetworkTrainingDefault):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self, x_shape, y_shape):
        #samples are expected in shape 606 x 1 ie x_shape should be N x 606 x 1
        self.assert_shapes(x_shape, y_shape)
        assert x_shape[1:] == (606, 1, 1)
        n_classes = y_shape[1]

        h1 = Convolution(filtersize=(6,1,1,16), stride=(1,1))  # h1 output: 601 x 1 x 16
        h2 = Convolution(filtersize=(6,1,16,24), stride=(1,1))  # h2 output: 596 x 1 x 24
        h3 = Convolution(filtersize=(4,1,24,48), stride=(2,1))  # h2 output: 297 x 1 x 48
        h4 = Convolution(filtersize=(6,1,48,48), stride=(3,1))  # h2 output: 98 x 1 x 48
        h5 = Linear(98*48, n_classes)
        self.model = Sequential([h1, Rect(),
                                 h2, Rect(),
                                 h3, Rect(),
                                 h4, Rect(),
                                 Flatten(),
                                 h5,
                                 SoftMax()])
        if not self.use_gpu:
            self.model.to_numpy()
        else:
            self.model.to_cupy()