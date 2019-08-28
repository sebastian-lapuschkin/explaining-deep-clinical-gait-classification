from .base import ModelArchitecture, ModelTraining
import sklearn.svm
import numpy
import helpers
from modules import Linear, Sequential, Flatten
from abc import abstractmethod


###########
# Abstract base class for Linear SVMs.
# parameterized classes below.
###########

class SvmLinearTemplate(ModelArchitecture, ModelTraining):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gpu = False #force CPU execution

    def _sanity_check_model_conversion(self, svm_model, nn_model, x_val):

        #use flatten layer to vectorize multi-dim array
        flatten = Flatten()
        flatten.to_numpy()

        y_pred_svm = svm_model.decision_function(flatten.forward(x_val))
        y_pred_nn  = nn_model.forward(x_val)

        rtol = 1e-7
        if y_pred_nn.shape[1] == 2:
            numpy.testing.assert_allclose(y_pred_svm, -y_pred_nn[:,0], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
            numpy.testing.assert_allclose(y_pred_svm, y_pred_nn[:,1] , rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
        else:
            numpy.testing.assert_allclose(y_pred_svm, y_pred_nn, rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal!')


    def train_model(self, x_train, y_train, x_val, y_val, *args, **kwargs):
        # train model using sklearn
        print('training {} model'.format(self.__class__.__name__))

        #use flatten layer to vectorize multi-dim array
        flatten = Flatten()
        flatten.to_numpy()

        x_train = flatten.forward(x_train)
        self.model.fit(x_train, y_train)
        self.model = self._convert_to_nn(self.model, y_train, x_val)


    def _convert_to_nn(self, svm_model, y_train, x_val):
        #convert to linear NN
        print('converting {} model to linear NN'.format(self.__class__.__name__))
        W = svm_model.coef_.T
        B = svm_model.intercept_

        if numpy.unique(y_train).size == 2:
            linear_layer = Linear(W.shape[0], 2)
            linear_layer.W = numpy.concatenate([-W, W], axis=1)
            linear_layer.B = numpy.concatenate([-B, B], axis=0)
        else:
            linear_layer = Linear(*(W.shape))
            linear_layer.W = W
            linear_layer.B = B

        svm_model = self.model
        nn_model = Sequential([Flatten(), linear_layer])
        if not self.use_gpu: nn_model.to_numpy()

        #sanity check model conversion
        self._sanity_check_model_conversion(svm_model, nn_model, x_val)
        print('model conversion sanity check passed')
        return nn_model


    def preprocess_data(self, x_train, x_val, x_test,
                              y_train, y_val, y_test):
        """
        prepare data and labels as input to the model.
        convert input multi-dim arrays into vectors
        """

        # only training labels need to be altered, since testing will be done on the svm-reformulation
        # test and validation labels need to be preserved
        y_train = numpy.argmax(y_train, axis = 1)

        return (x_train, x_val, x_test, y_train, y_val, y_test)


    def postprocess_relevance(self, *args, **kwargs):
        relevance = helpers.arrays_to_numpy(*args)
        return relevance



###########
# Svm model parameterization examples, with different regularizer weightings C and L2 normalization
###########

class SvmLinearL2C1e0(SvmLinearTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self, *args, **kwargs):
        self.model =  sklearn.svm.LinearSVC(penalty='l2', C=1e0)

class SvmLinearL2C1em1(SvmLinearTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self, *args, **kwargs):
        self.model =  sklearn.svm.LinearSVC(penalty='l2', C=1e-1)

class SvmLinearL2C1em2(SvmLinearTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self, *args, **kwargs):
        self.model =  sklearn.svm.LinearSVC(penalty='l2', C=1e-2)

class SvmLinearL2C5em2(SvmLinearTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self, *args, **kwargs):
        self.model =  sklearn.svm.LinearSVC(penalty='l2', C=5e-2)

class SvmLinearL2C1em3(SvmLinearTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self, *args, **kwargs):
        self.model =  sklearn.svm.LinearSVC(penalty='l2', C=1e-3)

class SvmLinearL2C5em3(SvmLinearTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self, *args, **kwargs):
        self.model =  sklearn.svm.LinearSVC(penalty='l2', C=5e-3)

class SvmLinearL2C1ep1(SvmLinearTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self, *args, **kwargs):
        self.model =  sklearn.svm.LinearSVC(penalty='l2', C=1e+1)