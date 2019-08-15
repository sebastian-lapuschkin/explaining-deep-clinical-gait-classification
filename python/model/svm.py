from .base import Model
import sklearn
import numpy
from modules import Linear, Sequential



class LinearSVM(Model):

    def __init__(self, *args, **kwargs):
        """
        Initializes model and populates lists of functions to execute.
        """
        super().__init__(*args, **kwargs)
        self.model = None #variable for storing a loaded/trained model (lrp toolbox format: modules.Sequential).
        self.use_gpu = False

    def _build_svm_model(self):
        """
        specialized SVM classes overwrite this.
        """
        self.model = sklearn.svm.LinearSVC()

    def _sanity_check_model_conversion(self, svm_model, x_val):
        y_pred_svm = svm_model.decision_function(x_val)
        y_pred_nn  = self.model.forward(x_val)

        rtol = 1e-7
        if y_pred_nn.shape[1] == 2:
            numpy.testing.assert_allclose(y_pred_svm, -y_pred_nn[:,0], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
            numpy.testing.assert_allclose(y_pred_svm, y_pred_nn[:,1] , rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
        else:
            numpy.testing.assert_allclose(y_pred_svm, y_pred_nn, rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal!')


    def train_model(self, x_train, y_train, x_val, y_val):
        # train model using sklearn
        print('training svm model')
        self._build_svm_model()
        self.model.fit(x_train, y_train)

        #convert to linear NN
        print('converting svm model to linear NN')
        W = self.model.coef_.T
        B = self.model.intercept_

        if numpy.unique(y_train).size == 2:
            linear_layer = Linear(W.shape[0], 2)
            linear_layer.W = numpy.concatenate([-W, W], axis=1)
            linear_layer.B = numpy.concatenate([-B, B], axis=0)
        else:
            linear_layer = Linear(*(W.shape))
            linear_layer.W = W
            linear_layer.B = B

        svm_model = self.model
        self.model = Sequential([linear_layer])
        if not self.use_gpu: self.model.to_numpy()

        #sanity check model conversion
        self._sanity_check_model_conversion(svm_model, x_val)
        print('model conversion sanity check passed')


    def preprocess_data(self, x_train, x_val, x_test,
                              y_train, y_val, y_test):
        """
        prepare data and labels as input to the model.
        default: do nothing, except moving data to preferred device
        overwrite to specify
        """
        x_train = numpy.reshape(x_train, [x_train.shape[0], -1])
        x_val = numpy.reshape(x_val, [x_val.shape[0], -1])
        x_test = numpy.reshape(x_test, [x_test.shape[0], -1])

        # only training labels need to be altered, since testing will be done on the svm-reformulation
        # test and validation labels need to be preserved
        y_train = numpy.argmax(y_train, axis = 1)

        return (x_train, x_val, x_test, y_train, y_val, y_test)


    def postprocess_data(self, *args, **kwargs):
        """
        prepare data and labels as after processing by the model to the model.
        e.g. in order to restore the original shape of the data.
        default: do nothing, except moving data back to cpu
        overwrite to specify
        """
        return None