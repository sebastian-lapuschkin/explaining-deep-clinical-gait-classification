# this file collects different training schedules for NN-type models
from .base import ModelTraining

####################################
# Training schemes for the NN models
####################################

class FullyConnectedTrainingDefault(ModelTraining):
    # this class provides the until now default training scheme for MLPs
    def train_model(self, x_train, y_train, x_val, y_val):
        print('training {} model (3 epochs, default setting)'.format(self.__class__.__name__))
        self.model.train(x_train, y_train, Xval=x_val, Yval=y_val, batchsize=5, lrate=0.005, status=500)  # train the model
        self.model.train(x_train, y_train, Xval=x_val, Yval=y_val, batchsize=5, lrate=0.001, status=500)  # slower training once the model has converged somewhat
        self.model.train(x_train, y_train, Xval=x_val, Yval=y_val, batchsize=5, lrate=0.0005, status=500) # one last epoch


class FullyConnectedTrainingIncreaseBatchSize(ModelTraining):
    # instead of only decreasing the lrate, we also increase the batch size and start with a larger batch size to begin with
    def train_model(self, x_train, y_train, x_val, y_val):
        print('training {} model (3 epochs, increasing batch size per epoch)'.format(self.__class__.__name__))
        self.model.train(x_train, y_train, Xval=x_val, Yval=y_val, batchsize=16, lrate=0.005, status=500)  # train the model
        self.model.train(x_train, y_train, Xval=x_val, Yval=y_val, batchsize=32, lrate=0.001, status=500)  # slower training once the model has converged somewhat
        self.model.train(x_train, y_train, Xval=x_val, Yval=y_val, batchsize=64, lrate=0.0005, status=500) # one last epoch
        #NOTE increasing the batch size might help, or not, or actually hurt performance, who knows. needs testing.

class FullyConnectedTrainingQuickTest(ModelTraining):
    # very short, rudimentary model training for testing
    def train_model(self, x_train, y_train, x_val, y_val):
        print('training {} model (quick test)'.format(self.__class__.__name__))
        self.model.train(x_train, y_train, iters=10)