# module for collecting model definitions.
from abc import ABC, abstractmethod #abstract base class
import importlib.util as imp
import helpers
import os
from modules import Sequential, Convolution, Linear
import model_io

class ModelTraining(ABC):
    """
    this class defines the training protocol of a model,
    together with the required pre-and post processing steps for the data,
    and a (probably for all models) standardized evaluation scheme.

    This class is intended to be used in a multiple inheritance scheme together with ModelArchitecture,
    where ModelArchitecture should be inherited from first and ModelTrainer second.

    This should allow an easy replacement of model trainng regime.
    """
    def __init__(self, *args, **kwargs):
        self.model = None #ModelTraining will also require access to self.model for training it. this variable is shared with ModelArchitecture

    @abstractmethod
    def train_model(self, x_train, y_train, x_val, y_val, force_device):
        """
        train the model.
        requires the model to be built before training.
        convert to lrp-toolbox-dnn after trainng: modules.Sequential (required!).
        overwrite it during inheritance to specify.
        force_device may (if supported) force the training execution either on cpu or gpu.
        """
        raise NotImplementedError()


class ModelArchitecture(ABC):
    """
    this class defines the architecture of the model,
    together with the required pre-and post processing steps for the data,
    and a (probably for all models) standardized evaluation scheme.

    This class is intended to be used in a multiple inheritance scheme together with ModelTrainer,
    where ModelArchitecture should be inherited from first and ModelTrainer second
    """

    def __init__(self, root_dir, data_name, target_name, split_index):
        """
        Initializes model and populates lists of functions to execute.
        """
        self.root_dir       = root_dir
        self.data_name      = data_name
        self.target_name    = target_name
        self.split_index    = split_index

        self.model = None #variable for storing a loaded/trained model (lrp toolbox format: modules.Sequential).
        self.use_gpu = imp.find_spec("cupy") is not None # use the GPU if desired/possible. default=True if cupy is available

    @abstractmethod
    def build_model(self, x_shape, y_shape):
        """
        build the model.
        overwrite this for impementing a model architecture.
        probably best called at the beginning of the overwritten train_model
        method, eg if information about data dimensions are required
        passable parameters may depend on the implementing classes

        Parameters:
        -----------

        x_shape: tuple. the shape of a data batch, usually (batchsize, data-axes....)

        y_shape: tuple. the shape of a label batch, usually (batchsize, number_of_classes)
        """
        pass

    def name(self):
        """
        generate a model name based on the names of
        the model itself, the data, the prediction target and the split index

        Parameters:
        -----------
            see Model.path_files
        """
        return '{}/{}/{}'.format(self.target_name, self.data_name, self.__class__.__name__)

    def path_dir(self):
        """
        generate path to the directory containing the model-related files based on the names of
        the model itself, the data, the prediction target and the split index

        Parameters:
        -----------
            see Model.path_files
        """
        return '{}/{}/part-{}'.format(self.root_dir, self.name(), self.split_index)

    def path_files(self):
        """
        generate path to the model-related files based on the names of
        the model itself, the data, the prediction target and the split index

        Parameters:
        -----------

        data_name: str - the name of the data/feature used for training

        target_name: str - the name of the prediction target (ie problem)

        split_index: int - the data split id of this model, ie. the split reserved for testing.

        Returns:
        --------
        model_file_path: str
        scores_file_path: str
        outputs_file_path: str
        """
        dir_name = self.path_dir()
        model_file_path     = dir_name + '/model.txt'   # model description
        scores_file_path    = dir_name + '/scores.txt'  # model scores, e.g. accuracy and loss
        outputs_file_path   = dir_name + '/outputs.mat' # model eval stats and relevances

        return model_file_path, scores_file_path, outputs_file_path

    def exists(self, explicit_path=None):
        """
        checks if pretrained model exists on disk
        model path is determined via data_name, target_name or split_index.
        can be overwritten when explicit_path is given
        """

        if explicit_path is not None:
            path_to_model = explicit_path
        else:
            path_to_model = self.path_files()[0]

        return os.path.isfile(path_to_model)

    def load_model(self, explicit_path=None):
        """
        load model from disk (model file should exist)
        model path is determined via data_name, target_name or split_index.
        can be overwritten when explicit_path is given
        """

        if explicit_path is not None:
            path_to_model = explicit_path
        else:
            path_to_model = self.path_files()[0]

        assert self.exists(explicit_path=path_to_model), "No file found at {}".format(path_to_model)
        self.model = model_io.read(path_to_model) # caution! prefers GPU, if available!
        if self.use_gpu:
            self.model.to_cupy()
        else:
            self.model.to_numpy()

    def save_model(self, explicit_path=None):
        """
        save the model to disk
        model path is determined via data_name, target_name or split_index.
        can be overwritten when explicit_path is given
        """

        if explicit_path is not None:
            path_to_model = explicit_path
        else:
            path_to_model = self.path_files()[0]

        helpers.ensure_dir_exists(os.path.dirname(path_to_model))
        model_io.write(self.model, path_to_model, fmt='txt')
        if self.use_gpu: self.model.to_cupy()

    def evaluate_model(self, x_test, y_test, force_device=None, lower_upper = None):
        """
        test model and computes relevance maps

        Parameters:
        -----------

        x_test: array - shaped such that it is ready for consumption by the model

        y_test: array - expected test labels

        target_shape: list or tuple - the target output shape of the test data and relevance maps.

        force_device: str - (optional) force execution of the evaluation either on cpu or gpu.
            accepted values: "cpu", "gpu" respectively. None does nothing.

        lower_upper: (array of float, array of float) - (optional): lower and upper bounds of the inputs, for LRP_zB.
            automagically inferred from x_test.
            arrays should match the feature dimensionality of the inputs, including broadcastable axes.
            e.g. if x_test is shaped (N, featuredims), then the bounds should be shaped (1, featuredims)

        Returns:
        --------

        results, packed in dictionary, as numpy arrays
        """

        assert isinstance(self.model, Sequential), "self.model should be modules.sequential.Sequentialm but is {}. ensure correct type by converting model after training.".format(type(self.model))
        # remove the softmax output of the model.
        # this does not change the ranking of the outputs but is required for most LRP methods
        # self.model is required to be a modules.Sequential
        results = {} #prepare results dictionary

        #force model to specific device, if so desired.
        x_test, y_test = helpers.force_device(self, (x_test, y_test), force_device)

        print('...forward pass for {} test samples for model performance eval'.format(x_test.shape[0]))
        y_pred = self.model.forward(x_test)

        #evaluate accuracy and loss on cpu-copyies of prediction vectors
        y_pred_c, y_test_c = helpers.arrays_to_numpy(y_pred, y_test)
        results['acc']    = helpers.accuracy(y_test_c, y_pred_c)
        results['loss_l1'] = helpers.l1loss(y_test_c, y_pred_c)
        results['y_pred'] = y_pred_c

        #NOTE: drop softmax layer AFTER forward for performance measures to obtain competetive loss values
        self.model.drop_softmax_output_layer()

        #NOTE: second forward pass without softmax for relevance computation
        print('...forward pass for {} test samples (without softmax) for LRP'.format(x_test.shape[0]))
        y_pred = self.model.forward(x_test) # this is also a requirement for LRP

        # prepare initial relevance vectors for actual class and dominantly predicted class, on model-device (gpu or cpu)
        R_init_act = y_pred * y_test #assumes y_test to be binary matrix

        y_dom = (y_pred == y_pred.max(axis=1, keepdims=True))
        R_init_dom = y_pred * y_dom #assumes prediction maxima are unique per sample


        # compute epsilon-lrp for all model layers
        for m in self.model.modules: m.set_lrp_parameters(lrp_var='epsilon', param=1e-5)
        print('...lrp (eps) for actual classes')
        results['R_pred_act_epsilon'] = self.model.lrp(R_init_act)

        print('...lrp (eps) for dominant classes')
        results['R_pred_dom_epsilon'] = self.model.lrp(R_init_dom)

        # eps + zB (lowest convolution/flatten layer) for all models here.

        # infer lower and upper bounds from data, if not given
        if not lower_upper:
            print('    ...inferring per-channel lower and upper bounds for zB from test data. THIS IS PROBABLY NOT OPTIMAL')
            lower_upper = helpers.get_channel_wise_bounds(x_test)
        else:
            print('    ...using input lower and upper bounds for zB')
        if self.use_gpu:
            lower_upper = helpers.arrays_to_cupy(*lower_upper)
        else:
            lower_upper = helpers.arrays_to_numpy(*lower_upper)

        # configure the lowest weighted layer to be decomposed with zB. This should be the one nearest to the input.
        # We are not just taking the first layer, since the MLP models are starting with a Flatten layer for reshaping the data.
        for m in self.model.modules:
            if isinstance(m, (Linear, Convolution)):
                m.set_lrp_parameters(lrp_var='zB', param=lower_upper)
                break

        print('...lrp (eps + zB) for actual classes')
        results['R_pred_act_epsilon_zb'] = self.model.lrp(R_init_act)

        print('...lrp (eps + zB) for dominant classes')
        results['R_pred_dom_epsilon_zb'] = self.model.lrp(R_init_dom)


        # compute CNN composite rules, if model has convolution layes
        has_convolutions = False
        for m in self.model.modules:
            has_convolutions = has_convolutions or isinstance(m, Convolution)

        if has_convolutions:
            # convolution layers found.

            # epsilon-lrp with flat decomposition in the lowest convolution layers
            # process lowest convolution layer with FLAT lrp
            # for "normal" cnns, this should overwrite the previously set zB rule
            for m in self.model.modules:
                if isinstance(m, Convolution):
                    m.set_lrp_parameters(lrp_var='flat')
                    break

            print('...lrp (eps+flat) for actual classes')
            results['R_pred_act_epsilon_flat'] = self.model.lrp(R_init_act)

            print('...lrp (eps+flat) for dominant classes')
            results['R_pred_dom_epsilon_flat'] = self.model.lrp(R_init_dom)




            # preparing alpha2beta-1 for those layers
            for m in self.model.modules:
                if isinstance(m, Convolution):
                    m.set_lrp_parameters(lrp_var='alpha', param=2.0)

            print('...lrp (composite:alpha=2) for actual classes')
            results['R_pred_act_composite_alpha2'] = self.model.lrp(R_init_act)

            print('...lrp (composite:alpha=2) for dominant classes')
            results['R_pred_dom_composite_alpha2'] = self.model.lrp(R_init_dom)

            # process lowest convolution layer with FLAT lrp
            for m in self.model.modules:
                if isinstance(m, Convolution):
                    m.set_lrp_parameters(lrp_var='flat')
                    break

            print('...lrp (composite:alpha=2+flat) for actual classes')
            results['R_pred_act_composite_alpha2_flat'] = self.model.lrp(R_init_act)

            print('...lrp (composite:alpha=2+flat) for dominant classes')
            results['R_pred_dom_composite_alpha2_flat'] = self.model.lrp(R_init_dom)


            #process lowest convolution layer with zB lrp
            for m in self.model.modules:
                if isinstance(m, Convolution):
                    m.set_lrp_parameters(lrp_var='zB', param=lower_upper)
                    break

            print('...lrp (composite:alpha=2+zB) for actual classes')
            results['R_pred_act_composite_alpha2_zB'] = self.model.lrp(R_init_act)

            print('...lrp (composite:alpha=2+zB) for dominant classes')
            results['R_pred_dom_composite_alpha2_zB'] = self.model.lrp(R_init_dom)




            # switching alpha1beta0 for those layers
            for m in self.model.modules:
                if isinstance(m, Convolution):
                    m.set_lrp_parameters(lrp_var='alpha', param=1.0)

            print('...lrp (composite:alpha=1) for actual classes')
            results['R_pred_act_composite_alpha1'] = self.model.lrp(R_init_act)

            print('...lrp (composite:alpha=1) for dominant classes')
            results['R_pred_dom_composite_alpha1'] = self.model.lrp(R_init_dom)

            # process lowest convolution layer with FLAT lrp
            for m in self.model.modules:
                if isinstance(m, Convolution):
                    m.set_lrp_parameters(lrp_var='flat')
                    break

            print('...lrp (composite:alpha=1+flat) for actual classes')
            results['R_pred_act_composite_alpha1_flat'] = self.model.lrp(R_init_act)

            print('...lrp (composite:alpha=1+flat) for dominant classes')
            results['R_pred_dom_composite_alpha1_flat'] = self.model.lrp(R_init_dom)


            #process lowest convolution layer with zB lrp
            for m in self.model.modules:
                if isinstance(m, Convolution):
                    m.set_lrp_parameters(lrp_var='zB', param=lower_upper)
                    break

            print('...lrp (composite:alpha=1+zB) for actual classes')
            results['R_pred_act_composite_alpha1_zB'] = self.model.lrp(R_init_act)

            print('...lrp (composite:alpha=1+zB) for dominant classes')
            results['R_pred_dom_composite_alpha1_zB'] = self.model.lrp(R_init_dom)


        print('...copying collected results to CPU and reshaping if necessary')
        for key in results.keys():
            tmp = helpers.arrays_to_numpy(results[key])[0]
            if key.startswith('R'):
                tmp = self.postprocess_relevance(tmp)[0]
            results[key] = tmp

        return results


    def preprocess_data(self, x_train, x_val, x_test,
                              y_train, y_val, y_test):
        """
        prepare data and labels as input to the model.
        default: do nothing, except moving data to preferred device
        overwrite to specify
        caution return data in order as given!
        """
        data = (x_train, x_val, x_test, y_train, y_val, y_test)
        if self.use_gpu:
            data = helpers.arrays_to_cupy(*data)
        return data

    def postprocess_relevance(self, *args, **kwargs):
        """
        postprocess relvance values by bringing them into a shape aligned to the data.
        """
        return helpers.arrays_to_numpy(*args)

    def to_gpu(self, *args, **kwargs):
        """
        attempts to transfer the model to gpu
        """
        self.model.to_cupy()
        self.use_gpu = True

    def to_cpu(self, *args, **kwargs):
        """
        attempts to transfer the model to cpu
        """
        self.model.to_numpy()
        self.use_gpu = False