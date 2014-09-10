import numpy as np
import argparse
import theano.tensor
import h5py
from pylearn2.models.autoencoder import DenoisingAutoencoder
from pylearn2.model_extensions.model_extension import ModelExtension 
from pylearn2.models import Model
from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import DefaultDataSpecsMixin
from pylearn2.utils import safe_zip
from pylearn2.config import yaml_parse
from pylearn2.datasets.preprocessing import Preprocessor
from pylearn2.datasets.preprocessing import Pipeline
from pylearn2.datasets.preprocessing import Standardize
from pylearn2.datasets.hdf5 import HDF5Dataset, HDF5DatasetIterator
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils.iteration import safe_izip
from functools import wraps
from pylearn2.costs.autoencoder import GSNFriendlyCost
import types

class ICMC(DenoisingAutoencoder):
    """
    Inherits the Denoising Autoencoder class but makes a few key changes:
    * Initial weights may be nonnegative
    * Allows the passing of a ModelExtension
    """
    def __init__(self, corruptor, nvis, nhid, act_enc, act_dec, 
                 tied_weights=True, irange=0.05, 
                 rng=9001, weights_nonnegative=False, extensions=None):
        super(ICMC, self).__init__(corruptor, nvis, nhid, act_enc, act_dec, 
                                   tied_weights, irange, rng)
        Model.__init__(self, extensions)
        if weights_nonnegative:
            self._params[2].set_value((self._params[2]+irange/2.0).eval())
            if not tied_weights:
                self._params[3].set_value((self._params[3]+irange/2.0).eval())
        self.weights_nonnegative = weights_nonnegative        

def relu(x):
    """
    Rectified linear activation
    """
    return theano.tensor.switch(x > 0., x, 0.0)

class AsymWeightDecay(ModelExtension):
    """
    This extension allows an assymetrical weight decay. It is called after the
    updates are computed.

    TODO: Consider adding this to the model's censor_updates method instead
    """
    def __init__(self, decayP=0, decayN=1, decay_bias=False):
        self.__dict__.update(locals())
        del self.self
        
    def post_modify_updates(self, updates):
        update_weights = ['W', 'Wprime']
        if self.decay_bias:
            update_weights.extend(['vb', 'hb'])
        for k,v in updates.items():
            if k.name in update_weights:
                updates[k] = v - theano.tensor.where(v < 0, v * self.decayN, 
                                                     v * self.decayP)

class L1(DefaultDataSpecsMixin, Cost):
    """
    Class for computing the L1 regularization penalty on the activation
    of the hidden layer. Should encourage sparse activation.
    """
    def __init__(self, coeffs):
        assert isinstance(coeffs, list)
        self.coeffs = coeffs

    def expr(self, model, data):
        if hasattr(model, 'autoencoders'):
            assert len(model.autoencoders) == len(self.coeffs)
        self.get_data_specs(model)[0].validate(data)
        X = data
        if hasattr(model, 'autoencoders'):
            layers = model.autoencoders
        else:
            layers = [model]
        layer_costs = []
        current = data
        for layer, coeff, in safe_zip(layers, self.coeffs):
            current = layer.encode(current)
            cost = theano.tensor.abs_(current).sum(axis=1).mean()
            layer_costs.append(coeff * cost)

        assert theano.tensor.scalar() != 0.
        layer_costs = [cost_ for cost_ in layer_costs if cost_ != 0.]
        if len(layer_costs) == 0:
            return theano.tensor.as_tensor_variable(0.)
        else:
            total_cost = reduce(lambda x, y: x + y, layer_costs)
        total_cost.name = 'L1_ActCost'
        assert total_cost.ndim == 0
        return total_cost

class MeanRelativeSquaredReconstructionError(GSNFriendlyCost):
    """
    Do not use this class as a learning objective. It is only meant 
    for comparing already-trained model. The entire test set must be passed to
    as input to this evaluator.
    """
    @staticmethod
    def cost(a, b):
        val1 =  ((a - b) ** 2).sum(axis=1)
        val2 = ((a - a.mean(axis=0)) ** 2).sum(axis=1)
        return (val1/val2).mean()

class Normalize(Preprocessor):
    def __init__(self, global_max=False):
        self._global_max = global_max
        self._max = None

    def apply(self, dataset, can_fit=False):
        X = dataset.get_design_matrix()
        if can_fit:
            self._max = X.max() if self._global_max else X.max(axis=0)
        else:
            if self._max is None:
                raise ValueError("can_fit is False, but Normalize object "
                                 "has no stored max")
        new = X / self._max
        dataset.set_design_matrix(new)
        
    def invert(self, dataset):
        X = dataset.get_design_matrix()
        if self._max is None:
            raise ValueError("cNormalize object has no stored max")
        new = X * self._max
        dataset.set_design_matrix(new)
    
class LogScale(Preprocessor):
    def __init__(self):
        self._max = None

    def apply(self, dataset, can_fit=False):
        X = dataset.get_design_matrix()
        if can_fit:
            self._max = X.max(axis=0)
        else:
            if self._max is None:
                raise ValueError("can_fit is False, but Normalize object "
                                 "has no stored max")
        new = np.log10(np.clip(X, 0.0001, self._max))
        dataset.set_design_matrix(new)

    def invert(self, dataset):
        X = dataset.get_design_matrix()
        new = 10 ** X
        dataset.set_design_matrix(new)

class Standardize(Standardize):
    def invert(self, dataset):
        X = dataset.get_design_matrix()
        new = X * (self._std_eps + self._std) + self._mean
        dataset.set_design_matrix(new)

class Pipeline(Pipeline):
    def invert(self, dataset):
        for item in self.items[::-1]:
            item.invert(dataset)

# Currently not used because loading from disk onto GPU is too slow
class HDF5Dataset(HDF5Dataset):
    def __init__(self, filename, X=None, topo_view=None, y=None,
                 load_all=False, **kwargs):
        if 'preprocessor' in kwargs:
            if ('fit_preprocessor' in kwargs and 
                kwargs['fit_preprocessor'] is False) or ('fit_preprocessor' 
                                                         not in kwargs):
                self._preprocessor = kwargs['preprocessor']
                kwargs['preprocessor'] = None
        else:
            self._preprocessor = None
        self.load_all = load_all
        if h5py is None:
            raise RuntimeError("Could not import h5py.")
        self._file = h5py.File(filename)
        if X is not None:
            X = self.get_dataset(X, load_all)
        if topo_view is not None:
            topo_view = self.get_dataset(topo_view, load_all)
        if y is not None:
            y = self.get_dataset(y, load_all)
        DenseDesignMatrix.__init__(self, X=X, topo_view=topo_view, y=y,
                                   **kwargs)

    def iterator(self, *args, **kwargs):
        iterator = super(HDF5Dataset, self).iterator(*args, **kwargs)
        iterator.__class__ = HDF5DatasetIterator
        iterator._preprocessor = self._preprocessor
        return iterator

class HDF5DatasetIterator(HDF5DatasetIterator):
    def next(self):
        next_index = self._subset_iterator.next()

        # convert to boolean selection
        sel = np.zeros(self.num_examples, dtype=bool)
        sel[next_index] = True
        next_index = sel

        rval = []
        for data, fn in safe_izip(self._raw_data, self._convert):
            try:
                this_data = data[next_index]
            except TypeError:
                this_data = data[next_index, :]
            if fn:
                this_data = fn(this_data)
            if self._preprocessor is not None:
                d = DenseDesignMatrix(X=this_data)
                self._preprocessor.apply(d)
                this_data = d.get_design_matrix()
            assert not np.any(np.isnan(this_data))
            rval.append(this_data)
        rval = tuple(rval)
        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval    

def populate_autoencoder_yaml(args, n_layers, nvis):
    autoencoder_template = """!obj:icmc.ICMC {
            extensions: [!obj:icmc.AsymWeightDecay {
                decayP: %(decayP).3f,
                decayN: %(decayN).3f
            }],
            corruptor: !obj:pylearn2.corruption.GaussianCorruptor {
                stdev: %(stdev).3f
            },
            nvis: %(nvis)d,
            nhid: %(nhid)d,
            act_enc: %(act_enc)s,
            act_dec: %(act_dec)s,
            tied_weights: %(tied_weights)s,
            weights_nonnegative: %(weights_nonnegative)s,
            irange: %(irange).3f,
            rng: [1978, 9, 7],
        }"""
    autoencoder_yaml = "["
    sparsity = not (all([sp == 0.0 for sp in args['sparsity']]))
    if sparsity:
        coeffs_yaml = """[1.0, !obj:icmc.L1 {{
                   coeffs: ["""
    else:
        coeffs_yaml = ""
    yamls = []
    for i in xrange(n_layers):
        if sparsity:
            coeffs_yaml += "{0:.3f}"
        if i is not 0:
            nvis = args['units'][i-1]
        tied_weights = ('false' if args['tied_weights'][i] in
                        ['False', '0'] else args['tied_weights'][i])
        weights_nonnegative = ('false' if args['weights_nonnegative'][i] in 
                           ['False', '0'] else args['weights_nonnegative'][i])
        act_enc = ('!import icmc.relu' if args['encoding'][i] == 'relu'
                   else args['encoding'][i])
        act_dec = ('!import icmc.relu' if args['decoding'][i] == 'relu'
                   else args['decoding'][i])
        yamls.append(autoencoder_template % {
                'decayP':args['positive_decay'][i],
                'decayN':args['negative_decay'][i],
                'stdev':args['corruption'][i],
                'nvis':nvis,
                'nhid':args['units'][i],
                'act_enc':act_enc,
                'act_dec':act_dec,
                'tied_weights':tied_weights,
                'weights_nonnegative':weights_nonnegative,
                'irange': args['irange'][i]})
        if sparsity:
            coeffs_yaml = coeffs_yaml.format(args['sparsity'][i])
        if i != n_layers-1:
            yamls[i] += ", "
            if sparsity:
                coeffs_yaml += ", "
    autoencoder_yaml += "".join(yamls)
    autoencoder_yaml += "]"
    if sparsity:
        coeffs_yaml += "]}]"
    return autoencoder_yaml,coeffs_yaml

def populate_yaml(args, n_layers):
    print("Inserting variables into template: "+args['yaml-template'])
    with open(args['yaml-template'], 'r') as f:
        yaml = f.read()
    base = '/global/data/casey/sarroff/projects/deepAE/data'

    kwargs = {} 
    kwargs['train_fn'] = (base+'/feat/'+args['feature']+'/'+
                          args['feature']+'_train.h5')
    print("Opening "+kwargs['train_fn'])
    h5file = h5py.File(kwargs['train_fn'], 'r')
    nvis = h5file['data'].shape[1]
    h5file.close()
    kwargs['preproc_pkl'] = args['preproc-pkl']
    kwargs['test_fn'] = (base+'/feat/'+args['feature']+'/'+
                         args['feature']+'_test.h5')
    kwargs['val_fn'] = (base+'/feat/'+args['feature']+'/'+
                        args['feature']+'_val.h5')
    autoencoder_yaml,coeffs_yaml = (populate_autoencoder_yaml
                                    (args, n_layers, nvis))
    kwargs['autoencoders'] = autoencoder_yaml
    kwargs['coeffs'] = coeffs_yaml
    kwargs['learning_rate'] = args['learning-rate']
    kwargs['save_path'] = (args['save-directory']+'/'+args['save-prefix']+
                           '_'+args['feature']+'.pkl')
    kwargs['save_path_best'] = (args['save-directory']+'/'+args['save-prefix']+
                                '_'+args['feature']+'_best.pkl')
    yaml = yaml % kwargs
    return yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('save-directory', help="Location to save model")
    parser.add_argument('save-prefix', help="Prefix for model name")
    parser.add_argument('yaml-template', help="Location of template")
    parser.add_argument('preproc-pkl', help="Location of pickled preprocessor")
    parser.add_argument('feature', choices=['stft', 'cqft_3bpo', 
                                            'cqft_12bpo', 'stft_orig'],
                        help="Which low level feature to use")
    parser.add_argument('learning-rate', type=float,
                        help="The initial global learning rate")
    parser.add_argument('-u', '--units', nargs='+', type=int,
                        help="An integer for the number of units in "+
                        "each hidden layer")
    parser.add_argument('-e', '--encoding', nargs='+', 
                        choices=['sigmoid', 'relu', 'linear'],
                        help="Encoding function for each layer")
    parser.add_argument('-d', '--decoding', nargs='+', 
                        choices=['sigmoid', 'relu', 'linear'],
                        help="decoding function for each layer")
    parser.add_argument('-t', '--tied-weights', nargs='+',
                        help="List of booleans indicating if tied weights"+
                        "should be used", default=["true"])
    parser.add_argument('-w', '--weights-nonnegative', nargs='+', 
                        help="List of booleans indicating if nonnegative "+
                        "weights should be used", default=["true"])
    parser.add_argument('-i', '--irange', nargs='+', type=float,
                        help="Range for weight initialization. If "+
                        "weights-nonnegative is set to True, the initial"+
                        "weights are also positive. Otherwise initial weights"+
                        "are centered around 0", required=True)
    parser.add_argument('-p', '--positive-decay', type=float, nargs='+',
                        default=[0.0],
                        help="Positive weight decay")
    parser.add_argument('-n', '--negative-decay', type=float, nargs='+',
                        default=[0.0],
                        help="Negative weight decay")
    parser.add_argument('-c', '--corruption', type=float, nargs='+',
                        default=[0.0],
                        help="Standard deviation of gaussian noise")
    parser.add_argument('-s', '--sparsity', type=float, nargs='+',
                        default=[0.0],
                        help="Sparsity coefficient for layer activations")
    args = vars(parser.parse_args())
    n_layers = len(args['units'])
    for k,v in args.items():
        if k not in ['feature', 'learning-rate', 'yaml-template', 
                     'save-directory', 'save-prefix', 'preproc-pkl']:
            assert len(v) == n_layers
                
    yaml = populate_yaml(args, n_layers)
    with open(args['save-directory']+'/'+args['save-prefix']+'.yaml', 
              'w') as f:
        f.write(yaml)
    train = yaml_parse.load(yaml)
    train.main_loop()

