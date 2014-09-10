import theano.tensor as T
import numpy as np

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.models.autoencoder import Autoencoder
from pylearn2.training_algorithms.learning_rule import LearningRule
from pylearn2.utils import sharedX


# Create extension of denoising autoencoder class?
# we need:
#  1. Method to update learning rate (might be a learning rule)
#  2. Method to decay weights (might be update callback)
#  3. Method to update activation function (might be update callback)
#      - Make class for this type of activation function

class NNSAE(Autoencoder)
"""
Inherits the Autoencoder class but makes a few key changes:
   * Weights are initialized in nonnegative range
   * Weights are tied
"""
    def __init__(self, nvis, nhid, act_enc, act_dec, irange=0.05, rng=9001):
        assert nvis > 0, "Number of visible units must be non-negative"
        assert nhid > 0, "Number of hidden units must be positive"

        self.input_space = VectorSpace(nvis)
        self.output_space = VectorSpace(nhid)

        # Save a few parameters needed for resizing
        self.nhid = nhid
        self.irange = irange
        self.tied_weights = True
        self.rng = make_np_rng(rng, which_method="randn")
        self._initialize_hidbias()
        if nvis > 0:
            self._initialize_visbias(nvis)
            self._initialize_weights(nvis)
        else:
            self.visbias = None
            self.weights = None

        seed = int(self.rng.randint(2 ** 30))

        # why a theano rng? should we remove it?
        self.s_rng = make_theano_rng(seed, which_method="uniform")

        self.w_prime = self.weights.T

        def _resolve_callable(conf, conf_attr):
            if conf[conf_attr] is None or conf[conf_attr] == "linear":
                return None
            # If it's a callable, use it directly.
            if hasattr(conf[conf_attr], '__call__'):
                return conf[conf_attr]
            elif (conf[conf_attr] in globals()
                  and hasattr(globals()[conf[conf_attr]], '__call__')):
                return globals()[conf[conf_attr]]
            elif hasattr(tensor.nnet, conf[conf_attr]):
                return getattr(tensor.nnet, conf[conf_attr])
            elif hasattr(tensor, conf[conf_attr]):
                return getattr(tensor, conf[conf_attr])
            else:
                raise ValueError("Couldn't interpret %s value: '%s'" %
                                    (conf_attr, conf[conf_attr]))

        self.act_enc = _resolve_callable(locals(), 'act_enc')
        self.act_dec = _resolve_callable(locals(), 'act_dec')
        self._params = [
            self.visbias,
            self.hidbias,
            self.weights,
        ]

    def _initialize_weights(self, nvis, rng=None, irange=None):
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        self.weights = sharedX(
            rng.rand(nvis, self.nhid) * irange,
            name='W',
            borrow=True
        )

class ScaledLogistic(Object):
    def __init__(self, init_a, init_b, n):
        self.a = sharedX(init_a * np.ones(n))
        self.b = sharedX(init_b * np.ones(n))

    def __call__(self, x):
        return 1/(1+T.exp(-1*self.a*x-self.b))

class NNSAECost(DefaultDataSpecMixin, Cost):
    def expr(self, model, data, **kwargs):
        X = data
        X_hat = model.reconstruct(X)
        return ((X - X_hat) ** 2).sum(axis=1).mean()

    def set_scaled_lr(self):
        pass

    def set_scaled_logistic_params(self):
        pass

class NNSAELearner(LearningRule):
    def __init__(self, model, lrateRO=0.01, regRO=0.0002, lrateIP=0.001, meanIP=0.2):
        self.lrateRO = lrateRO
        self.regRO = regRO
        self.lrateIP = lrateIP
        self.meanIP = meanIP

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        pass

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        h = model.
        lrate = self.lrateRO/(self.regRO + sum(power(net.h, 2)));

        pass

    def update_lrateRO(self):
        
