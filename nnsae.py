from itertools import izip
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from theano import function
from theano import config
import numpy as np
from pylearn2.models.autoencoder import Autoencoder
from pylearn2 import utils
from pylearn2.utils import sharedX
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.costs.cost import FixedVarDescr, Cost, DefaultDataSpecsMixin

class NNSAE(Autoencoder):
    """
    Inherits the Autoencoder class but makes a few key changes:
    * Weights are initialized in nonnegative range [0, 0.05]
    * Weights are tied by default
    * Adds the ScaledLogistic paramaters if necessary
    """
    def __init__(self, nvis, nhid, act_enc, act_dec, 
                 tied_weights=True, irange=0.05, rng=9001,
                 lrateRO=0.01, regRO=0.0002,
                 decayP=0, decayN=1,
                 lrateIP=0.001, meanIP=0.2):
        super(NNSAE, self).__init__(nvis, nhid, act_enc, act_dec, tied_weights,
                                    irange, rng)

        self.lrateRO = lrateRO
        self.regRO = regRO
        self.decayP = decayP
        self.decayN = decayN
        self.lrateIP = sharedX(lrateIP)
        self.meanIP = meanIP
        self.lrate = sharedX(lrateRO, 'lrate')

        if isinstance(self.act_enc, ScaledLogistic):
            self.act_enc.a.name = 'logistic_a'
            self.act_enc.b.name = 'logistic_b'
            self._params.append(self.act_enc.a)
            self._params.append(self.act_enc.b)
            self.logistic_a = self.act_enc.a
            self.logistic_b = self.act_enc.b

    def DNU__hidden_input(self, x):
        return T.dot(x, self.weights)

    def DNU_decode(self, hiddens):
        if self.act_dec is None:
            act_dec = lambda x: x
        else:
            act_dec = self.act_dec
        if isinstance(hiddens, T.Variable):
            return act_dec(T.dot(hiddens, self.w_prime))
        else:
            return [self.decode(v) for v in hiddens]

    def get_lr_scalers(self):
        rval = OrderedDict()
        for p in self.get_params():
            if p.name in ['W', 'Wprime', 'hb', 'vb']:
                rval[p] = self.lrate
            if p.name in ['logistic_a', 'logistic_b']:
                rval[p] = self.lrateIP
        return rval
    
    def update_lrateRO(self, data):
        lrate = self.lrateRO / (self.regRO + 
                                (self.encode(data) ** 2).sum(axis=1).mean())
        return lrate

    def get_scaled_logistic_updates(self, data):
        encoded = self.encode(data)
        pre_encoded = self._hidden_input(data)
        delta_b = (1 - (2 + 1/self.meanIP) * encoded + 
                   1/self.meanIP * encoded**2)
        delta_a = 1/self.logistic_a + pre_encoded*delta_b
        return (delta_a.mean(axis=0), delta_b.mean(axis=0))

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

    def _initialize_w_prime(self, nvis, rng=None, irange=None):
        assert not self.tied_weights, (
            "Can't initialize w_prime in tied weights model; "
            "this method shouldn't have been called"
        )
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        self.w_prime = sharedX(
            rng.rand(self.nhid, nvis) * irange,
            name='Wprime',
            borrow=True
        )

class ScaledLogistic(object):
    def __init__(self, n, init_a=1, init_b=-3):
        self.a = sharedX(init_a * np.ones(n, dtype=config.floatX))
        self.b = sharedX(init_b * np.ones(n, dtype=config.floatX))

    def __call__(self, x):
        return 1/(1+T.exp(-1*self.a*x-self.b))

class NNSAEMSE(MeanSquaredReconstructionError):
    def get_fixed_var_descr(self, model, data):
        self.get_data_specs(model)[0].validate(data)
        rval = FixedVarDescr()
        lrate = model.update_lrateRO(data)
        updates = OrderedDict()
        updates[model.lrate] = lrate
        rval.on_load_batch = [utils.function([data], updates=updates)]
        return rval

    def get_monitoring_channels(self, model, data):
        self.get_data_specs(model)[0].validate(data)
        rval = OrderedDict()
        rval['lrate'] = model.lrate
        return rval
    
    def get_gradients(self, model, data, **kwargs):
        cost = self.expr(model=model, data=data, **kwargs)
        params = list(model.get_params())
        grads = T.grad(cost, params, disconnected_inputs='ignore')
        gradients = OrderedDict(izip(params, grads))
        updates = OrderedDict()
        return gradients, updates
        
class NNSAEWeightDecay(object):
    def __init__(self, model):
        updates = OrderedDict()
        for p in model.get_params():
            # if p.name in ['W', 'Wprime']:
            if p.name in ['W', 'Wprime', 'hb', 'vb']:
                updates[p] = p - T.where(p < 0, p * model.decayN, 
                                         p * model.decayP)
        self.decay = function([], updates=updates)

    def __call__(self, algorithm):
        self.decay()

class NNSAEScaledLogisticSparsity(DefaultDataSpecsMixin, Cost):
    def expr(self, model, data, *args, **kwargs):
        return None

    def get_gradients(self, model, data, **kwargs):
        self.get_data_specs(model)[0].validate(data)
        delta_a, delta_b = model.get_scaled_logistic_updates(data)
        gradients = OrderedDict()
        for p in model.get_params():
            if p.name is 'logistic_a':
                gradients[p] = -delta_a
            if p.name is 'logistic_b':
                gradients[p] = -delta_b
        updates = OrderedDict()
        return gradients, updates

    def DNU_get_monitoring_channels(self, model, data, **kwargs):
        self.get_data_specs(model)[0].validate(data)
        rval = OrderedDict()
        for p in model.get_params():
            if p.name is 'logistic_a':
                for i in xrange(model.get_output_space().dim):
                    rval['logistic_a_%i'%i] = p[i] 
            if p.name is 'logistic_b':
                for i in xrange(model.get_output_space().dim):
                    rval['logistic_b_%i'%i] = p[i] 
        return rval
