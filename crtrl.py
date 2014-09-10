from pylearn2.models import Model
from pylearn2.space import VectorSpace
from pylearn2.utils.rng import make_np_rng
from pylearn2.costs.cost import Cost
from theano import shared
from theano import tensor
from theano import scan
from theano import function
from theano.compat.python2x import OrderedDict
import theano
import numpy

class CRTRL(Model):
    def __init__(self, seq_len, num_neurons, rng=9001, irange=.01,
                 **kwargs):
        super(CRTRL, self).__init__(**kwargs)
        self.irange = irange
        self.num_inputs = 2*(seq_len+num_neurons+1)
        self.seq_len = seq_len
        self.num_neurons = num_neurons
        self.rng = make_np_rng(rng, which_method="randn")
        self.input_space = VectorSpace(seq_len, dtype='complex64')
        self.output_space = VectorSpace(num_neurons, dtype='complex64')
        self._initialize_weights()
        seq = tensor.ccol('seq')
        carry_state = tensor.bscalar(name='carry_state')
        self.state = None
        self._reset_state()
        self.augmented_input = shared(numpy.zeros((1,self.num_inputs), 
                                                  dtype='complex64'),
                                      name='augmented_input')
        result,updates = scan(
            self._upward_pass, 
            sequences=[dict(input=seq, taps=range(-seq_len+1, 1))],
            non_sequences=[self._params[0], self.state, carry_state])
        self.upward_pass = function([seq, carry_state], result, 
                                    updates=updates, name='upward_pass')

    def _reset_state(self):
        zero = numpy.zeros((1,self.num_neurons), dtype='complex64')
        if self.state:
            self.state.set_value(zero)
        else:
            self.state = shared(zero, 'state')
        
    def _initialize_weights(self):
        init = self.irange * (self.rng.rand(self.num_inputs, 
                                            self.num_neurons) + 
                              1j*self.rng.rand(self.num_inputs, 
                                               self.num_neurons)
                              ).astype('complex64')
        self._params = [shared(init, name="W")]

    def _upward_pass(self, *args):
        # the input seuence consisteng of seq_len consecutive samples
        augmented_input = args[:-3]
        W = args[-3]
        # the output from the last pass 
        feedback = args[-2].T
        carry_state = args[-1]
        one = tensor.constant([[1+1j]], ndim=2, dtype='complex64', name='one')
        # create the augmented input, 1 x 2*(n+p+1)
        augmented_input = tensor.concatenate([augmented_input, feedback, one], 
                                             axis=0).T
        augmented_input = (tensor.concatenate(
                [augmented_input, tensor.conj(augmented_input)], axis=1))
        updates = OrderedDict()
        augmented_input = tensor.patternbroadcast(augmented_input, 
                                                  [False,False])
        updates[self.augmented_input] = augmented_input
        # 1 x n
        result = tensor.tanh(tensor.dot(augmented_input, W))
        updates[self.state] = tensor.switch(carry_state, result, feedback.T)
        return result,updates

    def infer(self, inputs, carry_state=1):
        return self.upward_pass(inputs, carry_state)[:,0,-1]
        
class CRTRLCost(Cost):
    def __init__(self, model):
        self.m = model
        self.sensitivities = (shared(
                numpy.zeros((self.m.num_inputs,
                             self.m.num_neurons,
                             self.m.num_neurons),
                            dtype='complex64')))
        self.grad = shared(numpy.zeros(self.m.num_neurons, dtype='complex64'))
        result,updates = self._compute_sensitivities()
        self.compute_sensitivities = theano.function([], result, 
                                                     updates=updates)

    def _compute_sensitivities(self):
        n = self.m.num_neurons
        p = self.m.seq_len
        N = n+p+1
        i = range(p+1,p+1+n)
        i.extend(range(N+p+1,N+n+p+1))
        sens = tensor.tensordot(tensor.tile(self.sensitivities, [1,2,1]), 
                               self.m._params[0][i].conj(), [1,0])
        result,updates = scan(
            lambda i :
                tensor.inc_subtensor(sens[:,i,i], 
                                     self.m.augmented_input.flatten().conj()),
            sequences=[tensor.arange(n)])
        return result,updates
 
    def expr(self, model, data, **kwargs):
        y = data[-1]
        y_hat = model.infer(data[:-1])
        e_real = tensor.real(y) - tensor.real(y_hat)
        e_imag = tensor.imag(y) - tensor.imag(y_hat)
        cost = 0.5 * (e_real**2 + e_imag**2)
        return cost

    def get_gradients(self, model, data, **kwargs):
        assert len(data) == model.seq_len +1, ("Training sequence must " + 
                                               "have the correct length.")
        updates = OrderedDict()
        gradients = OrderedDict()
        y = data[-1]
        up = model.upward_pass(data[:-1], 1)
        d = (1 - up.flatten())**2
        e = shared(numpy.zeros(model.num_neurons, dtype='complex64'))
        e = tensor.set_subtensor(e[0], (y - up[0,0,-1])[0])
        sens = ((self.compute_sensitivities() * d) * 
                e.dimshuffle(('x',0,'x'))).sum(axis=1)
        gradients[self.m._params[0]] = sens
        return gradients, updates
