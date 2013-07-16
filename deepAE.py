"""
 Implementation of a deep autoencoder with some helper functions for 
 reconstruction of the code layer and output.

 This code is based on the Theano stacked denoising autoencoder tutorial. The 
 main differences between this code and the tutorial are:
   * This is a deep autoencoder, the tutorial shows a stacked autoencoder
   * In order to implement a deepautoencoder, the model is unfolded into
       its symmetrical upper half.
   * A couple of helper functions are provided to rebuild a saved model and 
       plot results
   * The learned model parameters are saved, along with input variables cost 
       curves, and an arbitrary example of an original input and its 
       reconstruction.
"""
import sys
import time
import numpy
import argparse
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
#from dA import dA
from scipy.misc import comb
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from matplotlib import pyplot

SEED = 9778

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

class cA(object):
    """
    Contractive Auto-Encoder class (cA)

    The contractive autoencoder tries to reconstruct the input with an
    additional constraint on the latent space. With the objective of
    obtaining a robust representation of the input space, we
    regularize the L2 norm(Froebenius) of the jacobian of the hidden
    representation with respect to the input. Please refer to Rifai et
    al., 2011 for more details.

    If x is the input then equation (1) computes the projection of the
    input into the latent space h. Equation (2) computes the jacobian
    of h with respect to x.  Equation (3) computes the reconstruction
    of the input, while equation (4) computes the reconstruction
    error and the added regularization term from Eq.(2).

    .. math::

        h_i = s(W_i x + b_i)                                             (1)

        J_i = h_i (1 - h_i) * W_i                                        (2)

        x' = s(W' h  + b')                                               (3)

        L = -sum_{k=1}^d [x_k \log x'_k + (1-x_k) \log( 1-x'_k)]
             + lambda * sum_{i=1}^d sum_{j=1}^n J_{ij}^2                 (4)

    """

    def __init__(self, numpy_rng, input=None, n_visible=784, n_hidden=100,
                 n_batchsize=1, W=None, bhid=None, bvis=None):
        """
        Initialize the cA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the contraction level. The
        constructor also receives symbolic variables for the input, weights and
        bias.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given
                     one is generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone cA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type n_batchsize int
        :param n_batchsize: number of examples per batch

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_batchsize = n_batchsize
        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)),
                                      dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                                   dtype=theano.config.floatX),
                                 borrow=True)

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden,
                                                   dtype=theano.config.floatX),
                                 name='b',
                                 borrow=True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T

        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        """
        Computes the values of the hidden layer
        """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_jacobian(self, hidden, W):
        """
        Computes the jacobian of the hidden layer with respect to
        the input, reshapes are necessary for broadcasting the
        element-wise product on the right axis
        """
        return T.reshape(hidden * (1 - hidden),
                         (self.n_batchsize, 1, self.n_hidden)) * T.reshape(
                             W, (1, self.n_visible, self.n_hidden))

    def get_reconstructed_input(self, hidden):
        """
        Computes the reconstructed input given the values of the
        hidden layer
        """
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, contraction_level, learning_rate):
        """
        This function computes the cost and the updates for one trainng
        step of the cA
        """
        self.n_batchsize = 20
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        J = self.get_jacobian(y, self.W)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        self.L_rec = - T.sum(self.x * T.log(z) +
                             (1 - self.x) * T.log(1 - z),
                             axis=1)

        # Compute the jacobian and average over the number of samples/minibatch
        self.L_jacob = T.sum(J ** 2) / self.n_batchsize

        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(self.L_rec) + contraction_level * T.mean(self.L_jacob)

        # compute the gradients of the cost of the `cA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)

class deepAE(object):
    def __init__(self, n_ins=1280, numpy_rng=None,
                 hidden_layers_sizes=[1024, 32],
                 contraction_levels=None, pt_bs=20, **kwargs):
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(SEED)
        self.numpy_rng = numpy_rng
        self.cA_layers = []
        self.sigmoid_layers = []
        self.n_layers = len(hidden_layers_sizes)
        self.params = [None]*self.n_layers*4
        assert self.n_layers > 0
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_ins = n_ins
        if contraction_levels is None:
            contraction_levels = [0.1] * len(hidden_layers_sizes)
        self.contraction_levels = contraction_levels
        self.pt_bs = pt_bs
        self.x = T.matrix('x')
        self.x_prime = T.matrix('x')
        self.finetune_cost = self.mean_sq_diff(self.x, self.x_prime)

    def init_cA_layers(self):
        for i in xrange(self.n_layers):
            if i == 0:
                input = self.x
                n_visible = self.n_ins
            else:
                input = self.cA_layers[-1].get_hidden_values(
                    self.cA_layers[-1].x)
                n_visible = self.hidden_layers_sizes[i - 1]

            cA_layer = cA(numpy_rng=self.numpy_rng,
                          input=input,
                          n_visible=n_visible,
                          n_hidden=self.hidden_layers_sizes[i],
                          n_batchsize=self.pt_bs)
            self.cA_layers.append(cA_layer)

    def init_params(self):
        assert self.cA_layers
        for i in xrange(self.n_layers):
            self.params[i*2] = theano.shared(
                self.cA_layers[i].W.eval(), name='W', borrow=False)
            self.params[i*2 + 1] = theano.shared(
                self.cA_layers[i].b.evsal(), name='b', borrow=False)
            self.params[2*self.n_layers - 2 - 2*i] = theano.shared(
                self.cA_layers[self.n_layers - 1 - i
                               ].W_prime.eval(), 
                name='W', borrow=False)
            self.params[2*self.n_layers - 1 - 2*i] = theano.shared(
                self.cA_layers[self.n_layers - 1 - i
                               ].b_prime.eval(), 
                name='b', borrow=False)
            
    def init_sigmoid_layers(self):
        for i in xrange(self.n_layers):
            if i == 0:
                input = self.x
                n_in = self.n_ins
            else:
                input = self.sigmid_layers[-1].output
                n_in = self.hidden_layers_sizes[i - 1]
            n_out = self.hidden_layers_sizes[i]
            W=self.params[i*2]
            b=self.params[i*2 + 1]
            sigmoid_layer = HiddenLayer(rng=self.numpy_rng,
                                        input=input,
                                        n_in=n_in,
                                        n_out=n_out,
                                        W=W,
                                        b=b,
                                        activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params[i*2:i*2+1] = sigmoid_layer.params

        for i in xrange(self.n_layers):
            n = self.n_layers*2
            input=self.sigmoid_layers[-1].output
            n_in=(self.hidden_layers_sizes[self.n_layers - 1 - i])
            if i == self.n_layers - 1:
                n_out = self.n_ins
            else:
                n_out = self.hidden_layers_sizes[self.n_layers - 2 - i]
            W=self.params[n + i*2]
            b=self.params[n + i*2 + 1]
            sigmoid_layer = HiddenLayer(rng=self.numpy_rng,
                                        input=input,
                                        n_in=n_in,
                                        n_out=n_out,
                                        W=W,
                                        b=b,
                                        activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params[n+i*2:n+i*2+2] = sigmoid_layer.params

    @staticmethod
    def mean_sq_diff(x, x_prime):
        L = T.sum((x-x_prime)**2, axis=1)
        return T.mean(L)

    def build_pretraining_functions(self):
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        contraction_level = T.scalar('contraction')
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * self.pt_bs
        # ending of a batch given `index`
        batch_end = batch_begin + self.pt_bs

        pretrain_fns = []
        for cA in self.cA_layers:
            # get the cost and the updates list
            cost, updates = cA.get_cost_updates(contraction_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(inputs=[index,
                                         theano.Param(contraction_level, 
                                                      default=0.1),
                                         theano.Param(learning_rate, 
                                                      default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x: self.train_set[batch_begin:
                                                             batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)
        return pretrain_fns

    def build_finetune_functions(self):
        n_valid_batches = self.valid_set.get_value(borrow=True).shape[0]
        n_valid_batches /= self.ft_bs
        n_test_batches = self.test_set.get_value(borrow=True).shape[0]
        n_test_batches /= self.ft_bs
        index = T.lscalar('index')
        gparams = T.grad(self.finetune_cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * self.ft_lr))

        train_fn = theano.function(inputs=[index],
              outputs=self.fine_tune.cost,
              updates=updates,
              givens={
                  self.x: self.train_set[index * self.ft_bs:
                                      (index + 1) * self.ft_bs],
                  self.x_prime: self.sigmoid_layers[-1].output})

        valid_score_i = theano.function(inputs=[index],
              outputs=self.fine_tune.cost,
              givens={
                  self.x: self.valid_set[index * self.ft_bs:
                                      (index + 1) * self.ft_bs],
                  self.x_prime: self.sigmoid_layers[-1].output})

        test_score_i = theano.function(inputs=[index],
              outputs=self.fine_tune.cost,
              givens={
                  self.x: self.test_set[index * self.ft_bs:
                                      (index + 1) * self.ft_bs],
                  self.x_prime: self.sigmoid_layers[-1].output})

        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score

    def set_train_vars(self, pt_lr=0.001, pt_epochs=15, pt_bs=20,
                       ft_lr=0.1, ft_epochs=1000, ft_bs=20, ft_thr=0.995,
                       datasets=None):
        self.pt_lr = pt_lr
        self.pt_epochs = pt_epochs=15
        self.pt_bs = pt_bs
        self.ft_lr = ft_lr
        self.ft_epochs = ft_epochs
        self.ft_bs = ft_bs
        self.ft_thr=ft_thr
        if datasets is None:
            path = '/home/asarroff/global_groove/data/allW.v9.smoothed.npy'
            datasets = numpy.load(path, mmap_mode=None)
        nobs,nvars = datasets.shape
        rng = numpy.random.RandomState(SEED)
        perm = rng.permutation(nobs)
        datasets = datasets[perm]
        self.perm = perm

        self.pt_nbatches = nobs / 3 / self.pt_bs
        self.ft_nbatches = nobs / 3 / self.ft_bs
        if nvars != self.n_ins:
            print('WARNING: dataset variables truncated to fit model.')
        self.train_set = theano.shared(numpy.asarray
                                       (datasets[:nobs*1/3,:self.n_ins],
                                        dtype=theano.config.floatX),
                                        borrow=True)
        self.valid_set = theano.shared(numpy.asarray
                                       (datasets
                                        [nobs*1/3:nobs*2/3,:self.n_ins],
                                        dtype=theano.config.floatX),
                                        borrow=True)
        self.test_set = theano.shared(numpy.asarray
                                      (datasets[nobs*2/3:,:self.n_ins],
                                       dtype=theano.config.floatX),
                                       borrow=True)

    def pretrain(self):
        print ('... getting the pretraining functions')
        pretraining_fns = self.build_pretraining_functions()

        print ('... pre-training the model')
        start_time = time.clock()
        ## Pre-train layer-wise
        pretraining_costs = []
        for i in xrange(self.n_layers):
            # go through pretraining epochs
            for epoch in xrange(self.pt_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(self.pt_nbatches):
                    c.append(pretraining_fns[i](
                        index=batch_index,
                        contraction=self.contraction_levels[i],
                        lr=self.pt_lr))
                c = numpy.mean(c) / self.cA_layers[i].n_visible
                print ('Pre-training layer %i, epoch %i, ' % (i, epoch) +
                       'cost per vis unit: %f' % (c))
                pretraining_costs.append([i, epoch, c])
        end_time = time.clock()
        print >> sys.stderr, ('Pretraining ran for %.2fm' %
                              ((end_time - start_time) / 60.))
        self.pretraining_costs = pretraining_costs

    def finetune(self):
        if not self.unrolled:
            self.unroll()

        print '... getting the finetuning functions'
        train_fn, validate_model, test_model = self.build_finetune_functions()

        # look as this many examples regardless
        patience = 10 * self.ft_nbatches
        # wait this much longer when a new best is found
        patience_increase = 2.
        # a relative improvement of this much is considered significant
        improvement_threshold = self.ft_thr
        # go through this many minibatches before checking the network
        # on the validation set; in this case we check every epoch
        validation_frequency = min(self.ft_nbatches, patience / 2)

        best_params = None
        best_validation_loss = numpy.inf
        test_score = 0.
        start_time = time.clock()
        done_looping = False
        epoch = 0
        tr_losses = [[0, None]]
        v_losses = []
        te_losses = []

        print '... finetuning the model'
        v_losses.append([0, numpy.mean(validate_model())])
        print('Validation error before fine tuning: %f' %
              (v_losses[0][1]/self.n_ins))
        te_losses.append([0, numpy.mean(test_model())])
        print('Test error before fine tuning: %f' %
              (te_losses[0][1]/self.n_ins))

        while (epoch < self.ft_epochs) and (not done_looping):
            epoch = epoch + 1
            tr_c = []
            for minibatch_index in xrange(self.ft_nbatches):
                tr_c.append(train_fn(minibatch_index))
                iter = (epoch - 1) * self.ft_nbatches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    this_validation_loss = numpy.mean(validate_model())
                    tr_losses.append([iter, numpy.mean(tr_c)/self.n_ins])
                    v_losses.append([iter, this_validation_loss/self.n_ins])
                    te_losses.append([iter,
                                      numpy.mean(test_model())/self.n_ins])
                    print('epoch %i, minibatch %i/%i, validation error %f' %
                    (epoch, minibatch_index + 1, self.ft_nbatches,
                     (this_validation_loss/self.n_ins)))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if (this_validation_loss < best_validation_loss *
                            improvement_threshold):
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        print(('     epoch %i, minibatch %i/%i, test error of '
                              'best model %f') %
                        (epoch, minibatch_index + 1, self.ft_nbatches,
                         te_losses[-1][1]))

                if patience <= iter:
                    done_looping = True
                    break

        finetuning_costs = [tr_losses, v_losses, te_losses]
        end_time = time.clock()
        print(('Optimization complete with best validation score of %f '
               '\n\twith test performance %f') %
               (best_validation_loss/self.n_ins, te_losses[-1][1]))
        print >> sys.stderr, ('Finetuning ran for %.2fm' %
                              ((end_time - start_time) / 60.))
        self.finetuning_costs = finetuning_costs

    def code_out(self, ix=None):
        if ix is None:
            nobs = self.test_set.shape[0].eval()
            ix = range(nobs)
        y = []
        for i in xrange(self.n_layers):
            if i == 0:
                y.append(self.dA_layers[i].get_hidden_values(
                    self.test_set[ix]))
            else:
                y.append(self.dA_layers[i].get_hidden_values(y[-1]))
        return y[-1]

    def reconstruct_input(self, ix):
    	"""
    	Use this function to reconstruct the input from an example in the test 
        set before the model has been saved. If using a reconstructed model and
        arbitrary data, use reconstruct_input_ext
    	"""
        y = [self.code_out(ix)]
        for i in xrange(self.n_layers):
            if i == 0:
                y.append(self.dA_layers[
                    self.n_layers-1].get_reconstructed_input(y[-1]))
            else:
                y.append(self.dA_layers[
                    i+self.n_layers-1].get_hidden_values(y[-1]))
        return y[-1]

    def reconstruct_input_ext(self, model_in):
    	"""
    	Perform an autoencoding reconstruction given an arbitrary input. No 
        error checking is performed. The variable model_in must be a theano 
        shared tensor having the same number of variables ast the data that 
        was used to train the model. Multiple observations may be given. The 
        data should have observations along the rows and variables along the 
        columns.
    	"""
        y = []
        for i in xrange(self.n_layers*2):
            if i == 0:
                y.append(self.dA_layers[i].get_hidden_values(model_in))
            else:
                y.append(self.dA_layers[i].get_hidden_values(y[-1]))
        return y

def test_model(debug=False, path=None, **kwargs):
    rng = numpy.random.RandomState(SEED)
    if path is None:
        path = '../data/allcqft.npy'
    datasets = numpy.load(path, mmap_mode=None)
    if debug:
        print('DEBUGGING ON')
        datasets = datasets[:,-30*128:]
    nobs,nvars = datasets.shape
    model = deepAE(n_ins=nvars, **kwargs)
    model.init_cA_layers()
    model.set_train_vars(datasets=datasets, ft_epochs=100, pt_bs=model.pt_bs)
    model.pretrain()
    model.init_params()
    model.intit_sigmoid_layers()
    model.finetune()
    p = wrap_model_params(model)
    i = 100
    test = {}
    test['orig'] = model.test_set[i].eval()
    test['recon'] = model.reconstruct_input(i).eval()
    return p,model,test

def wrap_model_params(model):
    p = {}
    p['perm'] = model.perm
    p['n_ins'] = model.n_ins
    p['hidden_layers_sizes'] = model.hidden_layers_sizes
    p['contraction_levels'] = model.contraction_levels
    p['pt_lr'] = model.pt_lr
    p['pt_epochs'] = model.pt_epochs
    p['pt_bs'] = model.pt_bs
    p['ft_lr'] = model.ft_lr
    p['ft_epochs'] = model.ft_epochs
    p['ft_bs'] = model.ft_bs=20
    p['ft_thr'] = model.ft_thr
    return p

def rebuild(data, mult=None):
    """
    Rebuild a model using the saved l1_l2_l3.model.npy output file.
    """
    if data.__class__ is str:
        data = numpy.load(data).item()
    p = data['params']
    model = deepAE(n_ins=p['n_ins'], numpy_rng=None, theano_rng=None,
                hidden_layers_sizes=p['hidden_layers_sizes'],
                corruption_levels=p['corruption_levels'])
    model_weights = []
    for i in xrange(len(data['model_weights'])):
        model_weights.append(theano.shared(value=data['model_weights'][i][1],
                                           name=data['model_weights'][i][0],
                                           borrow=True))
    model.rebuild_layers(model_weights, mult=mult)
    return model

def plot_orig_recon(model, orig, recon):
    """
    Plot the saved original and reconstructed data saved during model 
    training. WARNING: The number of variables has been hard-coded. Edit this 
    code as necessary.
    """
    pyplot.clf()
    t = '-'.join(str(i) for i in model.hidden_layers_sizes)
    pyplot.subplot(2,1,1)
    # HARD CODED!
    pyplot.imshow(recon.reshape(87,-1), origin='lower', aspect='auto')
    pyplot.title(t)
    pyplot.ylabel('Freq Band')
    pyplot.subplot(2,1,2)
    pyplot.imshow(orig.reshape(87,-1), origin='lower', aspect='auto')
    pyplot.xlabel('Time')
    pyplot.ylabel('Freq Band')

def plot_training(model, pt_costs, ft_costs):
    pyplot.clf()
    t = '-'.join(str(i) for i in model.hidden_layers_sizes)
    pyplot.subplot(2,1,1)
    layers = model.hidden_layers_sizes
    for i in xrange(len(layers)):
        pyplot.plot([d[2] for d in pt_costs[i*15:i*15+15]], linewidth=2)
    pyplot.grid(True)
    pyplot.ylabel('Cross Entropy')
    pyplot.title('Pretraining (Top) & Finetuning (Bottom) Error')
    pyplot.legend(['Layer {0}'.format(i) for i in xrange(len(layers))])
    pyplot.subplot(2,1,2)
    pyplot.plot([d[1] for d in ft_costs[0][1:]], linewidth=2)
    pyplot.plot([d[1] for d in ft_costs[1][1:]], linewidth=2)
    pyplot.plot([d[1] for d in ft_costs[2][1:]], linewidth=2)
    pyplot.grid(True)
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Cross Entropy')
    pyplot.legend(['Training', 'Validation', 'Test'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build and test deep autoencoder for semantic hashing.')
    parser.add_argument('hidden_layers_sizes', nargs='*', type=int, 
                        help='a list  of the hidden layer sizes separates '+\
                            'by spaces. Ex: 2048 512 512 8')
    parser.add_argument('-o', dest='outpath', 
                        help='the directory where the model should be '+\
                            'stored. The name of the file will be '+\
                            'outpath/l1_l2_l3.model.npy where l1 l2 l3 are '+\
                            'the sizes of the hidden layers.')
    parser.add_argument('-d', dest='datapath',
                        help='the location of the training data. The '+\
                            'application expects a npy array where each '+\
                            'row is an observation and each column is a '+\
                            'variable. The data is randomly permuted and '+\
                            'split equally into training, vakidation, and '+\
                            'test data sets.')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    res = test_model(hidden_layers_sizes=args.hidden_layers_sizes,
                     debug=args.debug, path=args.datapath)
    fn_base = args.outpath+'/'+'_'.join(str(i) for i
                                        in args.hidden_layers_sizes)
    model_weights = []
    for i in xrange(len(res[1].params)):
        model_weights.append([res[1].params[i].name,
                              numpy.array(res[1].params[i].eval())])
    numpy.save(fn_base+'.model.npy', {'params':res[0],
                                      'model_weights':model_weights,
                                      'test':res[2],
                                      'pt_costs':res[1].pretraining_costs,
                                      'ft_costs':res[1].finetuning_costs})
