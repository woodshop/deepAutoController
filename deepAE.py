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
from dA import dA
from scipy.misc import comb
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from matplotlib import pyplot

SEED = 9778

class deepAE(object):
    def __init__(self, n_ins=1280, numpy_rng=None, theano_rng=None,
                 hidden_layers_sizes=[1024, 32],
                 corruption_levels=None, **kwargs):

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(SEED)

        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        assert self.n_layers > 0

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        self.hidden_layers_sizes = hidden_layers_sizes
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.n_ins = n_ins
        self.unrolled = False
        if corruption_levels is None:
            corruption_levels = [0] * len(hidden_layers_sizes)
        self.corruption_levels = corruption_levels

        self.x = T.matrix('x')

    def init_layers(self):
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = self.n_ins
            else:
                input_size = self.hidden_layers_sizes[i - 1]
            if i == 0:
                layer_input = self.x
            elif i == self.n_layers - 1:
                layer_input = self.dA_layers[-1].get_hidden_values(
                    self.dA_layers[-1].x)
            else:
                layer_input = self.dA_layers[-1].get_hidden_values(
                    self.dA_layers[-1].x)
            if i == self.n_layers - 1:
                dA_layer = dA(numpy_rng=self.numpy_rng,
                              theano_rng=self.theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=self.hidden_layers_sizes[i],
                              nl=True)
            else:
                dA_layer = dA(numpy_rng=self.numpy_rng,
                              theano_rng=self.theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=self.hidden_layers_sizes[i],
                              nl=True)
            self.dA_layers.append(dA_layer)
            self.params.extend([dA_layer.W, dA_layer.b])

    def rebuild_layers(self, model_weights, mult=None):
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = self.n_ins
                layer_input = self.x
            else:
                input_size = self.hidden_layers_sizes[i - 1]
                layer_input = self.dA_layers[-1].get_hidden_values(
                    self.dA_layers[-1].x)
            n_hidden=self.hidden_layers_sizes[i]
            if i == self.n_layers - 1:
                dA_layer = dA(numpy_rng=self.numpy_rng,
                              theano_rng=self.theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=n_hidden,
                              W=model_weights[i*2],
                              bhid=model_weights[i*2+1], mult=mult)
            else:
                dA_layer = dA(numpy_rng=self.numpy_rng,
                              theano_rng=self.theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=n_hidden,
                              W=model_weights[i*2],
                              bhid=model_weights[i*2+1])
            self.dA_layers.append(dA_layer)
            self.params.extend([self.dA_layers[-1].W,
                                self.dA_layers[-1].b])

        for i in xrange(self.n_layers):
            if i == self.n_layers - 1:
                n_hidden = self.n_ins
            else:
                n_hidden = self.hidden_layers_sizes[self.n_layers-i-2]
            input_size = self.hidden_layers_sizes[self.n_layers-i-1]
            layer_input = self.dA_layers[-1].get_hidden_values(
                self.dA_layers[-1].x)
            dA_layer = dA(numpy_rng=self.numpy_rng,
                          theano_rng=self.theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=n_hidden,
                          W=model_weights[2*self.n_layers+i*2],
                          bhid=model_weights[2*self.n_layers+i*2+1])
            self.dA_layers.append(dA_layer)
            self.params.extend([self.dA_layers[-1].W,
                                self.dA_layers[-1].b])
    def unroll(self):
        # Unrolls the upper half of the deep autoencoder. If there are n
        # hidden layers during pretraining, there are 2n-1 hidden layers after
        # unrolling. The weights of the symmetrical upper half initially match
        # those of the lower half but are left untied.
        for i in range(self.n_layers)[::-1]:
            W = theano.shared(self.dA_layers[i].W.get_value().T,
                              name='W')
            b = theano.shared(self.dA_layers[i].b_prime.get_value(),
                              name='b')
            self.params.extend([W, b])
            if i == self.n_layers - 1:
                self.dA_layers[-1].W_prime = W
                self.dA_layers[-1].b_prime = b
                continue

            if i == self.n_layers - 2:
                layer_input = self.dA_layers[-1].get_reconstructed_input(
                    self.dA_layers[-1].get_hidden_values(self.dA_layers[-1].x))
            else:
                layer_input = self.dA_layers[-1].get_hidden_values(
                    self.dA_layers[-1].x)
            if i == 0:
                n_hidden = self.n_ins
            else:
                n_hidden=self.hidden_layers_sizes[i-1]
            input_size = self.hidden_layers_sizes[i]
            dA_layer = dA(numpy_rng=self.numpy_rng,
                          theano_rng=self.theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=n_hidden, W=W, bhid=b)
            self.dA_layers.append(dA_layer)
        self.unrolled = True

    def cross_entropy(self):
        z = self.dA_layers[-1].get_hidden_values(self.dA_layers[-1].x)
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        return T.mean(L)

    def mean_sq_diff(self):
        z = self.dA_layers[-1].get_hidden_values(self.dA_layers[-1].x)
        L = T.sum((self.x-z)**2, axis=1)
        return T.mean(L)

    def pretraining_functions(self):
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # number of batches
        n_batches = self.train_set.get_value(borrow=True).shape[0] / self.pt_bs
        # begining of a batch, given `index`
        batch_begin = index * self.pt_bs
        # ending of a batch given `index`
        batch_end = batch_begin + self.pt_bs

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(corruption_level, default=0.0),
                              theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x: self.train_set[batch_begin:
                                                             batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)
        return pretrain_fns

    def build_finetune_functions(self):
        # compute number of minibatches for training, validation and testing
        n_valid_batches = self.valid_set.get_value(borrow=True).shape[0]
        n_valid_batches /= self.ft_bs
        n_test_batches = self.test_set.get_value(borrow=True).shape[0]
        n_test_batches /= self.ft_bs

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        c_func = self.mean_sq_diff()
        #c_func = self.cross_entropy()
        gparams = T.grad(c_func, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * self.ft_lr))


        train_fn = theano.function(inputs=[index],
              outputs=c_func,
              updates=updates,
              givens={
                  self.x: self.train_set[index * self.ft_bs:
                                      (index + 1) * self.ft_bs]})

        valid_score_i = theano.function(inputs=[index],
                outputs=c_func,
                givens={
                    self.x: self.valid_set[index * self.ft_bs:
                                        (index + 1) * self.ft_bs]})

        test_score_i = theano.function(inputs=[index],
                outputs=c_func,
                givens={
                    self.x: self.test_set[index * self.ft_bs:
                                       (index + 1) * self.ft_bs]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
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
        pretraining_fns = self.pretraining_functions()

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
                        corruption=self.corruption_levels[i],
                        lr=self.pt_lr))
                c = numpy.mean(c) / self.dA_layers[i].n_visible
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
    model.init_layers()
    model.set_train_vars(datasets=datasets, ft_epochs=100)
    model.pretrain()
    model.unroll()
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
    p['corruption_levels'] = model.corruption_levels
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
    pyplot.imshow(orig.reshape(87,-1)[2:], origin='lower', aspect='auto')
    pyplot.title(t)
    pyplot.ylabel('Freq Band')
    pyplot.subplot(2,1,2)
    pyplot.imshow(recon.reshape(87,-1)[2:], origin='lower', aspect='auto')
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
