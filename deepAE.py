import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess 
from pylearn2.config import yaml_parse
from pylearn2.costs.autoencoder import *

import sys
import os
import cPickle
import bregman
import theano
import random
import tables

class sevdig8000(dense_design_matrix.DenseDesignMatrix):
    """
    Loads CQFTs for approximately 8000 songs and flattens it such that
    one CQFT frame is one observation. Normalizes the data first so that the
    frames of each song are within the range [0,1]
    """
    def __init__(self, which_set):
        rng = [1978, 9, 7]
        path = "${PYLEARN2_DATA_PATH}/deepAE/7dig8000.npy"
        X = serial.load(path)
        X = np.cast['float32'](X)
        b,m,n = X.shape
        assert which_set in ['train', 'test']
        if which_set is 'train':
            X = X[:2*b/3]
        else:
            X = X[2*b/3:]
        X = X.transpose([0,2,1]).reshape((-1, m*n))
        X -= X.min(axis=1, keepdims=True)
        X /= X.max(axis=1, keepdims=True)
        X = X.reshape((-1, m))
        super(sevdig8000, self).__init__(X=X, topo_view=None, y=None,
                                       view_converter=None,
                                       axes=None,
                                       rng=rng, preprocessor=None,
                                       fit_preprocessor=False)
        assert not np.any(np.isnan(self.X))

class sevdig8000FFT(dense_design_matrix.DenseDesignMatrix):
    """
    Loads FFT frames. Data was loaded from 7Dig8000_all.h5.
    """
    def __init__(self, which_set):
        assert which_set in ['tra', 'val', 'tes']
        rng = [1978, 9, 7]
        path = preprocess("${PYLEARN2_DATA_PATH}/deepAE/data/maxvec.npy")
        m = np.load(path)
        path = preprocess("${PYLEARN2_DATA_PATH}/deepAE/data/"+
                          which_set+".npy")
        X = np.load(path).item()['X'].astype('float32')
        X /= m
        m,r = X.shape
        topo_view = X.reshape(m,r,1,1)
        super(sevdig8000FFT, self).__init__(X=X, topo_view=topo_view, y=None,
                                       view_converter=None,
                                       axes=['b', 0, 1, 'c'],
                                       rng=rng, preprocessor=None,
                                       fit_preprocessor=False)
        assert not np.any(np.isnan(self.X))

class sevdig8000H5(dense_design_matrix.DenseDesignMatrixPyTables):
    """
    Some docs here
    """
    def __init__(self, fn):
        rng = [1978, 9, 7]
        path = "${PYLEARN2_DATA_PATH}/deepAE/data/"+fn
        self.h5file = tables.open_file(preprocess(path))
        data = self.h5file.getNode('/', "Data")
        super(sevdig8000H5, self).__init__(X=data.X, topo_view=None, y=None,
                                           view_converter=None, axes=None,
                                           rng=rng)

class L1(DefaultDataSpecsMixin, Cost):
    """
    Class for computing the L1 regularization penalty on the activation
    of the hidden layer. Should encourage sparse activation.
    """
    def expr(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        X = data
        cost = theano.tensor.abs_(model.encode(X)).sum(axis=1).mean()
        return cost

class CE(DefaultDataSpecsMixin, Cost):
    """
    Class for computing CE regularization penalty on the activation
    of the hidden layer. Should encourage sparse activation.
    """
    def __init__(self, rho):
        self.rho = rho

    def expr(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        X = data
        rho = self.rho
        rho_hat = model.encode(X).mean(axis=0)
        cost = (rho*(tensor.log(rho)-tensor.log(rho_hat)) + 
                (1-rho)*(tensor.log(1-rho)-tensor.log(1-rho_hat))).sum()
        return cost

def gen_tables_subset(h5_path, how_many, save_path, seed=9778):
    tables.file._open_files.close_all()                                          
    source = tables.open_file(h5_path, 'r')
    data = source.get_node('/', "Data")
    nr,nc = data.X.shape
    h5file = tables.open_file(save_path, mode="w", 
                             title="7Didgital Framed Decomposed Dataset Subset")
    gcolumns = h5file.createGroup(h5file.root, "Data", "Data")
    atom = tables.Float32Atom()
    filters = tables.Filters(complib='blosc', complevel=5)
    h5file.createCArray(gcolumns, 'X', atom=atom, shape=(how_many,nc),
                        title="Data values", filters=filters)
    np.random.seed(seed)
    indexes = np.random.permutation(xrange(nr))[:how_many]
    for i,ix in enumerate(indexes):
        print(float(i)/how_many)
        gcolumns.X[i] = data.X[ix]
        if i % 1000 == 0:
            h5file.flush()
    h5file.flush()
    # need a better solution than this, which will load the whole dataset into
    # memory
    gcolumns.X[:] -= gcolumns.X[:].min()
    gcolumns.X[:] /= np.abs(gcolumns.X[:]).max()
    h5file.flush()
    h5file.close()
    source.close()

def gen_data_from_table(h5_path, save_path, seed=9778):
    tables.file._open_files.close_all()
    source_f = tables.open_file(h5_path, 'r')
    source = source_f.get_node("/", "Data")
    nr = source.X.nrows
    _,nc = source.X.chunkshape
    np.random.seed(seed)
    n_tra = 50000
    n_val = 10000
    n_tes = 10000
    n = n_tra + n_val + n_tes
    indexes = np.random.permutation(xrange(nr))[:n]

    print("Building training set.")
    with open(save_path+'/tra.npy', 'w') as f:
        data = {'ix' : indexes[:n_tra], 'X' : np.empty((n_tra, nc), dtype='float32')}
        for i,ix in enumerate(indexes[:n_tra]):
            print("\t{0}".format(float(i)/n_tra))
            data['X'][i] = source.X[ix]
        np.save(f, data)

    print("Building validation set.")
    with open(save_path+'/val.npy', 'w') as f:
        data = {'ix' : indexes[n_tra:n_tra+n_val], 'X' : np.empty((n_val, nc), 
                                                                  dtype='float32')}
        for i,ix in enumerate(indexes[n_tra:n_tra+n_val]):
            print("\t{0}".format(float(i)/n_val))
            data['X'][i] = source.X[ix]
        np.save(f, data)

    print("Building testing set.")
    with open(save_path+'/tes.npy', 'w') as f:
        data = {'ix' : indexes[-n_val:], 'X' : np.empty((n_tes, nc), dtype='float32')}
        for i,ix in enumerate(indexes[-n_val:]):
            print("\t{0}".format(float(i)/n_tes))
            data['X'][i] = source.X[ix]
        np.save(f, data)
    source_f.close()

def gen_tables_all_stat(data_base=
                        '/global/data/casey/sarroff/projects/groove/data', 
                        save_path='/scratch/7Dig_frames_c2.h5'):
    # current num of STFT frames in dataset: 12917952
    nframes = 12917952
    allkeys = np.load(data_base+'/allkeys.npy')
    data = np.load(data_base+'/lcqft/'+allkeys[0]+'.lcqft.npz')
    ndim = data['STFT'].shape[0]
    data.close()

    tables.file._open_files.close_all()
    h5file = tables.openFile(save_path, mode="w", 
                             title="7Didgital Framed Decomposed Dataset")
    gcolumns = h5file.createGroup(h5file.root, "Data", "Data")
    atom = tables.Float32Atom()
    filters = tables.Filters(complib='blosc', complevel=5)
    h5file.createCArray(gcolumns, 'X', atom=atom, shape=(nframes,ndim),
                        title="Data values", filters=filters)
    start = 0
    for i,k in enumerate(allkeys):
        print(float(i)/len(allkeys))
        data = np.load(data_base+'/lcqft/'+allkeys[i]+'.lcqft.npz')
        nr = data['STFT'].shape[1]
        gcolumns.X[start:start+nr] = np.abs(data['STFT']).astype('float32').T
        start += nr
        data.close()
        h5file.flush()
    h5file.close()

def gen_tables_all_ext(data_base=
                       '/global/data/casey/sarroff/projects/groove/data', 
                       save_path='/scratch/7Dig_frames_c3.h5'):
    # current num of STFT frames in dataset: 12917952
    nframes = 12917952
    allkeys = np.load(data_base+'/allkeys.npy')
    data = np.load(data_base+'/lcqft/'+allkeys[0]+'.lcqft.npz')
    ndim = data['STFT'].shape[0]
    data.close()

    tables.file._open_files.close_all()
    h5file = tables.openFile(save_path, mode="w", 
                             title="7Didgital Framed Decomposed Dataset")
    gcolumns = h5file.createGroup(h5file.root, "Data", "Data")
    atom = tables.Float32Atom()
    filters = tables.Filters(complib='blosc', complevel=5)
    h5file.createEArray(gcolumns, 'X', atom=atom, shape=(0,ndim),
                        title="Data values", filters=filters,
                        expectedrows=nframes)
    for i,k in enumerate(allkeys):
        print(float(i)/len(allkeys))
        data = np.load(data_base+'/lcqft/'+allkeys[i]+'.lcqft.npz')
        gcolumns.X.append(np.abs(data['STFT']).astype('float32').T)
        data.close()
        h5file.flush()
    h5file.close()

class ReconstructFromModel(object):
    def __init__(self, model_f=None):
        self.allkeys_path = '/global/data/casey/sarroff/projects/groove/data'
        if model_f is not None:
            self.load_model(model_f)

    def run(self, model_f, wav_f):
        d = os.path.dirname(wav_f)
        wav_b = os.path.basename(wav_f).split('.')[0]
        mod_b = os.path.basename(model_f).split('.')[0]
        print("Loading model: {0}".format(mod_b))
        self.load_model(model_f)
        print("Extracting features: {0}".format(wav_b))
        self.compute_stft_features(wav_f)
        print("Reconstructing...")
        self.reconstruct()
        print("Plotting...")
        self.plot_recon(d+"/"+mod_b+"_"+wav_b)
        print("Synthesizing audio...")
        self.synth_audio(d+"/"+mod_b+"_"+wav_b)
        print("")

    def load_model(self, model_f):
        f = open(model_f, 'r')
        self.model = cPickle.load(f)
        f.close()

    def load_features(self, data_base, key=None):
        if key is None:
            k = pick_rand_key()
        self.F = np.load(k+'.cqft.npz')

    def pick_rand_key(self):
        if self.allkeys is None:
            self.allkeys = np.load(self.allkeys_path)
        return self.allkeys[np.randint(len(self.allkeys))]

    def compute_stft_features(self, wav_f):
        p = bregman.features.Features.default_feature_params()
        p['feature'] = 'stft'
        p['hi'] = 10000
        p['nfft'] = 2048
        p['nhop'] = 1024
        p['wfft'] = 2048
        self.F = bregman.features.Features(wav_f, p)
        self.F.sample_rate = self.F.x.sample_rate
        self.F.feature_params['sample_rate'] = self.F.sample_rate

    def reconstruct(self):
        assert self.F is not None
        assert self.model is not None
        ndims = self.model.input_space.dim
        assert ndims == self.F.STFT.shape[0]
        orig = self.F.X
        orig -= orig.min()
        orig /= orig.max()
        self.orig = orig
        x = theano.shared(self.orig.T.astype('float32'))
        self.recon = self.model.reconstruct(x).T.eval()
        return self.orig, self.recon

    def plot_recon(self, fn=None):
        plt.figure()
        plt.subplot(1,2,1)
        bregman.features.feature_plot(self.orig, dbscale=True, nofig=True, 
                                      title_string="orig")
        plt.subplot(1,2,2)
        bregman.features.feature_plot(self.recon, dbscale=True, nofig=True,
                                      title_string="recon")
        plt.title("recon")
        if fn is not None:
            plt.savefig(fn)
            plt.close()

    def synth_audio(self, fn_base=None, inv_orig=False):
        if inv_orig:
            self.F.inverse()
            x_hat = self.F.x_hat
            x_hat /= np.abs(x_hat).max()
            bregman.sound.wavwrite(x_hat, fn_base+"_orig.wav",
                                   self.F.feature_params['sample_rate'])
        self.F.inverse(X_hat=self.recon)
        x_hat = self.F.x_hat
        x_hat /= np.abs(x_hat).max()
        if fn_base is not None:
            bregman.sound.wavwrite(x_hat, fn_base+"_recon.wav",
                                   self.F.feature_params['sample_rate'])
        return x_hat

def test_deepAE(argv, path='/home/asarroff/projects/deepAutoController/scripts'):
    SCRIPT = sys.argv[1]
    NHID = int(sys.argv[2])

    script_path = path+'/'+SCRIPT+'.yaml'
    print("Running "+script_path)
    f = open(script_path, 'r')
    yaml = f.read()
    f.close()
    
    inputs = {'SCRIPT' : SCRIPT,
              'NHID' : NHID}
    
    yaml = yaml % (inputs)
    train = yaml_parse.load(yaml)
    train.main_loop()

if __name__ == '__main__':
    test_deepAE(sys.argv)
