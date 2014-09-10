import glob
import os
import numpy as np
import h5py
import sys
import cPickle

def create_hdf5(base, feat):
    fnames = glob.glob(base+'/items/*.npy')
    with open(base+'/params.pkl', 'r') as f:
        p = cPickle.load(f)
    NSECS = 20
    FPS = p['sample_rate'] // p['nhop']
    nc = NSECS * FPS
    nr = np.load(fnames[0]).shape[0]
    h5file = h5py.File(base+'/'+feat+'_all.h5', 'w')
    datlen = len(fnames)
    strlen = len(fnames[0])
    h5file.create_dataset(name='data', shape=(datlen, nr, nc), dtype='float32')
    h5file.create_dataset(name='key', shape=(datlen, 1), dtype='S'+str(strlen))
    for i,fname in enumerate(fnames):
        key = os.path.basename(fname).split('.')[0]
        data = np.load(fname)
        h5file['data'][i] = data[:,5*FPS:(NSECS+5)*FPS].astype('float32')
        h5file['key'][i] = fname
        if i%100 == 0:
            h5file.flush()
            print(float(i)/datlen)
    h5file.flush()
    h5file.close()

def partition_hdf5(base, feat):
    h5file = h5py.File(base+'/'+feat+'_all.h5', 'r')
    datlen,nfeat,nframes = h5file['data'].shape
    ix = np.arange(datlen)
    np.random.seed(9778)
    np.random.shuffle(ix)

    train_ix = list(ix[:datlen/3])
    #train_ix = list(ix[:100])
    train_ix.sort()
    test_ix = list(ix[datlen/3:2*datlen/3])
    #test_ix = list(ix[100:200])
    test_ix.sort()
    val_ix = list(ix[2*datlen/3:])
    #val_ix = list(ix[200:300])
    val_ix.sort()

    print("Creating train dataset")
    train = h5py.File(base+'/'+feat+'_train.h5', 'w')
    data = h5file['data'][train_ix].transpose((0,2,1)).reshape((-1,nfeat))
    np.random.shuffle(data)
    label = np.zeros(data.shape[0]).astype('uint8')
    key = np.tile(h5file['key'][train_ix], nframes).flatten()
    train.create_dataset('data', data=data)
    train.create_dataset('label', data=label)
    train.create_dataset('key', data=key)
    train.flush()
    train.close()

    print("Creating test dataset")
    test = h5py.File(base+'/'+feat+'_test.h5', 'w')
    data = h5file['data'][test_ix].transpose((0,2,1)).reshape((-1,nfeat))
    label = np.zeros(data.shape[0]).astype('uint8')
    key = np.tile(h5file['key'][test_ix], nframes).flatten()
    test.create_dataset('data', data=data)
    test.create_dataset('label', data=label)
    test.create_dataset('key', data=key)
    test.flush()
    test.close()

    print("Creating validation dataset")
    val = h5py.File(base+'/'+feat+'_val.h5', 'w')
    data = h5file['data'][val_ix].transpose((0,2,1)).reshape((-1,nfeat))
    label = np.zeros(data.shape[0]).astype('uint8')
    key = np.tile(h5file['key'][val_ix], nframes).flatten()
    val.create_dataset('data', data=data)
    val.create_dataset('label', data=label)
    val.create_dataset('key', data=key)
    val.flush()
    val.close()

    h5file.close()

if __name__ == "__main__":
    base = sys.argv[1]
    feat = sys.argv[2]
    create_hdf5(base, feat)
    partition_hdf5(base, feat)
