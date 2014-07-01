from bregman import features
import glob
import cPickle
import numpy as np
import os
from multiprocessing import Pool
import itertools
import sys
import h5py

def extract(arg):
    fname,p,base = arg
    print(fname)
    F = features.Features(fname, p)
    key = os.path.basename(fname).split('.')[0]
    np.save(base+'/'+key+'.npy', F.X)
        
if __name__ == "__main__":
    which = sys.argv[1]
    p = features.default_feature_params()
    if which == 'cqft_3bpo':
        p['feature'] = 'cqft'
        p['nbpo'] = 3
        p['lo'] = 20
        p['hi'] = 11025
        p['sample_rate'] = 22050
        p['nfft'] = 16384
        p['wfft'] = 8192
        p['nhop'] = 2205
    elif which == 'cqft_12bpo':
        p['feature'] = 'cqft'
        p['nbpo'] = 12
        p['lo'] = 20
        p['hi'] = 11025
        p['sample_rate'] = 22050
        p['nfft'] = 16384
        p['wfft'] = 8192
        p['nhop'] = 2205
    elif which == 'stft':
        p['feature'] = 'stft'
        p['nbpo'] = None
        p['lo'] = None
        p['hi'] = None
        p['sample_rate'] = 22050
        p['nfft'] = 2048
        p['wfft'] = 2048
        p['nhop'] = 1024
    else:
        print("Unrecognized feature configuration")
        exit(1)
    base = '/scratch/sarroff/feat/'+which
    with open(base+'/'+which+'_params.pkl', 'w') as fid:
        cPickle.dump(p, fid, -1)
    base = '/scratch/sarroff/feat/'+which+'/items'
    fnames = glob.glob('/scratch/sarroff/wav/*.wav')
    nproc = int(sys.argv[2])
    pool = Pool(processes=nproc)
    pool.map(extract, zip(fnames, itertools.cycle([p]), 
                          itertools.cycle([base])))
    exit(0)
    
