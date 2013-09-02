import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys
import os

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

def test_deepAE(name,
                path='/home/asarroff/projects/deepAutoController/scripts'):
    script_path = path+'/'+name+'.yaml'
    f = open(script_path, 'r')
    yaml = f.read()
    f.close()
    yaml = yaml % ({'script': name})
    train = yaml_parse.load(yaml)
    train.main_loop()

if __name__ == '__main__':
    script = sys.argv[1]
    test_deepAE(script)
