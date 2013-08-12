import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial

class sevdig8000(dense_design_matrix.DenseDesignMatrix):
    def __init__(self):
        rng = 9778
        path = "${PYLEARN2_DATA_PATH}/7dig8000.npy"
        X = serial.load(path)
        X = np.cast['float32'](X)
        b,m,n = X.shape
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
