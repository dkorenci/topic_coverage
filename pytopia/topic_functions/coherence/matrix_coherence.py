from pytopia.tools.IdComposer import IdComposer

import numpy as np

class MatrixCoherence(IdComposer):
    '''
    Calculates coherence of a matrix using SVD-based measures described
    in the paper: "Can matrix coherence be efficiently and accurately estimated"
    '''

    def __init__(self, method):
        '''
        :param method: 'mu', 'mu0' or 'mu1'
        '''
        self.method = method
        IdComposer.__init__(self)

    def __call__(self, m):
        if isinstance(m, np.ndarray): return self.__ndCoherence(m)
        else: raise Exception('unsuported matrix type')

    def __ndCoherence(self, m):
        from numpy.linalg import svd, norm
        from numpy import dot, transpose as T
        d = svd(m, full_matrices=False)
        u, sigma, v = d[0], d[1], d[2]
        n, m_ = float(m.shape[0]), float(m.shape[1])
        r = min(n, m_)
        if self.method == 'mu':
            res = - np.sqrt(n)*np.max(np.abs(u))
        elif self.method == 'mu0':
            res = - n/r * np.max(np.sum(u*u, axis=1))
        elif self.method == 'mu1':
            res = - np.sqrt(n*m_/r) * np.max(np.abs(dot(T(u), v)))
        return res

