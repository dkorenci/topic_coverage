import numpy as np

def apply2RowPairs(m1, m2, f):
    '''
    Apply function that operates on numpy vectors to all pairs
    of rows of two matrices (with the same number of columns).
    :return: matrix of function values, with dim. rows(m1) x rows(m2)
    '''
    assert m1.shape[1] == m2.shape[1]
    R1, R2 = m1.shape[0], m2.shape[0]
    r = np.empty((R1, R2))
    for i in range(R1):
        for j in range(R2):
            r[i,j]=f(m1[i,:],m2[j,:])
    return r