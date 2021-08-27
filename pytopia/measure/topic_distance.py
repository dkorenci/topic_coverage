'''
Distance metrics between two topics.
Each metric is a callable implementing a function
f(m1, m2) -> double, where m1 and m2 are either arrays of the same length,
or matrices with same number of columns, in which case batch computation
will be performed if possible and a row x row distance matrix returned.
'''

from pytopia.measure.utils import apply2RowPairs

from scipy.spatial.distance import cosine as cosineSp, minkowski, cdist, \
    euclidean, sqeuclidean, chebyshev, canberra as canberraSp
from scipy.stats import entropy, spearmanr, pearsonr, kendalltau
from scipy.spatial.distance import jaccard, dice
import numpy as np

class normalize:
    '''
    Adapts a function that accepts ndarray arguments by normalizing
    these vectors (or matrix rows) to distributions.
    '''
    def __init__(self, func):
        self.func = func
        self.id = func.__name__ + 'norm'

    def __call__(self, *args):
        nargs = [norm2dist(a) for a in args]
        return self.func(*nargs)

def norm2dist(a):
    ''' Normalize nonzero ndarray or matrix to probability distribution. '''
    if len(a.shape) == 1:
        s = a.sum()
        return a/s if s != 0 else a
    else:
        sums = a.sum(axis=1)
        return a / sums[:, np.newaxis]

def klDivSymm(v1, v2):
    return 0.5 * (klDivZero(v1, v2) + klDivZero(v2, v1))

def klDivZero(v1, v2): return kullbackLeibler(v1, v2, 1e-9)

def kullbackLeibler(v1, v2, zero=None):
    '''
    :param t1, v2: ndarray representing probability distribution
    :param zero: if not None, perform check and replace all zero values with (small constant) zero
    :return:
    '''
    if zero is not None:
        v1, v2 = np.copy(v1), np.copy(v2)
        v1[v1 == 0] = zero
        v2[v2 == 0] = zero
    return entropy(v1, v2)
    #return (t1*np.log(t1/t2)).sum()

def spearmanCorr(v1, v2, topN=None):
    ''' Spearman rank correlation between top vector elements. '''
    return topIndCorr(v1, v2, spearmanr, topN)

def pearsonCorr(v1, v2, topN=None):
    ''' Spearman rank correlation between top vector elements. '''
    return topIndCorr(v1, v2, pearsonr, topN)

def kendalltauCorr(v1, v2, topN=None):
    ''' Kendall-tau rank correlation between top vector elements. '''
    res = topIndCorr(v1, v2, kendalltau, topN)
    N = float(topN if topN is not None else len(v1))
    nres = 2*res/N; nres /= (N-1) # norm. in steps to avoid multiplying large ints
    return nres

def topIndOverlap(v1, v2, topN=20):
    ''' Calculates the number of coordinates withing topN values of both vectors. '''
    ti1, ti2 = topInd(v1, topN), topInd(v2, topN)
    return len(np.intersect1d(ti1, ti2))/float(topN)

def diceDist(v1, v2, topN=20): return setOverlapDist(v1, v2, dice, topN)
def jaccardDist(v1, v2, topN=20): return setOverlapDist(v1, v2, jaccard, topN)

def setOverlapDist(v1, v2, d, topN=20):
    ''' Compute set distance (jaccard or dice) operating on bool arrays,
     sets are formed by taking indices of top coordinates. '''
    ti1, ti2 = topInd(v1, topN), topInd(v2, topN)
    bw1, bw2 = np.zeros(len(v1), np.bool), np.zeros(len(v1), np.bool)
    bw1[ti1] = 1; bw2[ti2] = 1
    #assert set(ti1) == set(np.where(bw1 == 1)[0])
    #assert set(ti2) == set(np.where(bw2 == 1)[0])
    #print d(bw1, bw2)
    return d(bw1, bw2)

def topInd(v, topN):
    ''' Top-valued vector indices, unsorted. '''
    if len(v) <= topN: return np.array(range(len(v)))
    return np.argpartition(v, topN)[-topN:]

def topIndCorr(v1, v2, corr, topN=20):
    '''
    Calculate correlation coefficient between top vector elements, ie on coordinates
    that are among topN values in either v1 or v2.
    :param v1, v2: numpy arrays
    :param corr: function calculating the correlation from two vector params
    '''
    if topN is None: res = corr(v1, v2)
    else:
        ind = np.union1d(topInd(v1, topN), topInd(v2, topN))
        res = corr(v1[ind], v2[ind])
    if not isinstance(res, float): res = res[0]
    if np.isnan(res): return 0.0
    else: return res

@normalize
def hellinger(v1, v2):
    return euclidean(np.sqrt(v1), np.sqrt(v2)) / np.sqrt(2)

def bhattacharyya(v1, v2):
    return np.sum(np.sqrt(v1*v2))

def jensenShannon(m1, m2):
    if len(m1.shape) == 1:
        avg = (m1+m2)*0.5
        return 0.5*(kullbackLeibler(m1, avg)+kullbackLeibler(m2, avg))
    else:
        return apply2RowPairs(m1, m2, jensenShannon)

def cosine(m1, m2):
    if len(m1.shape) == 1: return cosineSp(m1, m2)
    else: return cdist(m1, m2, 'cosine')

def l1(m1, m2):
    if len(m1.shape) == 1: return minkowski(m1, m2, 1)
    else: return cdist(m1, m2, 'minkowski', p=1)

l1norm = normalize(l1)

def l2(m1, m2):
    if len(m1.shape) == 1: return euclidean(m1, m2)
    else: return cdist(m1, m2, 'euclidean')

l2norm = normalize(l2)

def l2squared(m1, m2):
    if len(m1.shape) == 1: return sqeuclidean(m1, m2)
    else: return cdist(m1, m2, 'sqeuclidean')

def lInf(m1, m2):
    if len(m1.shape) == 1: return chebyshev(m1, m2)
    else: return cdist(m1, m2, 'chebyshev')

def canberra(m1, m2):
    if len(m1.shape) == 1:
        vecSize = len(m1)
        return canberraSp(m1, m2) / vecSize
    else:
        vecSize = m1.shape[1]
        return cdist(m1, m2, 'canberra') / vecSize

canberraNorm = normalize(canberra)

# distance measures supporting batch calculation of distances between matrix rows
supportMatrixCalc = [cosine, l1, l2, l2squared, lInf, canberra]
def supportsBatch(metric): return metric in supportMatrixCalc

def batchDistance(m1, m2, metric, **params):
    '''Calculate distances between rows of two matrices.'''
    res = cdist(m1, m2, metric=metric, **params)
    return res

def rescaledDot(t1, t2):
    t1desc = np.sort(t1)[::-1]
    t2asc = np.sort(t2)
    t2desc = t2asc[::-1]
    dot = np.dot(t1,t2)
    mn = np.dot(t1desc,t2asc)
    mx = np.dot(t1desc,t2desc)
    return 1-(dot-mn)/(mx-mn)
