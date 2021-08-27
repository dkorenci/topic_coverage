from pytopia.tools.IdComposer import IdComposer
from pytopia.topic_cluster.generic_functionality import TopicClusteringHelper

import numpy as np

class KmedoidTopicCluster(IdComposer):

    def __init__(self, numClusters, distance, seed=12345):
        '''
        :param numClusters: integer or 'auto'
        :param distance: distance function on two topic vectors
        '''
        self.numClusters = numClusters
        self.distance = distance
        self.seed = seed
        IdComposer.__init__(self)

    def __call__(self, topics):
        helper = TopicClusteringHelper(topics)
        helper.createTopicMatrix()
        dist = self.distance(helper.topicMatrix, helper.topicMatrix)
        np.fill_diagonal(dist, 0.0)
        kmc = KmedoidClusterer(self.numClusters, numIter=10, seed=self.seed)
        kmc.fit(dist)
        return helper.clusteringFromScikit(kmc)

class KmedoidClusterer():
    '''
    Scikit-learn compatible k-medoid clustering.
    '''
    def __init__(self, numClusters, numIter=100, seed=12345):
        self.numClusters = numClusters
        self.numIter = numIter
        self.seed = seed

    def fit(self, D):
        '''
        :param D: distance matrix encoding a metric
        :return:
        '''
        self.N, _ = D.shape
        self.D = D
        if isinstance(self.numClusters, (int, float)):
            ncl = self.numClusters if self.numClusters <= self.N else self.N
            med, cl = kMedoids(self.D, ncl, self.numIter, self.seed)
        elif isinstance(self.numClusters, tuple):
            ncl, med, cl = self.__chooseNumClusters()
        else: raise Exception('num.clusters must be either a number or (min, max, step) triple')
        self.labels_ = self.__clustersToLabels(cl, ncl)
        self.cluster_centers_indices_ = med

    def __chooseNumClusters(self, averageOver=5):
        '''
        Cluster with different numbers of clusters and return
         best clustering according to given criteria.
         :param averageOver: to obtain better stability of clustering quaility
            for each number of clusters, run clustering this many times
        '''
        min, max, step = self.numClusters
        maxScore, bestMed, bestCl, bestNcl = None, None, None, None
        for ncl in range(min, max+1, step):
            # locally (over average tries) best solutions
            lmaxScore, lbestMed, lbestCl = None, None, None
            avgScore = 0.0
            for i in range(averageOver):
                med, cl = kMedoids(self.D, ncl, self.numIter, self.seed+i)
                labels = self.__clustersToLabels(cl, ncl)
                #score = silhouette_score(self.D, labels, 'precomputed')
                score = -(2*ncl - 2*clusterLogLikelihood(self.D, labels, med, l=0.1))
                avgScore += score
                if lmaxScore is None or score > lmaxScore:
                    lmaxScore = score
                    lbestMed, lbestCl = med, cl
            score = avgScore / averageOver
            print '%d %.3f' % (ncl, score)
            if maxScore is None or score > maxScore:
                maxScore = score
                bestMed, bestCl, bestNcl = lbestMed, lbestCl, ncl
        return bestNcl, bestMed, bestCl

    def __clustersToLabels(self, clusters, ncl):
        '''
        :param clusters: cluster index -> list of cluster members mapping
        :param ncl: indexes are 0..numClusters-1
        :return: array element index -> cluster index
        '''
        labels_ = np.empty(self.N, dtype='int')
        for k in xrange(ncl):
            for e in clusters[k]: labels_[e] = k
        return labels_

from math import log
def clusterLogLikelihood(D, labels, med, l=1.0):
    '''
    Compute log-likelihood of data from clusters.
    For each datapoint x belonging to cluster with medioid m(x),
    likelihood is proportional to exp(-d(x, m(x)))
    :param D: distance matrix
    :param labels: cluster labels
    :param lambda: exp. distribution parameter
    :return:
    '''
    N, _ = D.shape
    return N*log(l) - l*sum(D[x, med[labels[x]]] for x in range(N))


def kMedoids(D, K, maxIter=100, seed=12345, init='kmeans++'):
    '''
    Slightly adapted clustering code from publication:
    NumPy / SciPy Recipes for Data Science: k-Medoids Clustering
    (https://www.researchgate.net/publication/272351873)
    :param D: distance matrix encoding a metric
    :param K: number of clusters
    :param maxIter: max. number of iterations
    :return:
    '''
    m, n = D.shape # determine dimensions of distance matrix D
    assert m == n
    if n < K: K = n # if the num of samples is less then num. clusters, make it equal
    np.random.seed(seed)
    if init == 'kmeans++': centers = kmeansppInit(D, K)
    elif init == 'random': centers =  np.random.choice(n, K, replace=False)
    M = np.sort(centers)
    Mnew = np.copy(M)
    C = {} # initialize a dictionary to represent clusters
    for t in range(maxIter):
        # determine clusters, i.e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(K):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(K):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        Mnew = np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew): break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(K):
            C[kappa] = np.where(J==kappa)[0]
    return M, C

def kmeansppInit(D, K, verbose=False):
    '''
    Choose K initial cluster centroids using the method from kmeans++.
    :param D: distance matrix encoding a metric
    :param K: number of clusters
    :return: indexes of cluster centroids
    '''
    n, _ = D.shape
    c = []
    c.append(np.random.choice(n, 1)[0])
    rem = set(range(n)) # set of remaining elements to choose clusters from
    rem.remove(c[0])
    for i in range(1, K):
        J = np.argmin(D[:, c], axis=1)
        # get distance to closest cluster for each point
        d = np.array([D[e, c[J[e]]] for e in range(n)]) # todo: implement with numpy
        if verbose:
            print c
            print J
            print [x for x in d]
        d /= d.sum()
        reml = list(rem) # fix ordering
        newi = np.random.choice(len(reml), 1, p=[d[e] for e in reml])
        newc = reml[newi[0]]
        c.append(newc); rem.remove(newc)
    if verbose: print c
    return np.array(c)


from pytopia.measure.topic_distance import cosine as cosineDist
from pytopia.topic_cluster.testing import runTopicModelsTest

def basicTest():
    hac = KmedoidTopicCluster(50, cosineDist)
    print hac.id

def test():
    basicTest()
    clusters1 = runTopicModelsTest(KmedoidTopicCluster, distance=cosineDist, numClusters=5)
    assert len(clusters1) == 5
    clusters2 = runTopicModelsTest(KmedoidTopicCluster, distance=cosineDist, numClusters=10)
    assert len(clusters2) == 10

if __name__ == '__main__':
    test()
