'''
Wrapper class and factory methods for sklearn clusterers.
'''

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from pytopia.tools.IdComposer import IdComposer

class Clustering(IdComposer):
    '''
    Pytopia wrapper for sklearn clustering algorithms.
    Provides pytopia id-ability and adapts to skelarn algorithm interface.
    '''

    def __init__(self, cls, **params):
        '''
        :param cls: class of the sklearn Clusterer
        :param params:
        '''
        atts = []
        for name, value in params.iteritems():
            self.__dict__[name] = value
            atts.append(name)
        className = cls.__name__
        IdComposer.__init__(self, attributes=atts, class_=className)
        self.__params = params
        self.__cls = cls

    def fit(self, X, y=None):
        self.__clusterer = self.__cls(**(self.__params))
        self.__clusterer.fit(X, y)

    def __getattr__(self, name):
        if hasattr(self.__clusterer, name):
            return getattr(self.__clusterer, name)
        elif name in self.__params: return self.__params[name]
        else: return None

    def __setattr__(self, key, value):
        if key.startswith('_') or key.startswith('__'):
            self.__dict__[key] = value
        else:
            self.__params[key] = value
            #setattr(self.__clusterer, key, value)

def kmeans(init='k-means++'):
    return Clustering(KMeans, init=init, n_clusters=2)

def spectral(affinity, n_neighbours=10, kernel_params=None):
    return Clustering(SpectralClustering, affinity=affinity, n_neighbors=n_neighbours,
                            kernel_params=kernel_params, n_clusters=2)

def hac(linkage, affinity):
    return Clustering(AgglomerativeClustering, linkage=linkage,
                      affinity=affinity, n_clusters=2)

if __name__ == '__main__':
    cl = Clustering(SpectralClustering, affinity='cosine', n_neighbors=10,
                            n_clusters=2, random_state=566, n_jobs=3)
    print cl.id
    cl.n_jobs = 33
    cl.someAttr = 44