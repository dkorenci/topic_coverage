from topic_coverage.topicmatch.supervised_iter0 import topicDistances, topicDocDistances
from pytopia.measure.topic_distance import *
from pytopia.context.ContextResolver import resolve

import numpy as np

# setup shared cache, for multiprocessing, the assumption is that sklearn
# is setup to use multiprocessing as parallel execution backend
from multiprocessing import Manager
_manager = Manager()
_cache = _manager.dict()

class CachedSklearnTPFE():
    '''
    Class that handles scikit-learn compatibility, by implementing a transformer,
    and caching in multiprocessing-compatible way, as well as pickling.
    Wraps an stateless TopicPairFeatureExtractor that operates on topic pairs.
    '''

    def __init__(self, features=None, multiprocCache=True):
        '''
        :param features: string describing base feature extractor
        :param multiprocCache: if True, use multiprocessing Manager managed cache
        '''
        if multiprocCache: self._cache = _cache
        else: self._cache = dict()
        self.features = features
        self._createExtractor()
        self._multiproc = multiprocCache

    def switchOffMultiproc(self):
        ''' This can be desireable for models built with multiproc cache,
         but when loaded for single-proc use multiproc cache causes slowdown,
         especially for large number of processed examples. '''
        self._multiproc = False
        self._cache = dict()

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, 'features=%s'%self.features)

    def _createExtractor(self):
        if self.features is None: self._extractor = None
        else: self._extractor = TopicPairFeatureExtractor(self.features)

    def _getCacheAsPyDict(self):
        '''
        Return cache, which is multiprocessing-managed dict,
        as normal python dict, for serialization.
        '''
        if self._multiproc:
            c = self._cache
            if self._extractor is not None and len(c) > 0:
                # save only queries originated from this extractor, not entire shared cache
                d = { q:c[q] for q in c.keys() if self._queryNative(q) }
                return d
            else: return dict()
        else: return self._cache

    def _restoreCache(self, dict_):
        ''' Init cache and add dict_ mappings to it. '''
        if self._multiproc: self._cache = _cache
        else: self._cache = dict()
        for k, v in dict_.iteritems():
            if k not in self._cache: self._cache[k] = v

    def __getstate__(self):
        return self.features, self._multiproc, self._getCacheAsPyDict()

    def __setstate__(self, state):
        self.features, self._multiproc, pydict = state
        self._createExtractor()
        self._restoreCache(pydict)

    # sklearn interface methods
    def fit(self, data, y=None): return self
    def get_params(self, *args, **kwargs): return { 'features': self.features }
    def set_params(self, **params):
        if 'features' in params:
            features = params['features']
            if self.features != features:
                self.features = features
                self._createExtractor()

    def _query(self, pair):
        '''
        Form cache-level "query" from topic pair, this is necessary because the
        cache can be shared between extractors, so query must contain all the
        feat-extraction data (in addition to topic pair), to avoid clashes
        :param pair: pair of Topics
        '''
        p = (self._extractor.id, pair[0], pair[1])
        return p

    def _queryNative(self, query):
        ''' True if the cache "query" originated from object's extractor. '''
        return query[0] == self._extractor.id

    def _readCache(self, pair):
        ''' features for the topic pair, or None if not in cache '''
        q = self._query(pair)
        if q in self._cache: return self._cache[q]
        else: return None

    def _writeCache(self, pair, feats):
        q = self._query(pair)
        self._cache[q] = feats

    def transform(self, data):
        '''
        :param data: iterable of (Topic, Topic)
        :return: ndarray with pair features
        '''
        res = []
        for t1, t2 in data:
            pair = (t1.id, t2.id)
            feats = self._readCache(pair)
            if feats is None:
                feats = self._extractor(t1, t2)
                self._writeCache(pair, feats)
            res.append(feats)
        return np.array(res)

def spearmanCorrTop(v1, v2): return spearmanCorr(v1, v2, 20)

def pearsonCorrTop(v1, v2): return pearsonCorr(v1, v2, 20)

def kendalltauCorrTop(v1, v2): return kendalltauCorr(v1, v2, 20)

class TopicPairFeatureExtractor():

    def __init__(self, type='allmetrics'):
        self._type = type

    @property
    def id(self): return self._type

    def __call__(self, t1, t2):
        '''
        :param t1, t2: Topic-like objects
        :return: ndarray with pair's features
        '''
        if self._type == 'allmetrics':
            topd = topicDistances(t1, t2, allmetricsOld)
            docd = topicDocDistances(t1, t2, allmetricsOld)
            feats = np.concatenate((topd, docd))
        elif self._type == 'allmetrics2':
            metr = topicAndDocDists(t1, t2, metrics)
            feats = np.concatenate((metr, correlationAndOverlap(t1, t2)))
        elif self._type == 'cosine': feats = topicAndDocDists(t1, t2, [cosine])
        elif self._type == 'core1':
            feats = topicAndDocDists(t1, t2, [cosine, hellinger, l1norm, l2norm])
        elif self._type == 'core1nocos':
            feats = topicAndDocDists(t1, t2, [hellinger, l1norm, l2norm])
        elif self._type == 'core1nol2':
            feats = topicAndDocDists(t1, t2, [cosine, hellinger, l1norm])
        elif self._type == 'core2':
            feats = topicAndDocDists(t1, t2, [cosine, hellinger, l1norm, l2norm, kendalltauCorrTop])
            #feats = np.concatenate((feats, correlationAndOverlap(t1, t2)))
        elif self._type == 'core2nocos':
            feats = topicAndDocDists(t1, t2, [hellinger, l1norm, l2norm, kendalltauCorrTop])
        elif self._type == 'core3':
            feats = topicAndDocDists(t1, t2, [cosine, hellinger, l1norm, pearsonCorr])
        elif self._type == 'core3nocos':
            feats = topicAndDocDists(t1, t2, [hellinger, l1norm, pearsonCorr])
        elif self._type == 'valueindif':
            feats = topicAndDocDists(t1, t2, [cosine, hellinger, l1norm, l2norm])
            feats = np.concatenate((feats, correlationAndOverlap(t1, t2)))
        elif self._type == 'vectors':
            return topicVectors(t1, t2)
        elif self._type == 'allvectors':
            return allVectors(t1, t2)
        else: raise Exception('unknown featureset: %s'%self._type)
        return feats

# sets of topic distance functions
allmetricsOld = [cosine, klDivSymm, jensenShannon, l1, l2,
             canberra, spearmanCorr, pearsonCorr, hellinger, bhattacharyya]

metrics = [l1norm, l2norm, canberraNorm, hellinger]

def topicAndDocDists(t1, t2, metrics):
    '''
    :param metrics: set of metrics on topic pairs
    :return: vector of distances between topic-word and topic-doc vectors, for all metrics
    '''
    topd = topicDistances(t1, t2, metrics)
    docd = topicDocDistances(t1, t2, metrics)
    return np.concatenate((topd, docd))

def corpusTopicWeights(t):
    '''
    Create vector of topic proportions in corpus texts.
    Corpus is the corpus used to build the topic's model.
    '''
    model = t.model
    corpus = resolve(model).corpus
    cti = resolve('corpus_topic_index_builder')(corpus, model)
    tmx = cti.topicMatrix()
    return tmx[:, t.topicId]

def topicDistances(t1, t2, metrics=None):
    return [m(t1.vector, t2.vector) for m in metrics]

def topicVectors(t1, t2):
    return np.concatenate((t1.vector, t2.vector))

def allVectors(t1, t2):
    return np.concatenate((t1.vector, t2.vector,
                           corpusTopicWeights(t1), corpusTopicWeights(t2)))

def topicDocDistances(t1, t2, metrics):
    dvec1, dvec2 = corpusTopicWeights(t1), corpusTopicWeights(t2)
    return [m(dvec1, dvec2) for m in metrics]

def correlationAndOverlap(t1, t2):
    tv1, tv2 = t1.vector, t2.vector
    dv1, dv2 = corpusTopicWeights(t1), corpusTopicWeights(t2)
    corrs = []
    for corr in [spearmanCorr, pearsonCorr, kendalltauCorr]:
        ct = [corr(tv1, tv2, n) for n in [10, 20, 50, None]]
        cd = [corr(dv1, dv2, n) for n in [10, 20, 50, None]]
        corrs.extend(ct+cd)
    ot = [topIndOverlap(tv1, tv2, n) for n in [10, 20, 50]]
    od = [topIndOverlap(dv1, dv2, n) for n in [10, 20, 50]]
    jact = [jaccardDist(tv1, tv2, n) for n in [10, 20, 50]]
    jacd = [jaccardDist(dv1, dv2, n) for n in [10, 20, 50]]
    dicet = [diceDist(tv1, tv2, n) for n in [10, 20, 50]]
    diced = [diceDist(dv1, dv2, n) for n in [10, 20, 50]]
    cos = [cosine(tv1, tv2), cosine(dv1, dv2)]
    #res = np.concatenate((spt, spd, ot, od, cos))
    res = np.concatenate((corrs, ot, od, jact, jacd, dicet, diced, cos))
    #print ','.join('%5g'%v for v in res)
    return res

def featExtract(t1, t2, features, metrics):
    if features == 'distances': return topicDistances(t1, t2, metrics)
    elif features == 'vectors': return topicVectors(t1, t2)
    elif features == 'doc-distances': return topicDocDistances(t1, t2, metrics)
    elif features == 'all-distances':
        topd = topicDistances(t1, t2, metrics)
        docd = topicDocDistances(t1, t2, metrics)
        return np.concatenate((topd, docd))

def metricSet(mset='all'):
    if mset == 'all':
        metrics = [cosine, klDivSymm, jensenShannon, l1, l2, canberra,
                   spearmanCorr, pearsonCorr, hellinger, bhattacharyya]
    elif mset == 'kl': metrics = [klDivZero] #[klDivSymm]
    elif mset == 'metric': metrics = [l1, l2, lInf]
    elif mset == 'cosine': metrics = [cosine]
    elif mset == 'value-inv': # metrics invariant to concrete vector values, using angle, raknings, ...
        metrics = [cosine, pearsonCorr, spearmanCorr]
    elif mset == 'corr': metrics = [spearmanCorr, pearsonCorr]
    else: raise Exception('unknown metric set: %s' % mset)
    return metrics

# implement FeatureExtractor class - receives a pair, returns a feature vector
# .. class is the mapper of params to atomic classes implementim feature extraction
# .. can handle id-ability too
# implement feat-extractors: MetricSetFE, VectorFE
# .. alternatively, put all in TPFE
