from pytopia.context.ContextResolver import resolve, resolveId
from pytopia.tools.IdComposer import IdComposer

from scipy.stats import spearmanr, pearsonr
import numpy as np

class CorrProject():
    ''' Projects correlation funcs returning (corr, p-val) tuples to corr '''
    def __init__(self, corr): self._corr = corr
    def __call__(self, *args, **kwargs): return self._corr(*args, **kwargs)[0]

spearmanrnp, pearsonrnp = CorrProject(spearmanr), CorrProject(pearsonr)

def correlation(m1, m2, objects, corrFunc, m1app2set=False, m2app2set=False,
                verbose=False, bootstrap=False):
    '''
    Calculate correlation of two measures.
    :param m1, m2: function accepting an object (topic model, topic, ...), returning a number
    :param objects: list of either objects and/or lists of objects
            (in which case function value is averaged over list elements)
    :param corrFunc: function computing a correlation, or a list of such functions
    :param m1app2set, m2app2set: if True, apply m1/m2 to entire list, instead of averaging
            this is for application of stability functions
    :return: a single correlation value or list of correlations
    '''
    import numpy as np
    scores1, scores2 = [], []
    for o in objects:
        objlist = o if isinstance(o, list) else [o]
        if m1app2set == False: sc1 = np.average([m1(mod) for mod in objlist])
        else: sc1 = m1(objlist)
        scores1.append(sc1)
        if m2app2set == False: sc2 = np.average([m2(mod) for mod in objlist])
        else: sc2 = m2(objlist)
        scores2.append(sc2)
        if verbose:
            print objlist[0].id
            print '%g, %g' % (sc1, sc2)
    if isinstance(corrFunc, list):
        if bootstrap:
            for corr in corrFunc: bootstrap_correlation(corr, scores1, scores2, bootstrap)
        return [corr(scores1, scores2) for corr in corrFunc]
    else:
        if bootstrap:
            bootstrap_correlation(corrFunc, scores1, scores2, bootstrap)
        return corrFunc(scores1, scores2)

def bootstrap_correlation(corrFunc, scores1, scores2, numIter):
    import numpy
    from sklearn.utils import resample
    N = len(scores1)
    stats = list()
    for i in range(numIter):
        # prepare train and test sets
        sc1resamp, sc2resamp = resample(scores1, scores2, n_samples=N)
        score = corrFunc(sc1resamp, sc2resamp)
        if isinstance(score, tuple): stats.append(score[0])
        else: stats.append(score)
    for alpha in [0.95, 0.99]:
        lowp = ((1.0 - alpha) / 2.0) * 100
        highp = (alpha + ((1.0 - alpha) / 2.0)) * 100
        lower = numpy.percentile(stats, lowp)
        upper = numpy.percentile(stats, highp)
        name = corrFunc.__name__
        print('%12s: %.1f confidence interval [%g, %g]' % (name, alpha * 100, lower, upper))

class ModelCoverageFixref():
    '''
    Callable that returns, for a model, coverage of a referent model
    measured by a coverage function.
    '''

    def __init__(self, cov, ref):
        '''
        :param cov: coverage function accepting two topic models
        :param ref: referent model being "covered"
        '''
        self._cov, self._ref = cov, ref

    def __call__(self, model): return self._cov(self._ref, model)

class ModelAggTopicCoh():
    '''
    Calculation of model-level aggregation of topic functions.
    '''

    def __init__(self, f, params='object', agg='average'):
        '''
        :param f: callable returning a number for a topic
        :param params: if 'ids' f accepts (modelId, topicId), if 'object' f accepts Topic objects
        :param agg: method of agregation: 'average' or 'median'
        '''
        self._f, self._params, self._agg = f, params, agg

    def __call__(self, model):
        model = resolve(model); mid = model.id
        tvals = []
        for i, tid in enumerate(model.topicIds()):
            fval = None
            if self._params == 'ids': fval = self._f((mid, tid))
            elif self._params == 'object': fval = self._f(model.topic(tid))
            tvals.append(fval)
        if self._agg == 'average': return np.average(tvals)
        elif self._agg == 'median': return np.median(tvals)

class Topic2ModelMatch(IdComposer):
    '''Calculates weather a topic matches any topic in a fixed model. '''

    def __init__(self, model, matcher):
        '''
        :param model: TopicModel
        :param matcher: function receiving two topics and returning match result
        '''
        self.model, self.matcher = resolve(model), matcher
        IdComposer.__init__(self)

    def __call__(self, topic):
        for t in self.model:
            if self.matcher(t, topic): return 1
        return 0

class Topic2ModelDist(IdComposer):
    ''' Calculated clostest dist between a topic vector and vectors of topics of a model. '''

    def __init__(self, model, dist):
        '''
        :param model: TopicModel
        :param dist: distance function receiving two topics
        '''
        self.model, self.dist = resolve(model), dist
        IdComposer.__init__(self)

    def __call__(self, topic):
        mdist = None
        for t in self.model:
            d = self.dist(t.vector, topic.vector)
            if mdist is None or d < mdist: mdist = d
        return mdist
