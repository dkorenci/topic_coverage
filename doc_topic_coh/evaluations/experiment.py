from pytopia.tools.IdComposer import IdComposer, deduceId
from pytopia.resource.loadSave import pickleObject
from doc_topic_coh.evaluations.tools import labelMatch

from logging_utils.setup import *
from logging_utils.tools import fullClassName

from os import path
import cPickle

import sys, random
from traceback import print_exception
from multiprocessing import Pool
from sys_utils.multiprocess import unpinProcess

class IdList(list): pass
class IdDict(dict): pass

class ScorerParallelWrapper():
    '''
    Callable wrapper around topic coherence scorer for parallel processing.
    '''

    def __init__(self, scorer):
        self.__scorer = scorer

    def __call__(self, topic):
        unpinProcess()
        try:
            return self.__scorer(topic)
        except:
            e = sys.exc_info()
            print 'processing of topic %s failed' % str(topic)
            print_exception(e[0], e[1], e[2])
            return None

class TopicScoringExperiment(IdComposer):
    '''
    Experiment evaluating functions for scoring topic model topics
    using roc_auc measure on a list of topics.
    '''

    def __init__(self, paramSet, scorerBuilder, ltopics, posClass, folder,
                 verbose=True, cache=False, parallel=False):
        '''
        :param paramSet: list-like of paramsets passed to scorerBuilder
                each 'paramset' is a dict of parameters
        :param scorerBuilder: callable building scoring functions from a paramset
        :param ltopics: list of (topic, labels) - topic is (modelId, topicId),
                labels is a string or dict of 'label':{0 or 1}
        :param posClass: string or a list of strings defining which label is positive class
        :param verbose: if true, print results during experiment runtime
        :param cache: if True, force scorerBuilder to cache function results
        '''
        self.setOfParams = paramSet
        self.scorerBuilder = scorerBuilder.__name__
        self.__builder = scorerBuilder
        self.ltopics = ltopics
        self.posClass = posClass
        IdComposer.__init__(self, class_='TSE')
        self.verbose = verbose
        self.parallel = parallel
        self.__log = createLogger(fullClassName(self), INFO)
        self.baseFolder = folder
        self.folder = path.join(folder, self.id)
        if not path.exists(self.folder): os.mkdir(self.folder)
        self.cache = cache
        if cache:
            self.cacheFolder = path.join(self.baseFolder, 'function_cache')
            if not path.exists(self.cacheFolder): os.mkdir(self.cacheFolder)
        self.__rSetUp = False

    def __saveResults(self):
        cPickle.dump(self.resultTable, open(self.__resfile(), 'wb'))
        cPickle.dump(self.processedParams, open(self.__paramfile(), 'wb'))

    def __loadResults(self):
        self.resultTable = \
            cPickle.load(open(self.__resfile(), 'rb')) \
                if path.exists(self.__resfile()) else []
        self.processedParams = \
            cPickle.load(open(self.__paramfile(), 'rb')) \
                if path.exists(self.__paramfile()) else set()

    def __resfile(self): return path.join(self.folder, 'resultTable.pickle')
    def __paramfile(self): return path.join(self.folder, 'processedParams.pickle')

    def __msg(self, msg, level='info'):
        ''' Log message and print if self.verbose '''
        if level == 'info': self.__log.info(msg)
        elif level == 'warning': self.__log.warning(msg)
        if self.verbose:
            if level != 'warning': print msg
            else: print 'WARNING: %s' % msg

    def run(self):
        from time import time
        import gc
        gc.enable()
        self.__msg('STARTING EXPERIMENT: %s' % self.id)
        self.__loadResults()
        for p in self.setOfParams:
            pid = paramsId(p)
            print pid
            if pid in self.processedParams:
                self.__msg('params already processed: %s' % str(p))
                continue
            self.__msg('evaluating parameters: %s' % str(p))
            # make cp, 'call params' where additional non-algorithmic
            # parameters for builder are added
            if self.cache:
                cp = p.copy()
                cp['cache'] = self.cacheFolder
            else: cp = p
            scorer = self.__builder(**cp)()
            t = time()
            if 'parallel' in p: parallel = p['parallel']
            else: parallel = self.parallel
            if not parallel:
                score, classes, scores = self.__auc(scorer, self.ltopics, self.posClass, True)
            else:
                score, classes, scores = \
                    self.__aucParallel(scorer, self.ltopics, self.posClass, parallel)
            t = time()-t
            self.__msg('score %.4f , time %.4f' % (score, t))
            gc.collect()
            result = {}
            result['params'] = p.copy()
            result['scorerId'] = scorer.id
            result['roc_auc'] = score
            result['time'] = t
            result['pid'] = pid
            result['classes'] = classes
            result['scores'] = scores
            self.resultTable.append(result)
            self.processedParams.add(pid)
            self.__saveResults()

    def printResults(self, file=True, width=20, confInt=False):
        self.__loadResults()
        print self.id
        sr = sorted(self.resultTable, key=lambda r: -r['roc_auc'])
        if file: outf = open(self.__outfile(), mode='w')
        for i, r in enumerate(sr):
            if confInt:
                low, high = self.__confInterval(r['classes'], r['scores'])
                conf = ' [%5.3f, %5.3f]' % (low, high)
            else: conf = ''
            msg = '%3d: auc: %5.3f%s, time: %8.3f, %s, %s, %s' % \
                  (i, r['roc_auc'], conf, r['time'] ,
                   self.__paramSummary(r),
                    paramsId(r['params'], width, None), self.__formatParams(r['params']))
            print msg
            if file: outf.write(msg+'\n')

    def __paramSummary(self, r):
        '''
        :param r: evaluation result (a dict)
        :return:
        '''
        vecs = r['params']['vectors'] if 'vectors' in r['params'] else None
        return 'type:%18s, vectors:%12s' % (r['params']['type'], vecs)

    def __selectTopModels(self, deltaAuc=0.03, percentile=None, selectMin=10):
        '''
        Select results, ie. experimental data, for top performing models.
        Result table must be loaded.
        :return: list of selected results, selected[0] being the top performing model
        '''
        sr = sorted(self.resultTable, key=lambda r: -r['roc_auc'])
        topAuc = sr[0]['roc_auc']
        if percentile is None:
            selected = [ r for r in sr if abs(topAuc-r['roc_auc']) < deltaAuc ]
            if selectMin and len(selected) < selectMin:
                selected = sr[:1 + selectMin]
        else:
            import numpy as np
            aucs = [ r['roc_auc'] for r in sr ]
            perc = np.percentile(aucs, percentile*100)
            selected = [ r for r in sr if r['roc_auc'] >= perc ]
            print 'num. models above %.4f percentile: %d' % (perc, len(selected))
            if selectMin and len(selected) < selectMin:
                selected = sr[:selectMin]
        return selected

    def evalOnTopics(self, topics, deltaAuc=0.03, percentile=None, plotMin=10,
                     plot=False, save=True, saveDev=True):
        '''
        Eval performance of models close to the top model on a set of topics.
        :param topics: performance is evaluated on this set of topics
        :param deltaAuc: models within this distance from the top are plotted
        :param plotMin: take minimally this many top models, regardless of deltaAuc
        :return:
        '''
        from doc_topic_coh.evaluations.iteration6.croelect_topics import \
            croelectizeParamset, croelectize
        # sort models by auc, select top ones
        self.__loadResults()
        sr = sorted(self.resultTable, key=lambda r: -r['roc_auc'])
        if saveDev: # save the dev results
            import pickle
            eid = 'eval_%s_%s' % (self.setOfParams.id, self.ltopics.id)
            res = [ r['roc_auc'] for r in sr ]
            print 'DEV', eid
            pickle.dump(res, open(self.__evalfile(eid), 'wb'))
        # selected[0] must be the top dev model
        selected = self.__selectTopModels(deltaAuc, percentile, plotMin)
        # evaluate models
        results = []
        print 'NUM SELECTED: %d' % len(selected)
        for i, r in enumerate(selected):
            print 'evaluating (old): ', self.__formatParams(r['params'])
            params = croelectize(r['params'])
            if self.cache:
                params['cache'] = self.cacheFolder
            print 'evaluating: ', self.__formatParams(params)
            scorer = self.__builder(**params)()
            score, classes, scores = self.__auc(scorer, topics, self.posClass, False)
            results.append(score)
        if plot:
            from matplotlib import pyplot as plt
            fig, axes = plt.subplots(1, 1)
            axes.boxplot(results)
            axes.scatter([1] * len(results), results, alpha=0.5)
            axes.plot(1, results[0], 'ro')
            plt.show()
        if save: # save result list, top model in the first position
            import pickle
            eid = 'eval_%s_%s' % (self.setOfParams.id, topics.id)
            print 'TST', eid
            pickle.dump(results, open(self.__evalfile(eid), 'wb'))

    def evalOnTopicsPrintTop(self, topics, thresh, th2per=None,
                             deltaAuc=0.03, percentile=None, selectMin=10):
        '''
        Eval performance of models close to the top model on a set of topics.
        Print selection on these evaluated models with top performance.
        :param topics: performance is evaluated on this set of topics
        :param thresh: models are printed if eval performance is above thresh
        :param deltaAuc, percentile, selectMin: pre-eval selection of top models
        :return:
        '''
        # sort models by auc, select top ones
        self.__loadResults()
        # sr = sorted(self.resultTable, key=lambda r: -r['roc_auc'])
        # selected[0] must be the top dev model
        selected = self.__selectTopModels(deltaAuc, percentile, selectMin)
        # evaluate and select models
        results = []
        for i, r in enumerate(selected):
            params = r['params']
            if self.cache:
                params['cache'] = self.cacheFolder
            #print 'evaluating: ', self.__formatParams(params)
            scorer = self.__builder(**params)()
            score, classes, scores = self.__auc(scorer, topics, self.posClass, False)
            results.append((score, classes, score))
        import numpy as np
        if thresh == 'median':
            thresh = np.median([r[0] for r in results])
        sortSel = np.argsort([r[0] for r in results])[::-1]
        for i in sortSel:
            score, classes, scores = results[i]
            r = selected[i]
            if score >= thresh:
                params = r['params']
                devScore, evalScore = r['roc_auc'], score
                #print self.__formatParams(params)
                #print 'dev %.3f , eval %.3f' % (devScore, evalScore)
                if params['type'] == 'graph' and 'weightFilter' in params:
                    th = params['weightFilter'][1]
                    #print 'threshold percentile: ', \
                    thPerc = th2per(params['vectors'], params['distance'].__name__, th)
                    params['thresh_perc'] = thPerc
                print self.__printAsLatexRow(params, devScore, evalScore)


    def __printAsLatexRow(self, params, dev, test):
        '''
        Return params and results formatted as latex table row
        :param params: coherence measure params
        :param dev, test: AUC scores
        '''
        def lvec(params):
            vec = params['vectors']
            if vec == 'probability': return 'bag-of-words'
            elif vec == 'tf-idf': return vec
            else: # glove, glove-avg, word2vec, word2vec-avg
                dist = params['distance'].__name__ if 'distance' in params else None
                if dist == 'cosine':
                    if vec.endswith('-avg'): return vec[:-4]
                    else: return vec
                else:
                    if not vec.endswith('-avg'): return vec+'-sum'
                    else: return vec
        if params['type'] == 'graph':
            def lalgo(algo):
                if algo == 'communicability': return 'subgraph'
                else: return algo
            return \
            '\cpv{%s} & $%d$ & \cpv{%s} & $%.2f$ & \cpv{%s} & \cpv{%s} & %.3f & %.3f \\\\' % \
                (lvec(params), params['threshold'], params['distance'].__name__,
                 params['thresh_perc'], params['weighted'], lalgo(params['algorithm']),
                 dev, test
                 )
        elif params['type'] in ['avg-dist', 'variance']:
            def lagg(algo):
                if algo == 'avg-dist': return 'average'
                else: return algo
            return \
            '\cpv{%s} & $%d$ & \cpv{%s} & \cpv{%s} & %.3f & %.3f \\\\' % \
            (lvec(params), params['threshold'], params['distance'].__name__,
                lagg(params['type']), dev, test)
        elif params['type'] == 'density':
            def ldimr(dimr):
                if dimr == None: return '\cpv{None}'
                else: return '$%d$'%dimr
            def lcovar(cov):
                if cov == 'spherical': return 'scalar'
                elif cov == 'diag': return 'diagonal'
            return \
            '\cpv{%s} & $%d$ & %s & \cpv{%s} & %.3f & %.3f \\\\' % \
            (lvec(params), params['threshold'],
               ldimr(params['dimReduce']), lcovar(params['covariance']), dev, test)
        else: raise Exception('unknown coherence type: %s'%params['type'])


    def __evalfile(self, eid):
        return path.join(self.folder, '%s.pickle'%eid)

    def testSignificance(self, s1, s2, sort=True, N=2000000, seed=8847, method = "bootstrap"):
        '''
        Test statistical significance of difference in two scorers AUCs.
        :param s1: index of scorer s1
        :param s2: index of scorer s2
        :param sort: indexes should be applied to results sorted by AUC, descending
        :param N: number of random shuffles used to calculate significance
        :return: p-value for significance
        '''
        self.__loadResults()
        res = self.resultTable if not sort else sorted(self.resultTable, key=lambda r: -r['roc_auc'])
        r1, r2 = res[s1], res[s2]
        assert r1['classes'] == r2['classes']
        #sig = self.__permutationTest(r1['classes'], r1['scores'], r2['scores'], N, seed)
        #sig = self.__boostrapTest(r1['classes'], r1['scores'], r2['scores'], N, seed)
        sig = self.__procTest(r1['classes'], r1['scores'], r2['scores'], N, seed, method)
        print 'roc1 %.4f, roc2 %.4f significance p-value: %.4f' \
              % (r1['roc_auc'], r2['roc_auc'], sig)

    def significance(self, scoreInd=None, threshold=0.01, N=10000, seed=8847,
                     method ="delong", correct='holm'):
        '''
        Test statistical significances between to scorer and other scorers.
        :param scoreInd: indexes of scorers to compare
        :param threshold: if not None, print only significances above the threshold
        :param N: number of random shuffles used to calculate significance
        :param correct: None or method param for R's p.adjust() p-value adjusting method
        :return: p-value for significance
        '''
        self.__loadResults()
        res = sorted(self.resultTable, key=lambda r: -r['roc_auc'])
        if not scoreInd: scoreInd = range(0, len(res))
        rtop = res[scoreInd[0]]
        sigs = []
        for si in scoreInd[1:]:
            r = res[si]
            sig = self.__procTest(rtop['classes'], rtop['scores'], r['scores'],
                                  N, seed, method)
            sigs.append(sig)
        if correct:
            if isinstance(correct, basestring): sigs = self.__correct(sigs, correct)
            else: sigs = self.__correct(sigs)
        print 'auc top %.3f' % rtop['roc_auc'], rtop['params']
        for i, si in enumerate(scoreInd[1:]):
            r, sig = res[si], sigs[i]
            if not threshold or sig > threshold:
                print 'auc %3d %.3f p-value: %.3f' % (si, r['roc_auc'], sig), r['params']
        # print only parameters, in format of list of dicts
        print '['
        for i, si in enumerate(scoreInd[1:]):
            r, sig = res[si], sigs[i]
            if not threshold or sig > threshold:
                 print self.__formatParams(r['params']) + ','
        print ']'

    def __formatParams(self, p):
        '''
        Format coherence measure parameters as string representing python dict,
        so that it can be usable in code without need for corrections.
        :param p: dict of 'param_name': param_value
        '''
        def modval(val):
            if hasattr(val, '__name__'): return val.__name__
            elif isinstance(val, basestring):
                return '\'%s\'' % val
            else: return str(val)
        return '{%s}'% \
               (', '.join('%s: %s'%(modval(pname), modval(pval))
                          for pname, pval in p.iteritems() if pname != 'cache' ))

    def __initR(self):
        if self.__rSetUp == True: return
        import rpy2.robjects as ro
        self.R = ro.r
        self.R('require(pROC)')
        self.R('options(warn=-1)')
        self.R('sprintf <- function(...) {}')
        self.R('deparse <- function(...) {}')
        self.__rSetUp = True

    def __correct(self, pvalues, method):
        self.__initR()
        padj = self.R['p.adjust']
        res = padj(pvalues, method=method)
        return res

    def __confInterval(self, classes, scores, level=0.95, method='bootstrap', N=2000):
        from rpy2.robjects.vectors import FloatVector, IntVector
        self.__initR()
        roc = self.R['roc']
        ciAuc = self.R['ci.auc']
        classes, scores = IntVector(classes), FloatVector(scores)
        rocO = roc(classes, scores)
        params = {'conf.level':level, 'boot.n':N}
        res = ciAuc(rocO, method=method, parallel=True, **params)
        # print res
        # for i, e in enumerate(res): print i, ':', e
        return res[0], res[2]

    def __procTest(self, classes, scores1, scores2, N, rseed, method = "bootstrap"):
        from rpy2.robjects.vectors import FloatVector, IntVector
        self.__initR()
        rocTest = self.R['roc.test']
        roc = self.R['roc']
        scores1, scores2 = FloatVector(scores1), FloatVector(scores2)
        classes = IntVector(classes)
        roc1 = roc(classes, scores1)
        roc2 = roc(classes, scores2)
        params={'boot.n':N}
        paired = True
        res = rocTest(roc1, roc2, paired=paired, method=method,
                      #alternative="greater",
                      alternative="two.sided",
                      parallel=True, **params)
        # print res
        # for i, e in enumerate(res):
        #     print 'EL', i, res[i]
        if method == "bootstrap": return float(res[7][0])
        else:
            if paired: return float(res[6][0])
            else: return float(res[7][0])
        # res = rocTest(response=classes, predictor1=scores1, predictor2=scores2,
        #               na_rm = True, method=None, alternative="two.sided")
        #print rocTest(response=classes, predictor1=scores1, predictor2=scores2)

    def __boostrapTest(self, classes, scores1, scores2, N, rseed):
        '''
        Test statistical significance of difference in two scorers AUCs
         using boostraping of samples.
        :param classes: array of true 0-1 classes
        :param scores1: ranks assigned by scorer1
        :param scores2: ranks assigned by scorer2
        :return:
        '''
        import numpy as np
        from sklearn.metrics import roc_auc_score as auc
        from numpy.random import randint, seed
        seed(rseed)
        classes = np.array(classes)
        scores1, scores2 = np.array(scores1), np.array(scores2)
        auc1, auc2 = auc(classes, scores1), auc(classes, scores2)
        aucDiff = abs(auc1 - auc2)
        print auc1, auc2
        L = len(classes)
        nge = 0
        for i in range(N):
            ind = randint(0, L, 2*L)
            sc, ss1, ss2 = classes[ind], scores1[ind], scores2[ind]
            auc1, auc2 = auc(sc, ss1), auc(sc, ss2)
            if abs(auc1 - auc2) >= aucDiff: nge += 1
        return (nge+1.0)/(N+0.1)

    def __permutationTest(self, classes, scores1, scores2, N, rseed, boolArrays=True):
        '''
        Test statistical significance of difference in two scorers AUCs
         using randomized stratified shuffling test.
        :param classes: array of true 0-1 classes
        :param scores1: ranks assigned by scorer1
        :param scores2: ranks assigned by scorer2
        :return:
        '''
        import numpy as np
        # generate base for scorer response - all pairs of (positiveClass, negativeClass)
        posi = [i for i, c in enumerate(classes) if c == 1]
        negi = [i for i, c in enumerate(classes) if c == 0]
        pni = [(pi, ni) for pi in posi for ni in negi]
        baseSize, L = len(pni), float(len(pni))
        resp1 = [int(scores1[pi] > scores1[ni]) for pi, ni in pni]
        resp2 = [int(scores2[pi] > scores2[ni]) for pi, ni in pni]
        auc1, auc2 = sum(resp1)/L, sum(resp2)/L
        print auc1, auc2
        aucDiff = abs(auc1-auc2)
        print aucDiff
        from numpy.random import seed, binomial
        seed(rseed)
        memoryPerBatch = 300000000
        batchSize = memoryPerBatch / baseSize
        processed = 0; nge = 0; cnt = 0
        print 'batchSize %d , numBatches %.2f' % (batchSize, N / float(batchSize))
        while (processed < N):
            if processed+batchSize > N:
                batchSize = N-processed
            processed += batchSize
            select = binomial(1, 0.5, (batchSize, int(L)))
            if boolArrays: select = select.astype(np.bool, copy=True)
            responses = np.array([resp1, resp2])
            if boolArrays: responses = responses.astype(np.bool, copy=True)
            shuff1 = np.choose(select, responses)
            invSelect = ~select if boolArrays else 1-select
            shuff2 = np.choose(invSelect, responses)
            shauc1 = np.sum(shuff1, 1)/L
            shauc2 = np.sum(shuff2, 1)/L
            nge += np.sum(np.abs(shauc1-shauc2) >= aucDiff)
            cnt+=1
            print 'batch %d finished' % cnt
        return (nge+1.0)/(N+1.0)

    def __outfile(self): return path.join(self.folder, 'results.txt')

    def __auc(self, scorer, ltopics, label, tolerant=False):
        '''
        Calculate area under roc curve for task of classifying topics
        using a specified topic measure.
        :param scorer: topic measure, callable mapping a topic to number
        :param ltopics: labeled topics, list of (topic, label)
        :param label: label representing the positive class
        '''
        from sklearn.metrics import roc_auc_score
        from traceback import print_exception
        import sys
        mvalues, classes = [], []
        errors = 0
        for t, tl in ltopics:
            if not tolerant:
                mv = scorer(t)
            else:
                try:
                    mv = scorer(t)
                except:
                    e = sys.exc_info()
                    if e[0] == KeyboardInterrupt: raise KeyboardInterrupt
                    self.__msg('processing of topic %s failed' % str(t), 'warning')
                    print_exception(e[0], e[1], e[2])
                    mv = None
                    errors += 1
            if mv is not None:
                mvalues.append(mv)
                classes.append(labelMatch(tl, label))
        # classes = [ for _, tl in ltopics ]
        if errors:
            self.__msg('out of %d, %d measure calculations failed' % (len(ltopics), errors))
        return roc_auc_score(classes, mvalues), classes, mvalues

    def __aucParallel(self, scorer, ltopics, label, numProcesses):
        '''
        Calculate area under roc curve for task of classifying topics
        using a specified topic measure.
        :param scorer: topic measure, callable mapping a topic to number
        :param ltopics: labeled topics, list of (topic, label)
        :param label: label representing the positive class
        '''
        from sklearn.metrics import roc_auc_score
        from copy import copy
        # shuffle topics for load balancing
        ltopics = copy(ltopics)
        random.seed(42532543)
        random.shuffle(ltopics)
        # perform parallel calculation
        p = Pool(numProcesses)
        topics, labels = [ t for t, tl in ltopics ], [ tl for t, tl in ltopics ]
        result = p.map(ScorerParallelWrapper(scorer), topics)
        # process results and calculate auc
        errors = 0
        classes, coherences = [], []
        for i, c in enumerate(result):
            if c is not None:
                coherences.append(c)
                classes.append(labelMatch(labels[i], label))
            else: errors += 1
        if errors:
            self.__msg('out of %d, %d measure calculations failed' % (len(ltopics), errors))
        return roc_auc_score(classes, coherences), classes, coherences

def paramsId(params, width=None, header='params'):
    '''
    Converts map to hashable string uniquely identifying it.
    Keys are assumed to be strings.
    Values are transformed if they are functions or methods.
    '''
    def valueStr(obj):
        import types
        if obj == None: return None
        if hasattr(obj, 'id'):
            return obj.id
        else:
            if isinstance(obj, types.FunctionType):
                return obj.__name__
            elif isinstance(obj, types.ClassType):
                return obj.__name__
            else:
                return str(obj)
    wmod = '%s' if width is None else '%'+('%ds'%width)
    p = ','.join( wmod  % ('%s:%s' % (key, valueStr(val)))
                  for key, val in params.iteritems() )
    if header: res = '%s[%s]'%(header, p)
    else: res = p
    return res



