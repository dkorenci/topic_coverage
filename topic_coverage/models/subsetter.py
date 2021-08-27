from pytopia.topic_model.TopicModel import TopicModel, Topic
from pytopia.tools.IdComposer import IdComposer, createId
from pytopia.context.ContextResolver import resolveIds, resolve
from pytopia.resource.ResourceBuilder import SelfbuildResourceBuilder
from file_utils.location import FolderLocation as loc
from pytopia.resource.loadSave import pickleObject, loadResource
from pytopia.context.Context import Context

import numpy as np
from scipy.stats import entropy as scipyEntropy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from os import path
from math import log
import math, copy, cPickle, os

def entropy(td):
    ''':param td: TextData'''
    return scipyEntropy(td.topics)

def entropyLL(td):
    '''
    Entropy times log doc length.
    :param td: TextData
    '''
    return scipyEntropy(td.topics)*log(td.numWords)

class Uncov2ProbLogistic():
    def __init__(self, steepness=1.0, ceiling=1.0):
        self.steepness = steepness
        self.ceiling = ceiling

    @property
    def id(self): return '%s[steep_%.4f;ceil_%.4f]' % \
                         (self.__class__.__name__, self.steepness, self.ceiling)

    def fit(self, vals):
        '''
        Fit self to uncoverdness values,
        so that __call__(uncovVal) can return (relative) probability
        '''
        mean = np.mean(vals)
        self.vecF = lambda v: self.ceiling / (1.0 + np.exp((v-mean) * (-self.steepness)))
        self.scalF = lambda v: self.ceiling / (1.0 + math.exp((v-mean) * (-self.steepness)))

    def __call__(self, uncov):
        '''
        :param uncov: ndarray or a single number
        :return: value(s) mapped to relative probabilities, ndarray or a single number
        '''
        if isinstance(uncov, np.ndarray): return self.vecF(uncov)
        else: return self.scalF(uncov)

class Subsetter(TopicModel, IdComposer):

    def __init__(self, corpus, dictionary, text2tokens, tmpFolder,
                 builder, builderParams, numIter, uncov=entropyLL,  buildId=None, paramsId=None,
                 uncov2prob=Uncov2ProbLogistic(), rseed=12345):
        '''
        :param corpus, dictionary, text2tokens: basic topic model building resources
        :param builder: Pytopia topic model builder.
        :param builderParams: param map for the builder, one set of params of a list (for each iteration)
        :param buildId: information for id-composition, describing builder params and other build-relevant info
        :param numIter: number of iterations to run
        '''
        self.corpus, self.dictionary, self.text2tokens = \
            resolveIds(corpus, dictionary, text2tokens)
        # todo replace createId with resolveId
        # when implementing formating of ids for (decorated) builder classes
        self.builder, self.builderId = builder, createId(builder)
        self.builderParams, self.buildId, self.paramsId = builderParams, buildId, paramsId
        self.numIter = numIter
        # todo solve id-objects pickling in pytopia spirit
        self.uncov, self.uncovId = uncov, createId(uncov)
        self.uncov2prob, self.uncov2probId = uncov2prob, createId(uncov2prob)
        self.tmpFolder = tmpFolder
        self.rseed = rseed
        idParams = ['corpus', 'dictionary', 'text2tokens', 'builderId', 'buildId', 'paramsId',
                                   'rseed', 'numIter', 'uncovId', 'uncov2probId']
        IdComposer.__init__(self, idParams)

    def topicIds(self): return range(self._numTopics)

    def numTopics(self): return self._numTopics

    def topicVector(self, topicId): return self._topics[topicId]

    def _setupTmpFolder(self):
        if not path.exists(self.tmpFolder): os.mkdir(self.tmpFolder)
        for f in loc(self.tmpFolder).files(): os.remove(f)

    def _createBuildersContext(self):
        ''' Create Context with subbsetter-level builders  '''
        from pytopia.context.Context import Context
        from pytopia.resource.builders_context import cachedResourceBuilder
        from pytopia.resource.corpus_index.CorpusIndex import CorpusIndexBuilder
        from pytopia.resource.corpus_topics.CorpusTopicIndex import CorpusTopicIndexBuilder
        from pytopia.resource.bow_corpus.bow_corpus import BowCorpusBuilder
        from file_utils.location import FolderLocation
        ctx = Context('subsetter_builders_context[%s]'%self.tmpFolder)
        cf = FolderLocation(self.tmpFolder)
        ctx.add(cachedResourceBuilder(
            CorpusIndexBuilder(), cf('corpus_index'), id='corpus_index_builder'))
        ctx.add(cachedResourceBuilder(
            BowCorpusBuilder(), cf('bow_corpus'), id='bow_corpus_builder'))
        ctx.add(cachedResourceBuilder(
            CorpusTopicIndexBuilder(), cf('corpus_topic_index'), id='corpus_topic_index_builder'))
        return ctx

    def _initLocalContext(self):
        '''Create local context, with local (tmp) builders and resampled corpora. '''
        ctx = Context('subsetter_local_contex')
        ctx.merge(self._createBuildersContext())
        self._localContext = ctx

    def _addToLocalContext(self, obj): self._localContext.add(obj)

    def _builder(self, i):
        '''builder for the i-th iteration'''
        from pytopia.resource.builders_context import cachedResourceBuilder
        from file_utils.location import FolderLocation
        builder = indexOrReturn(self.builder, i)
        cf = FolderLocation(self.tmpFolder)
        bid = 'subsetter_models_builder[%s]' % (builder.__class__.__name__)
        return cachedResourceBuilder(builder, cf(bid), id=bid)

    def _bparams(self, i):
        '''builder params for the i-th iteration'''
        return indexOrReturn(self.builderParams, i)

    def _addTopics(self, model):
        '''
        Add topics of the model to the pool of topics
        :param model: TopicModel compatible object
        '''
        if not hasattr(self, '_topics'):
            dict = resolve(self.dictionary)
            self._numTopics = model.numTopics()
            self._maxIndex = dict.maxIndex()
            self._topics = np.empty((self._numTopics, self._maxIndex+1), np.float64)
            for i, ti in enumerate(model.topicIds()):
                self._topics[i] = model.topicVector(ti)
        else:
            self._topics.resize((self._numTopics+model.numTopics(), self._maxIndex+1))
            for i, ti in enumerate(model.topicIds()):
                self._topics[self._numTopics+i] = model.topicVector(ti)
            self._numTopics += model.numTopics()

    def build(self):
        self._setupTmpFolder()
        self._initLocalContext()
        # todo create custom cached resource builders for model, cti and ci and bowcorp
        corpus = resolve(self.corpus)
        sscorpus = corpus
        for i in range(self.numIter):
            model = None
            with self._localContext:
                builder, params = self._builder(i), self._bparams(i)
                params['text2tokens'], params['dictionary'] = self.text2tokens, self.dictionary
                params['corpus'] = sscorpus
                mid = builder.resourceId(**params)
                model = resolve(mid)
                if model is None:
                    print 'model not in context, building: %s' % mid
                    model = builder(**params)
                else: print 'model found in context: %s' % mid
            self._addTopics(model) # add model's topics to pool of topics
            self._addToLocalContext(model)
            # if its the last model built, its not necessary to resample documents
            if i == self.numIter - 1: break
            ## doc-topic proportions building
            cti = None
            with self._localContext:
                ctibuilder = resolve('corpus_topic_index_builder')
                cti = ctibuilder(corpus, model, self.dictionary, self.text2tokens)
                # for calculating doc. lengths
                bowCorpus = resolve('bow_corpus_builder')(corpus, self.text2tokens, self.dictionary)
                corpind = resolve('corpus_index_builder')(corpus)
            ## calculating 'uncoveredness' and resampling the corpus
            uncovs, txtIds = [], []
            for txto in corpus:
                txtIds.append(txto.id)
                # todo calc. num words, with bow corpus
                tt = cti.textTopics(txto.id)
                ttv = textTopics2Vector(model, tt)
                bow = bowCorpus[corpind.id2index(txto.id)]
                numWords = sum(n for _, n in bow)
                uncovs.append(self.uncov(TextData(ttv, numWords, txto.id)))
            self.uncov2prob.fit(uncovs)
            # resample text ids from uncoveredness
            N = len(corpus)
            uncovs = np.array(uncovs)
            print uncovs
            docProbs = self.uncov2prob(uncovs)
            docProbs /= docProbs.sum()
            np.random.seed(self.rseed)
            idResample = np.random.choice(txtIds, N, replace=True, p=docProbs)
            # create corpus of resampled texts
            sscorpusId = 'params[%s]_builder[%s]_iter[%d]_seed[%g]' % (self.paramsId, self.buildId, i, self.rseed)
            sscorpus = ResampledCorpus(corpus, idResample, sscorpusId)
            self._addToLocalContext(sscorpus)

    ### persistence

    def __getstate__(self):
        return IdComposer.__getstate__(self), self._numTopics

    def __setstate__(self, state):
        IdComposer.__setstate__(self, state[0])
        self._numTopics = state[1]

    def save(self, folder):
        pickleObject(self, folder)
        np.save(self.__topicWordsFname(folder), self._topics)
        #np.save(self.__docTopicsFname(folder), self._docTopic)

    def load(self, folder):
        self._topics = np.load(self.__topicWordsFname(folder))
        #self._docTopic = np.load(self.__docTopicsFname(folder))

    def __topicWordsFname(self, f): return path.join(f, 'topicWordMatrix.npy')
    #def __docTopicsFname(self, f): return path.join(f, 'docTopicsMatrix.npy')

    def __createPlots(self, uncovs, iteration):
        fig, axes = plt.subplots(1)
        xplt = np.linspace(min(uncovs), max(uncovs), 1000, endpoint=True)
        yplt = self.uncov2prob(xplt)
        axes.hist(uncovs, bins=100, color='gray', normed=True)
        axes.plot(xplt, yplt, color='r')
        plt.savefig(path.join(self.tmpFolder, 'plot_%d.pdf' % iteration))

def indexOrReturn(o, i):
    '''
    :param o: list or other object
    :param i: integer index
    :return: o[i] if o is list, otherwise o
    '''
    if isinstance(o, list): return o[i]
    else: return o

def textTopics2Vector(model, tt):
    '''
    :param model: TopicModel
    :param tt: map topicId -> topicWeight
    :return: vector where indices correspond to topicIds, as ordered by model.topicIds
    '''
    T = model.numTopics()
    vec = np.empty(T, np.float64)
    for i, ti in enumerate(model.topicIds()): vec[i] = tt[ti]
    return vec

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

class TextData():
    '''Text data relevant for the algorithm.'''

    def __init__(self, topics, numWords, id = None):
        '''
        :param topics: array of document's topic proportions
        :param numWords: number of words in the document
        '''
        self.topics, self.numWords = topics, numWords
        self.id = id

from pytopia.corpus.Corpus import Corpus
from pytopia.corpus.Text import copyText
class ResampledCorpus(Corpus, IdComposer):
    '''
    Adapter for a pytopia Corpus, for resampling with repetition of the texts within the base corpus.
    Ids of new texts will be (oldId, positionInSample).
    '''

    def __init__(self, baseCorpus, sample, sampleId):
        '''
        :param sample: list of ids of text from the base corpus
        :param baseCorpus: Corpus or id
        :param sampleId: id of the sample, for correct forming of self.id
        '''
        self.baseCorpus = resolveIds(baseCorpus)
        self._base = resolve(baseCorpus)
        self.sampleId = sampleId
        self._sample = sample
        IdComposer.__init__(self)
        self._initData()

    def _initData(self):
        self._textCache = {}
        self._uniqBaseIds = set(id for id in self._sample)
        self._size = len(self._sample)

    def _getText(self, id):
        '''
        Cached fetching of texts from base corpus, with id rewriting.
        Return None if baseId is not in base or if id is out of range.
        :param id: basecorpus-level text id
        '''
        if not (id >= 0 and id < self._size): return None
        if id not in self._textCache:
            baseId = self._sample[id]
            txto = self._base.getText(baseId)
            if txto is not None:
                txto = copyText(txto)
                txto.id, txto.baseId = id, baseId
                self._textCache[id] = txto
            else: self._textCache[id] = None
        return self._textCache[id]

    def getText(self, id): return self._getText(id)

    def getTexts(self, ids):
        return [self._getText(id) for id in ids if self._getText(id) is not None]

    def __iter__(self):
        for id in self.textIds(): yield self._getText(id)

    def textIds(self): return range(self._size)

    def __len__(self): return self._size

def testResampledCorpus(corpus, sampleSize=100):
    from pytopia.testing.validity_checks import checkCorpus
    from topic_coverage.resources import pytopia_context
    corpus = resolve(corpus)
    txtIds = corpus.textIds(); N = len(txtIds)
    sample = [ txtIds[i] for i in np.random.choice(range(N), sampleSize, replace=True) ]
    #sample = [188015L]*20
    print sample
    scorpus = ResampledCorpus(corpus, sample, 'test_resampled')
    checkCorpus(scorpus)

def testSubsetter():
    from topic_coverage.resources import pytopia_context
    from pytopia.adapt.gensim.lda.builder import GensimLdaModelBuilder
    ss = Subsetter('us_politics', 'us_politics_dict', 'RsssuckerTxt2Tokens',
                    builder=GensimLdaModelBuilder, builderParams=[], buildId='testbuild', numIter=2,
                    tmpFolder='')
    print ss.id

def testSubsetterBuildLda():
    from topic_coverage.resources import pytopia_context
    from topic_coverage.modelbuild.modelbuild_iter1 import modelsContext
    from pytopia.adapt.gensim.lda.builder import GensimLdaModelBuilder, GensimLdaOptions
    bulildParams = [
    {
        'options': GensimLdaOptions(numTopics=50, alpha=1.0, eta=0.01, offset=1.0,
                         decay=0.5, chunksize=1000, passes=5, seed=3245+i)
    } for i in range(5)
    ]
    ss = Subsetter('us_politics', 'us_politics_dict', 'RsssuckerTxt2Tokens',
                    builder=GensimLdaModelBuilder(), builderParams=bulildParams,
                   buildId='testbuild', numIter=2, tmpFolder='/datafast/topic_coverage/subsetter_tmp/')
    with modelsContext():
        ss.build()
    print ss.id
    return ss
    # print type(ss), ss.__class__.__name__
    # ss.save('/datafast/topic_coverage/subsetter_tmp/test_save')
    # ssl = loadResource('/datafast/topic_coverage/subsetter_tmp/test_save')
    # print ssl.id

def testSubsetterBuildNmf(T=50):
    from topic_coverage.resources import pytopia_context
    from pytopia.adapt.scikit_learn.nmf.adapter import SklearnNmfBuilder
    from topic_coverage.modelbuild.modelbuild_iter1 import modelsContext
    from topic_coverage.modelbuild.modelbuild_iter1 import nmfSklearnUsPoliticsParams
    bparams = nmfSklearnUsPoliticsParams(T=T)
    ss = Subsetter('us_politics', 'us_politics_dict', 'RsssuckerTxt2Tokens',
                   builder=SklearnNmfBuilder(), builderParams=bparams, paramsId='nmfTestbuildParams',
                   buildId='testbuildNmf', numIter=2, tmpFolder='/datafast/topic_coverage/subsetter_tmp/')
    with modelsContext():
        ss.build()
    print ss.id
    return ss

def coverageExperiment():
    from topic_coverage.experiments.clustered_models_v0 import \
        coverageScoringExperiment, coverMetrics
    from topic_coverage.modelbuild.modelbuild_iter2 import *

    from pytopia.context.ContextResolver import resolve
    from pytopia.measure.avg_nearest_dist import AverageNearestDistance, TopicCoverDist
    from pytopia.measure.topic_distance import cosine as cosineDist

    coverMetrics = [
        AverageNearestDistance(cosineDist, pairwise=False),
        TopicCoverDist(cosineDist, 0.4)
    ]

    target = resolve('gtar_themes_model')
    models = [testSubsetterBuildNmf(100)]
    #models = [testSubsetterBuildLda()]
    coverageScoringExperiment(target, [models], coverMetrics)

if __name__ == '__main__':
    #testResampledCorpus('us_politics')
    #testSubsetterBuildLda()
    #testSubsetterBuildNmf()
    coverageExperiment()