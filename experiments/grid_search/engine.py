from multiprocessing import Manager, Pool
import os
import pickle
import random
from sets import Set
import sys
from time import clock
from traceback import print_exception

from resources.resource_builder import *
from experiments.grid_search.options import gridSearch2Options
from experiments.rsssucker_lda import loadBowStream_numpy, loadRssSuckerDictionary
from models.adapters import GensimLdamodel
from gensim_mod.models.ldamodel import LdaModel as LdaModel_mod
from models.interfaces import TopicModel
from models.label import modelLabel
from resources.resource_builder import loadDictionary, loadBowCorpus
from pymedialab_settings import settings
from utils.utils import normalize_path, unpinProcess


class GridPoint():
    def __init__(self, opt):
        self.options = opt
        self.eval = {}


class Grid():
    def __init__(self):
        self.points = []

    def removeVectors(self):
        'remove large ndarrays from options'
        for p in self.points:
            # todo check explicitly for ndarray
            if not isinstance(p.options.eta, (int, long, float)):
                p.options.eta = []

    def addPoint(self, pt):
        self.points.append(pt)

    def containsOption(self, opt):
        for p in self.points:
            if p.options == opt : return True
        return False


def gridSearch():
    # setup files for saving results
    gs_folder = settings.object_store+'grid_search/'
    gs_file = gs_folder + 'grid.pickle'
    if not os.path.exists(gs_folder): os.makedirs(gs_folder)
    # load dictionary and bow documents
    database = 'rsssucker_topus1_27022015'
    bows = loadBowStream_numpy(database)
    dictionary = loadRssSuckerDictionary(database)
    # sample train and test sets
    numDocs = len(bows);
    testSize = 1000; trainSize = 1000; seed = 12356
    train, test = sampleSplitIndexes(seed, numDocs, trainSize, testSize)
    testDocs = bows[test]; trainDocs = bows[train]
    print 'num train: ' + str(len(trainDocs))
    # create options and evaluation functions
    options = gridSearch2Options()
    evalFunctions = [ PerplexityEval(testDocs), CoherenceEval(trainDocs) ]
    # start grid search
    grid = pickle.load(file(gs_file,'rb')) if os.path.exists(gs_file) else None
    for opt in options:
        if grid is not None:
            if grid.containsOption(opt): continue
        #todo logging, timing
        print 'evaluation for: ' + modelLabel(opt)
        gensim_model = opt.getModel(dictionary);
        tmark = clock()
        gensim_model.update(trainDocs)
        print 'training_time: ' + str(clock() - tmark)
        model = GensimLdamodel(gensim_model)
        gpoint = GridPoint(opt)
        for func in evalFunctions:
            val = func(model)
            gpoint.eval[str(func)] = val
            print str(func) + ' : ' + '%.3f' % val

        if grid is None : grid = Grid()
        grid.addPoint(gpoint)
        pickle.dump(grid, file(gs_file,'wb'))

    # folder
    # traverse options, check for model folder, generate
    return 0

class OptionsEvaluator():
    'evaluate one set of model options, for parallel grid search'
    def __init__(self, trainSet, funcs, dict, gs_file, lock, save_folder = None):
        self.trainSet = trainSet
        self.funcs = funcs
        self.dict = dict
        self.lock = lock
        self.gs_file = gs_file
        self.save_folder = save_folder

    def __call__(self, opt):
        try:
            self.doWork(opt)
        except:
            e = sys.exc_info()
            print 'processing of options %s failed' % modelLabel(opt)
            print_exception(e[0],e[1],e[2])
            return None


    def doWork(self, opt):
        print 'evaluation for: ' + modelLabel(opt)
        # read already processed grid points
        self.lock.acquire()
        grid = pickle.load(file(self.gs_file,'rb')) \
                if os.path.exists(self.gs_file) else None
        if grid is not None:
            # removing (large) numpy array etas from file
            # todo: solve better
            grid.removeVectors()
            pickle.dump(grid, file(self.gs_file,'wb'))
        self.lock.release()

        if grid is not None:
            if grid.containsOption(opt):
                print 'already processed: '+modelLabel(opt)
                return None
        unpinProcess()
        gpoint = GridPoint(opt)
        if opt.eval_passes is not None:
            gpoint.passesEval = []
            opt.eval_results = gpoint.passesEval
        gensim_model = opt.getModel(self.dict);
        t = clock()
        gensim_model.update(self.trainSet)
        t = clock() - t
        model = GensimLdamodel(gensim_model)
        gpoint.eval['time'] = t
        for f in self.funcs:
            val = f(model)
            gpoint.eval[str(f)] = val
            print str(f) + ' : ' + '%.3f' % val
        print 'time %.3f' % t
        # save model
        if self.save_folder is not None:
            model.save(self.save_folder+modelLabel(opt))
        # save processed point to grid
        self.lock.acquire()
        grid = pickle.load(file(self.gs_file,'rb')) \
                if os.path.exists(self.gs_file) else None
        if grid is None : grid = Grid()
        grid.addPoint(gpoint)
        pickle.dump(grid, file(self.gs_file,'wb'))
        self.lock.release()
        return gpoint


def gridSearchParallel(folder, corpusId, options, processes,
                       testSize, seed=12345, propagateSeed = True,
                       trainSize = None, shuffleOpts = True, label = '', evalPasses = False):
    '''
    :param folder: place to store model evaluations and models
    :param corpusId: corpus to do grid search on
    :param options: base of grid search - a list of model options
    :param processes: number of processes to train and eval models in parallel
    :param testSize: size of test set, for perplexity evaluation
    :param trainSize: size of train set, if None, train set will be all except test
    :param seed: random seed for taking test and train set
    :param propagateSeed: if true, set seed as seed for model training
    :param shuffleOpts: do random shuffle on options before assigning to processes, for load balancing
    :param label: label for grid search results file
    :return:
    '''
    # setup files for saving results
    gs_folder = settings.object_store + normalize_path(folder)
    gs_file = gs_folder + 'grid%s.pickle'%label
    model_folder = gs_folder+'models/'
    if not os.path.exists(gs_folder): os.makedirs(gs_folder)
    if not os.path.exists(model_folder): os.makedirs(model_folder)
    print gs_file
    # load dictionary and bow documents
    dictionary = loadDictionary(corpusId)
    bows = loadBowCorpus(corpusId)
    # sample train and test sets
    numDocs = len(bows); print 'number of documents: ' + str(numDocs)
    #testSize = 5000; trainSize = 100000; seed = 123567
    train, test = sampleSplitIndexes(seed, numDocs, trainSize, testSize)
    testDocs = bows[test]; trainDocs = bows[train]
    print 'num train %d , num test %d ' % (len(trainDocs), len(testDocs))
    # create options and evaluation functions
    evalFunctions = [ PerplexityEval(testDocs), CoherenceEval(trainDocs) ]
    # start grid search
    lock = Manager().Lock()
    evaluator = OptionsEvaluator(trainDocs, evalFunctions, dictionary, gs_file, lock, model_folder)
    p = Pool(processes)
    if evalPasses:
        for o in options : o.eval_passes = evalFunctions
    if propagateSeed:
        for o in options : o.seed = seed
    if shuffleOpts: random.seed(seed); random.shuffle(options)
    result = p.map(evaluator, options)


class PerplexityEval():
    def __init__(self, documents):
        self.docs = documents

    def __str__(self): return 'perplexity'

    def __call__(self, model):
        if not isinstance(model, (TopicModel, LdaModel_mod)): raise TypeError
        if isinstance(model, TopicModel): return model.perplexity(self.docs)
        else: return GensimLdamodel(model).perplexity(self.docs)

class CoherenceEval():
    def __init__(self, docs, M = 20):
        self.docs = docs
        self.M = M
        self.buildWordToDocMap()

    def __str__(self): return 'coherence'

    def __call__(self, model):
        if not isinstance(model, (TopicModel, LdaModel_mod)): raise TypeError
        if isinstance(model, LdaModel_mod):
            model = GensimLdamodel(model)
        coh = 0.0; T = len(model.topic_indices())
        for i in model.topic_indices() :
            topind = model.top_word_indices(i, self.M)
            c = 0; pairs = 0
            for j in range(len(topind))[1:] :
                for k in range(len(topind))[:j] :
                    wj = topind[j]; wk = topind[k]
                    if wk in self.w2doc and wj in self.w2doc:
                        denom = float(len(self.w2doc[wk]))
                        nom = len(self.w2doc[wj].intersection(self.w2doc[wk]))
                        c += np.log((nom+1)/denom)
                        pairs += 1
            c /= pairs # average over (j,k) word pairs
            coh += c
        coh /= T # average over topics
        return coh



    def buildWordToDocMap(self):
        'build word index -> containing document set  map for bow corpus'
        self.w2doc = {}
        for di, doc in enumerate(self.docs):
            for wi, _ in doc:
                if not wi in self.w2doc: self.w2doc[wi] = Set()
                self.w2doc[wi].add(di)


def sampleSplitIndexes(seed, N, train, test):
    random.seed(seed)
    ind = [i for i in range(N)]
    random.shuffle(ind)
    testSet = ind[:test]; trainSet = ind[test:]
    if train is not None : trainSet = trainSet[:train]
    return trainSet, testSet