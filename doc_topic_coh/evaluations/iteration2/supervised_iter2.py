from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer as SB
from doc_topic_coh.evaluations.iteration2.best import expFolder, cacheFolder, dev, test
from doc_topic_coh.evaluations.iteration2.best import paramsBestAndBaselines
from doc_topic_coh.evaluations.supervised import *

# instantiate scorer form params
IS = lambda p : SB(cache=cacheFolder, **p)

bestCorpusVectors = map(IS, paramsBestAndBaselines(['corpus_vectors']))

def sup(label, scorers, posClass = ['theme', 'theme_noise'], oneByOne=True, all=True):
    print label
    # one by one
    if oneByOne:
        for sc in scorers:
            print sc.id
            runNestedCV([sc()], test, posClass, logistic())
            runNestedCV([sc()], test, posClass, svm())
    # all together
    if all:
        print 'ALL SCORERS'
        scorers = [sc() for sc in scorers]
        runNestedCV(scorers, test, posClass, logistic())
        runNestedCV(scorers, test, posClass, svm())
        runNestedCV(scorers, test, posClass, randomForest())
        runNestedCV(scorers, test, posClass, knn())

if __name__ == '__main__':
    #supBestAlgo()
    #supBaseline()
    #sup('baseline_new', blineNew)
    #sup('baseline_palmetto', blinePalmetto)
    #sup('baseline_all', blineAll)
    #sup('best_algo_v2w', bestAlgoW2V)
    sup('best_corpus_vectors', bestCorpusVectors, oneByOne=False)
    #sup('all', bestAlgo + bestAlgoW2V + blineAll, oneByOne=False)