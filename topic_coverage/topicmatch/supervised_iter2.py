from os.path import join

import numpy as np

from pytopia.measure.topic_distance import *
from topic_coverage.resources.pytopia_context import topicCoverageContext
from topic_coverage.topicmatch.labeling_iter1_uspolfinal import uspolLabelingModelsContex as uspolLabModels
from topic_coverage.topicmatch.supervised_data import dataset, getLabelingContext
from topic_coverage.topicmatch.supervised_models import *

from pyutils.stat_utils.utils import Stats
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer

def runNestedCV(model, grid, score, dataset, evalMetrics=None,
                  folds=5, seed=None, verbose=False, n_jobs=3):
    innerFolds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    outerFolds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scorer = make_scorer(score)
    prscorer, recscorer = make_scorer(precision_score), make_scorer(recall_score)
    cvFitter = GridSearchCV(estimator=model, param_grid=grid, cv=innerFolds, scoring=scorer,
                            verbose=verbose, n_jobs=1)
    # prscore = cross_val_score(cvFitter, X=features, y=labels, cv=outerFolds, scoring=prscorer,
    #                         verbose=verbose, n_jobs=n_jobs)
    # recscore = cross_val_score(cvFitter, X=features, y=labels, cv=outerFolds, scoring=recscorer,
    #                         verbose=verbose, n_jobs=n_jobs)
    # print 'precision', prscore
    # print 'recall', recscore
    features, labels = dataset
    score = cross_val_score(cvFitter, X=features, y=labels, cv=outerFolds, scoring=scorer,
                            verbose=verbose, n_jobs=n_jobs)
    return score

def nestedCvForModelsetAndFeatureset(dataset, modelgrids, featsets, optScore,
                                     evalMetrics=None, rseed=1234567, numThreads=3,
                                     folds=5):
    '''
    Run nested CV
    :param dataset: list of (Topic, Topic, label)
    :param modelgrids: list of callables, each returning a sklearn classifier and a grid
    :param featsets: list of strings describing feature sets
    :param optScore: sklearn classif. score function, used for optimal model params selection
    :return:
    '''
    print 'FEATURES: %s , SCORE: %s' % (featsets, optScore.__name__)
    for mg in modelgrids:
        model, _ = mg()
        modLabel = str(model.__class__.__name__)
        pipe, grid = createPipeline(mg, featsets, modelSeed=rseed)
        scores = []
        scores.extend(runNestedCV(pipe, grid, optScore, dataset, evalMetrics,
                                  seed=rseed, n_jobs=numThreads, folds=folds))
        #scores.extend(runNestedCV(pipe, grid, optScore, dataset, evalMetrics, seed=rseed*2, n_jobs=numThreads))
        ss = Stats(scores)
        scr = ','.join('%.3f' % s for s in scores)
        scrSummary = 'avg %.3f std %.3f vals: %s ' % (ss.mean, ss.std, scr)
        print '%28s %s' % (modLabel, scrSummary)

def runTestCv(corpus='uspol'):
    ctx = getLabelingContext(corpus)
    with ctx:
        dset = dataset(0.75, corpus=corpus, split=True)
        nestedCvForModelsetAndFeatureset(dset, [logistic, gbt], featsets=['allmetrics'],
                                         optScore=f1_score, rseed=9)

def runVectorfeatsCv(corpus='uspol'):
    ctx = getLabelingContext(corpus)
    with ctx:
        dset = dataset(0.75, corpus=corpus, split=True)
        nestedCvForModelsetAndFeatureset(dset, [randomForest], featsets=['allvectors'],
                                         optScore=f1_score, rseed=9, numThreads=2)

def runAllModels(corpus='uspol', feats='allmetrics', thresh=0.75,
                 modelgrids = [logistic, randomForest, mlp, svm], folds=5):
    ctx = getLabelingContext(corpus)
    models = ','.join(m.__name__ for m in modelgrids)
    print 'RUNNING corpus %s, feats %s, thresh %g ; MODELS: %s' % (corpus, feats, thresh, models)
    with ctx:
        dset = dataset(thresh, corpus=corpus, split=True)
        nestedCvForModelsetAndFeatureset(dset, modelgrids, featsets=[feats],
                                         optScore=f1_score, rseed=9, numThreads=3, folds=folds)

def warn(*args, **kwargs): pass

import warnings
warnings.warn = warn

if __name__ == '__main__':
    with topicCoverageContext():
        runAllModels('uspol', feats='core1')
        runAllModels('pheno', feats='core1')
