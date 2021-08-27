from topic_coverage.modelbuild.modelbuild_iter1 import modelsContext
from topic_coverage.topicmatch.data_iter0 import loadDataset

from pytopia.measure.topic_distance import *
from pytopia.context.ContextResolver import resolve
from sklearn.linear_model import LogisticRegression
from pyutils.stat_utils.utils import Stats
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer

import numpy as np

def logistic():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(), {'C':[0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
                                  'penalty':['l1', 'l2']
                                  }

def randomForest():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(), {'n_estimators':[5, 10, 20, 50],
                                      'max_features':[1, 'sqrt', None]}

def svm():
    from sklearn.svm import SVC
    return SVC(), { 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                    'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

def mlp():
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(), \
           {'hidden_layer_sizes':[(5, ), (10, )],
            'activation': ['logistic'],
            'learning_rate_init' :[0.1, 0.01, 0.001, 0.0001],
            'max_iter': [500],
            'alpha': [0.0001, 0.01],
            'solver': ['lbfgs']}

def knn():
    from sklearn.neighbors import KNeighborsClassifier as KNN
    return KNN(), { 'n_neighbors':[5, 10, 15],
                    'weights': ['uniform', 'distance'],
                    #'metric': ['minkowski', 'cosine'] }
                   }

def gbt():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(), \
           {
              'loss': ['deviance', 'exponential'],
              'learning_rate': [0.1, 0.5, 0.8],
              'n_estimators': [30, 50, 100],
              'max_depth':[2,3,4],
              'min_samples_leaf': [2]
           }

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
def nestedCVScore(features, labels, model, grid, score=accuracy_score,
                  folds=5, seed=1234567, verbose=False, n_jobs=2):
    '''
    Perform nested cross-validation with specified number of folds
      both in outer and inner loop.
    :param features: matrix of features
    :param labels: vector of labels
    :param model: classifier
    :param grid: parameter grid, map: parameter_name -> values
    :param score: score function
    '''
    innerFolds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    outerFolds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    model.random_state = seed
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
    score = cross_val_score(cvFitter, X=features, y=labels, cv=outerFolds, scoring=scorer,
                            verbose=verbose, n_jobs=n_jobs)
    return score

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

def topicDocDistances(t1, t2, metrics):
    dvec1, dvec2 = corpusTopicWeights(t1), corpusTopicWeights(t2)
    return [m(dvec1, dvec2) for m in metrics]

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

dsetV1 = None
def extractFeaturesAndLabels(dataset, features='distances', metrics='all'):
    '''
    Create sklearn-compatible dataset from the dataset of topic pairs.
    :param dataset: iterable of (Topic, Topic, label)
    :param features: which topic-related vectors are used, and weather raw vectors or distances are used
    :param metrics: if vector distances are used, this param determines the set of distance metrics
    :return: feature matrix, label vector
    '''
    global dsetV1
    if dsetV1: return dsetV1
    metrics = metricSet(metrics)
    labels = np.array([int(label) for _, _, label in dataset], np.int)
    features = np.array([featExtract(t1, t2, features, metrics)
                         for t1, t2, _ in dataset], np.float64)
    dsetV1 = features, labels
    return features, labels

def runNestedCV(modelGrid, score, dataset, features='distances', metrics='all', verbose=False):
    model, grid = modelGrid
    feats, labels = extractFeaturesAndLabels(dataset, features, metrics)
    return nestedCVScore(feats, labels, model, grid, score=score, verbose=verbose)

def cvFitModel(modelgrid, features, labels, score=f1_score,
               folds=5, seed=1234567, verbose=False, n_jobs=3):
    '''
    Perform nested cross-validation with specified number of folds
      both in outer and inner loop.
    :param features: matrix of features
    :param labels: vector of labels
    :param model: classifier
    :param grid: parameter grid, map: parameter_name -> values
    :param score: score function
    '''
    folds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    model, grid = modelgrid
    model.random_state = seed
    scorer = make_scorer(score)
    cvFitter = GridSearchCV(estimator=model, param_grid=grid, cv=folds,
                            scoring=scorer, verbose=verbose, n_jobs=n_jobs)
    cvFitter.fit(features, labels)
    return cvFitter#.estimator

def learningCurve(model, features, labels, testSize=0.33, steps=10,
                  scoreFunc=f1_score, average=1, rndseed=8832, title=''):
    from sklearn.model_selection import train_test_split
    scores = np.empty((average, steps))
    trainSizes = []
    for a in range(average):
        X_train, X_test, y_train, y_test = train_test_split(
                features, labels, stratify=labels, test_size=testSize, random_state=rndseed+a)
        trainSize = len(X_train)
        for i in range(1, steps+1):
            if i < steps: sz = int((trainSize/float(steps))*i)
            else: sz = trainSize
            if a == 0: trainSizes.append(sz)
            X = X_train[:sz]
            y = y_train[:sz]
            model.fit(X, y)
            score = scoreFunc(y_test, model.predict(X_test))
            scores[a, i-1] = score
    # plot learning curve
    print np.mean(scores, axis=0)
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    if title: ax.set_title(title)
    ax.set_xlabel('train set size')
    ax.set_ylabel('%s'%scoreFunc.__name__)
    ax.set_ylim(0, 1.0)
    from scipy.interpolate import interp1d
    # plot all scores per train size
    for i in range(steps):
        for y in scores[:, i]: ax.plot(trainSizes[i], y, 'x', color='black')
    # plot average scores and interpolation curve
    x = trainSizes; y = np.mean(scores, axis=0)
    xi = np.linspace(x[0], x[-1], num=100, endpoint=True)
    linInterp = interp1d(x, y)
    ax.plot(x, y, 'o', xi, linInterp(xi), '-')
    ax.yaxis.grid(True)
    plt.show()

def cvFitAndLearnCurve(modelgrid, dataset, featSet='distances', metrics='all',
                        testSize=0.33, steps=10, score=f1_score, average=10):
    '''
    Perform grid-search with CV to get optimal params, then plot learning curves of the optimal model.
    '''
    features, labels = extractFeaturesAndLabels(dataset, featSet, metrics)
    print score
    optmodel = cvFitModel(modelgrid, features, labels, score=score)
    title = 'model: %s , features: %s , metrics: %s' % \
            (optmodel.__class__.__name__, featSet, metrics)
    learningCurve(optmodel, features, labels, testSize, steps,
                  scoreFunc=score, average=average, title=title)

def optMatchModel(modelgrid, dataset, featSet='distances', metrics='all', score=f1_score):
    ''' Select optimal topic match model for given
    features by performing X-valid evaluation of grid params. '''
    features, labels = extractFeaturesAndLabels(dataset, featSet, metrics)
    return cvFitModel(modelgrid, features, labels, score=score)

def cvAllClassifiers(dataset, features='distances', metrics='all'):
    modgrids = [logistic(), randomForest(), gbt(), knn(), mlp(), svm()]
    #modgrids = [knn(), svm()]
    #modgrids = [mlp()]
    print '******* FEATURES: %s , METRICS: %s' % (features, metrics)
    scores = [accuracy_score, f1_score]
    for mg in modgrids:
        model, _ = mg
        print '--- model: %s' % str(model.__class__.__name__)
        for scoreFunc in scores:
            score = runNestedCV(mg, scoreFunc, dataset, features, metrics)
            ss = Stats(score)
            scr = ','.join('%.3f' % s for s in score)
            print '%15s average %.3f std %.3f vals: %s ' % \
                  (scoreFunc.__name__, ss.mean, ss.std, scr)

def createLearnCurve(dataset):
    model = LogisticRegression(C=100)
    feats, labels = extractFeaturesAndLabels(dataset, 'distances')
    learningCurve(model, feats, labels, 240, 10)

def analyzeKl():
    data = loadDataset()
    def topicData(t):
        v = t.vector
        print 'min %g, max %g, zeros %d' % (min(v), max(v), sum(v==False))
        print ','.join('%g'%val for val in v)
        print resolve(t.model).id
    for t1, t2, _ in data:
        print 'KL-divergence: %g' % kullbackLeibler(t1.vector, t2.vector, 1e-8)
        topicData(t1)
        print
        topicData(t2)
        print '\n'

def uspolBinLab():
    '''
    Dataset of us politics topic pairs labeled with binary labeling scheme.
    '''
    dataFolder='/home/damir/Dropbox/projekti/doktorat/D1 eksplorativa/mjerenje pokrivenosti/supervised/oznaceni parovi/uspol_binary_prelim/labeled/'
    return loadDataset(dataFolder)

if __name__ == '__main__':
    with modelsContext():
        #testDsetCreation()
        #runNestedCV(logistic(), f1_score, verbose=True)
        #cvAllClassifiers('all-distances', 'cosine')
        #analyzeKl()
        #selectedModel1()
        cvFitAndLearnCurve(logistic(), uspolBinLab(), 'all-distances', 'cosine', 100, 10, average=10, score=f1_score)
        #cvFitAndLearnCurve(mlp(), uspolBinLab(),'all-distances', 'cosine', 100, 10, average=10, score=f1_score)
        # cvFitAndLearnCurve(randomForest(), 'all-distances', 'all',
        #                    100, 10, average=10, score=accuracy_score)