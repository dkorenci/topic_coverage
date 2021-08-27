from doc_topic_coh.evaluations.tools import labelsMatch
from pytopia.topic_functions.coherence.doc_matrix_coh_factory import \
    avg_dist_coherence as avgcoh, variance_coherence as varcoh
from doc_topic_coh.dataset.topic_splits import iter0DevTestSplit, topicLabelStats
from doc_topic_coh.dataset.topic_labels import labelAllTopics, labelingStandard

import numpy as np

# theme, theme_noise, theme_mix, theme_mix_noise, noise

def createDataset(ltopics, label, scorers):
    '''
    Create sklearn compatible dataset from a list of labeled topics and functions on topics.
    Features for each topics are values of the functions, and labels
    are binarized topic labels.
    :param ltopics: list of (topic, labeling)
    :param label: string or a list of strings designating a positive label
    :param scorers: list of functions accepting a topic and returning a number (score)
    '''
    labels = np.array(labelsMatch(ltopics, label), dtype=np.int32)
    features = np.array(
        [ [ s(t) for s in scorers ] for t, _ in ltopics ],
        dtype=np.float64
    )
    return features, labels

def logistic():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(), {'C':[0.01, 0.1, 1.0, 10, 100]}

def randomForest():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(), {'n_estimators':[5, 10, 20, 50],
                                      'max_features':[1, 'sqrt', None]}

def svm():
    from sklearn.svm import SVC
    return SVC(), { 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                    'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

def knn():
    from sklearn.neighbors import KNeighborsClassifier as KNN
    return KNN(), { 'n_neighbors':[5, 10, 15],
                    'weights': ['uniform', 'distance'],
                    #'metric': ['minkowski', 'cosine'] }
                   }

from sklearn.metrics import f1_score, accuracy_score
def nestedCVScore(features, labels, model, grid, score=accuracy_score,
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
    from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
    from sklearn.metrics import make_scorer
    innerFolds = KFold(n_splits=folds, shuffle=True, random_state=seed)
    outerFolds = KFold(n_splits=folds, shuffle=True, random_state=seed)
    model.random_state = seed
    scorer = make_scorer(score)
    cvFitter = GridSearchCV(estimator=model, param_grid=grid, cv=innerFolds, scoring=scorer,
                            verbose=verbose, n_jobs=1)
    score = cross_val_score(cvFitter, X=features, y=labels, cv=outerFolds, scoring=scorer,
                            verbose=verbose, n_jobs=n_jobs)
    return score


def testDsetCreation():
    scorers = [
        varcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='l2'),
        avgcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='l2'),
        varcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='cosine'),
        avgcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='cosine'),
    ]
    dev, test = iter0DevTestSplit()
    features, labels = createDataset(dev, ['theme', 'theme_noise'], scorers)
    print features
    print labels

def testNestedCV():
    scorers = [
        varcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='l2', center='median'),
        #avgcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='l2'),
        #varcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='cosine'),
        #avgcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='cosine'),
    ]
    dev, test = iter0DevTestSplit()
    features, labels = createDataset(test, ['theme', 'theme_noise'], scorers)
    print 'dataset created'
    model, grid = logistic()
    score = nestedCVScore(features, labels, model, grid, verbose=True)
    print score
    print np.average(score)

def runNestedCV(scorers, ltopics, posClass, modelGrid, verbose=False):
    features, labels = createDataset(ltopics, posClass, scorers)
    N, numPos, numNeg = float(len(labels)), sum(labels==1.0), sum(labels==0.0)
    #print 'positive prop. %.4f' % (numPos/N)
    model, grid = modelGrid
    print 'model: ', str(model.__class__.__name__)
    score = nestedCVScore(features, labels, model, grid, verbose=verbose)
    scr = '[%s]'%(','.join('%.3f'%s for s in score))
    avg = np.average(score)
    print 'average %.3f'%avg, scr

if __name__ == '__main__':
    #testDsetCreation()
    testNestedCV()