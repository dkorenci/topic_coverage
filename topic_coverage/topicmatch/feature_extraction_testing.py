from os.path import join

import numpy as np

from topic_coverage.resources.pytopia_context import topicCoverageContext

from pytopia.measure.topic_distance import *
from topic_coverage.topicmatch.feature_extraction import CachedSklearnTPFE, TopicPairFeatureExtractor
from topic_coverage.topicmatch.labeling_iter1_uspolfinal import uspolLabelingModelsContex as uspolLabModels
from topic_coverage.topicmatch.supervised_data import dataset

def prelimTestFE():
    ctx = uspolLabModels()
    with ctx:
        tpfe = CachedSklearnTPFE()
        pairs, labels = dataset(split=True)
        print pairs
        print tpfe.transform(pairs)

def getTpfePipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    pipe = [('tpfe', CachedSklearnTPFE('allmetrics')),
            ('logreg', LogisticRegression())]
    pipe = Pipeline(pipe)
    grid = {'logreg__C':[0.01, 0.1, 1.0, 10, 100],
            'tpfe__features':['allmetrics']}
            #'tpfe__features': ['vectors']}
    return pipe, grid

def simpleClassificationTest():
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    pipe, grid = getTpfePipeline()
    ctx = uspolLabModels()
    with ctx:
        pairs, labels = dataset(split=True)
        X_train, X_test, y_train, y_test = \
            train_test_split(pairs, labels, test_size = 0.33, random_state = 42)
        pipe.fit(X_train, y_train)
        print f1_score(y_test, pipe.predict(X_test))

def cvClassificationTest():
    from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
    from sklearn.metrics import make_scorer
    from sklearn.metrics import f1_score
    pipe, grid = getTpfePipeline()
    ctx = uspolLabModels()
    model = pipe; score = f1_score; folds = 5; seed=1234567; verbose=False
    with ctx:
        pairs, labels = dataset(labelAgg=0.75, split=True)
        innerFolds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        outerFolds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        model.random_state = seed
        scorer = make_scorer(score)
        cvFitter = GridSearchCV(estimator=model, param_grid=grid, cv=innerFolds, scoring=scorer,
                                verbose=verbose, n_jobs=1)
        print cvFitter
        score = cross_val_score(cvFitter, X=pairs, y=labels, cv=outerFolds,
                                scoring=scorer, verbose=verbose, n_jobs=3)
        print score

def saveLoadTest():
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import make_scorer
    from sklearn.metrics import f1_score
    #from cPickle import dump, load
    from sklearn.externals.joblib import dump, load
    pipe, grid = getTpfePipeline()
    ctx = uspolLabModels()
    model = pipe; score = f1_score; folds = 5; seed=1234567; verbose=False
    savefolder = '/datafast/topic_coverage/tmp/sklearnLoadSaveTest/'
    fname = join(savefolder, 'model.pickle')
    with ctx:
        pairs, labels = dataset(labelAgg=0.75, split=True)
        innerFolds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        outerFolds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        model.random_state = seed
        scorer = make_scorer(score)
        cvFitter = GridSearchCV(estimator=model, param_grid=grid, cv=innerFolds, scoring=scorer,
                                verbose=verbose, n_jobs=3)
        cvFitter.fit(pairs, labels)
        print cvFitter
        bestmodel = cvFitter.best_estimator_
        predlabs = cvFitter.predict(pairs)
        print bestmodel
        print bestmodel.named_steps['tpfe']
        dump(bestmodel, open(fname, 'wb'))
        lbestmodel = load(open(fname, 'rb'))
        lpredlabs = lbestmodel.predict(pairs)
        print 'equality', np.array_equal(predlabs, lpredlabs)

if __name__ == '__main__':
    with topicCoverageContext():
        # TODO zasto je sporije sa novim querijem koji sadrzi i matcher-data?
        #prelimTestFE()
        #simpleClassificationTest()
        #cvClassificationTest()
        saveLoadTest()