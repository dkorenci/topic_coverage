'''
Definition, building, saving and loading of supervised topic matching models.
'''
from os import path

from sklearn.externals.joblib import dump, load
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

from topic_coverage.topicmatch.feature_extraction import CachedSklearnTPFE

from pytopia.utils.logging_utils.setup import *
log = createLogger(__file__, INFO)

class scale:
    '''
    Adapts a function that returning sklearn model and grid
    by adding feature scaling transformer and modifying grid
    parameters for the new pipeline.
    '''
    def __init__(self, func, scaler='robust'):
        self.func = func
        self.scaler = scaler
        self.__name__ = func.__name__+'scaled'

    def __call__(self, *args):
        mod, grid = self.func()
        mname, scname = self.func.__name__, 'scaler'
        scaler = StandardScaler() if self.scaler == 'standard' else RobustScaler()
        pipe = [(scname, scaler),
                (mname, mod)]
        pipe = Pipeline(pipe)
        grid = {'%s__%s' % (mname, k): v for k, v in grid.iteritems()}
        return pipe, grid

def logistic():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(), {'C':[0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
                                  'penalty':['l1', 'l2']
                                  }
logisticScaled = scale(logistic)

def randomForest():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(), {'n_estimators': [10, 20, 50, 100],
                                      'max_features': [2, 0.5, None],
                                      'max_depth': [2, 3, None],
                                      'criterion': ['gini'],
                                    }

def svm():
    from sklearn.svm import SVC
    return SVC(), { 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'gamma': ['auto', 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

def mlp():
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(), \
           {'hidden_layer_sizes':[(3, ), (5, ), (10, )],
            'activation': ['logistic'],
            'max_iter': [500],
            'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'solver': ['lbfgs']}

mlpScaled = scale(mlp)

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

def featSelectKbest():
    from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
    return SelectKBest(mutual_info_classif, k=10)
    #return SelectKBest(chi2, k=10)

def createPipeline(modelgrid, features, modelSeed=None):
    '''
    Create skelarn pipeline with grid for grid search.
    Pipeline is created from classification model and list of features.
    Grid is created from model grid and list of features.
    :param modelgrid: callable returning a model and parameter grid
    :param features: list of strings describing feature sets
    :return:
    '''
    from sklearn.pipeline import Pipeline
    model, grid = modelgrid(); modelLabel = modelgrid.__name__
    model.random_state = modelSeed
    fextrLabel = 'tpfe'; fselectLabel = 'fselect'
    #print modelLabel, fextrLabel
    pipe = [(fextrLabel, CachedSklearnTPFE()),
            #(fselectLabel, featSelectKbest()),
            (modelLabel, model)]
    pipe = Pipeline(pipe)
    grid = {'%s__%s'%(modelLabel, k):v for k, v in grid.iteritems()}
    grid['%s__features'%fextrLabel]=[f for f in features]
    #print grid
    return pipe, grid


def cvFitModel(dataset, modelgrid, features, score=f1_score,
               folds=5, rseed=1234567, verbose=False, n_jobs=1):
    pipe, grid = createPipeline(modelgrid, [features], modelSeed=rseed)
    features, labels = dataset
    folds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rseed)
    cvFitter = GridSearchCV(estimator=pipe, param_grid=grid, cv=folds,
                            scoring=make_scorer(score), verbose=verbose, n_jobs=n_jobs)
    cvFitter.fit(features, labels)
    return cvFitter.best_estimator_

from topic_coverage.settings import supervised_models_folder

def buildLoadSaveModel(modelId, dataset, modelgrid, features, score=f1_score,
               folds=5, rseed=1234567, verbose=False, n_jobs=1):
    mfile = path.join(supervised_models_folder, modelId+'.joblib')
    if path.exists(mfile):
        log.info("buildLoadSaveModel loading existing model: %s" % mfile)
        return load(open(mfile, 'rb'))
    log.info("buildLoadSaveModel building model: %s" % modelId)
    model = cvFitModel(dataset, modelgrid, features, score, folds, rseed, verbose, n_jobs)
    if hasattr(model, 'id'): raise Exception('model already has id:\n%s'%model)
    else: model.id = modelId
    print model.id
    print model
    dump(model, open(mfile, 'wb'))
    return model

def switchOffMultiprocCache(model):
    fextr = model.named_steps['tpfe']
    assert isinstance(fextr, CachedSklearnTPFE)
    fextr.switchOffMultiproc()

if __name__ == '__main__':
    pass