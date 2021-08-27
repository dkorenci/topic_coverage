from experiments.labeling.labelsets import *
from experiments.classification.datasets import *
from experiments.classification.transformers import CorpusTransformer
from agenda.mapping import labeled_files
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

import os, pickle #, joblib
from sklearn.externals import joblib

class SupervisedMapper():
    '''
    mapper to a set of labels that has a binary classifier per label
     and uses it to determine label assignment to text
    '''
    def __init__(self, labelClass):
        '''
        :param map label -> scikit_learn classifier returning 0 or 1:
        '''
        self.labelClass = labelClass

    def map(self, txto):
        return { l : self.predict(l, txto) for l in self.labelClass }

    def predict(self, label, txto):
        classifier = self.labelClass[label]
        pred = classifier.predict([txto])
        return pred[0]

    def description(self): return 'supervised_mapper'

# baseline_svc_devel_all
def getBaselineDevelMapper(id):
     labels = labels_rights_final()
     devel = labeled_files.devel_all_file
     mapperFile = object_store+('mappers/%s.pickle' %id)
     if os.path.exists(mapperFile):
         return joblib.load(mapperFile)
         #return pickle.load(open(mapperFile,'rb'))
     labelClass = {}
     c = -1
     for l in labels:
        print l
        devel_texts, devel_labels = getTextsAndLabelsFromFile(devel, l)
        pipe = Pipeline([('corpus_transform', CorpusTransformer('us_politics')),
                         ('svc', SVC(kernel='linear'))])
        pipe.fit(devel_texts, devel_labels)
        labelClass[l] = pipe
        c -= 1
        if c == 0: break
     mapper = SupervisedMapper(labelClass)
     joblib.dump(mapper, mapperFile)
     #pickle.dump(mapper, open(mapperFile,'wb'))
     return mapper

def getSupervisedMapper(labels, trainTexts, id = None, classifier='baseline'):
    '''
    construct text to labels mapper that maps with binary classifiers per label
    :param id: id used to save and load the mapper
    :param labels: set of labels for the mapper
    :param trainTexts: either a file (in a labeling folder) or a set of labeled texts
    :param classifier: classifier on which the mapper will be based
    :return:
    '''
    if id is not None:
        # try to load mapper
        mapperFile = object_store+('mappers/%s.pickle' %id)
        print mapperFile
        if os.path.exists(mapperFile):
            return joblib.load(mapperFile)
        # init folder for storing classifiers
        classiferFolder = object_store+('mappers/%s_classifiers/' %id)
        if not os.path.exists(classiferFolder): os.mkdir(classiferFolder)
    # load texts from file if neccessary
    if isinstance(trainTexts, (str, unicode)):
        trainTexts = getLabeledTextObjectsFromParse(parseLabeledTexts(labeling_folder+trainTexts))
    labelClass = {}
    c = -1
    for l in labels:
        print '************* LABEL: %s **************' % l
        labelClass[l] = None
        # check if classifier is already build and saved
        trainClassifier = True
        if id is not None:
            classifierFile = classiferFolder+l+'.pickle'
            if os.path.exists(classifierFile):
                print 'existing classifier %s' % classifierFile
                clf = joblib.load(classifierFile)
                print clf.best_estimator_.get_params()
                trainClassifier = False
        # train classifier and save
        if trainClassifier :
            devel_texts, devel_labels = getTextsAndLabels(trainTexts, l)
            clf = trainClassifierForMapper(devel_texts, devel_labels, classifier)
            if id is not None:
                print 'storing classifier %s' % classifierFile
                joblib.dump(clf, classifierFile)
            else: labelClass[l] = clf
        c -= 1
        if c == 0: break
    # load saved classifiers
    if id is not None:
        for l in labels:
            classifierFile = classiferFolder+l+'.pickle'
            if os.path.exists(classifierFile):
                print 'loading classifier %s' % classifierFile
                labelClass[l] = joblib.load(classifierFile)
    # construct and save mapper
    mapper = SupervisedMapper(labelClass)
    if id is not None: joblib.dump(mapper, mapperFile)
    return mapper
# map {size -> [eval1, eval2, ...]}

def trainClassifierForMapper(texts, labels, classifier):
    'given set of train Texts and binary labels, train a classifer according to specified strategy'
    if classifier == 'baseline': return trainBaselineClassifierForMapper(texts, labels)
    elif classifier == 'optWeights': return trainOptWeightsClassifier(texts, labels)
    elif classifier == 'best': return trainBestClassifier(texts, labels)

def trainBaselineClassifierForMapper(texts, labels):
    pipe = Pipeline([('corpus_transform', CorpusTransformer('us_politics', normalize='unit-vector')),
                 ('svc', SVC(kernel='linear'))])
    pipe.fit(texts, labels)
    return pipe

def trainBestClassifier(texts, labels):
    pipe = Pipeline([('corpus_transform', CorpusTransformer('us_politics', normalize=None)),
                 ('svc', SVC(kernel='linear'))])
    pipe.fit(texts, labels)
    return pipe

def trainOptWeightsClassifier(texts, labels, labelClass = 1):
    pipe = Pipeline([('transformer', CorpusTransformer('us_politics')), ('svc', SVC(kernel='linear'))])
    # calc and class counts, determine number of folds
    classCounts = {}
    for c in labels:
        if c not in classCounts: classCounts[c] = 1
        else: classCounts[c] += 1
    cc = classCounts[labelClass]
    print 'label class count %d' % cc
    if cc >= 5 : num_folds = 5
    elif cc >= 3 : num_folds = cc
    else: return trainBaselineClassifierForMapper(texts, labels)
    print 'num_folds %d' % num_folds
    # start grid search
    mlparams = [ { 'svc__class_weight': ['auto', {0:1,1:1}, {0:1,1:10}, {0:1,1:100},  {0:1,1:1000} ] ,
                   'transformer__normalize' : [ None, 'unit-vector' ] } ]
    # test params
    # mlparams = [ { 'svc__class_weight': ['auto',  {0:1,1:1000} ] } ]; num_folds = 3
    score = 'f1'
    cv = StratifiedKFold(labels, num_folds)
    clf = GridSearchCV(pipe, param_grid=mlparams, cv=cv, verbose=100, scoring = score, n_jobs=2)
    clf.fit(texts, labels)
    print '************* BEST **************'
    print clf.best_estimator_.get_params()
    return clf

