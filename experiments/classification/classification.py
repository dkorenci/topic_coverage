from experiments.labeling.labelsets import *
from resources.resource_builder import *
from corpus.factory import CorpusFactory
from experiments.classification.datasets import *
from experiments.classification.tresh_linearsvc import *

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

def createTextFeatures():
    devel = 'labeled/devel_200-1000_ristov_corrected.txt'
    develFeats = constructTextFeatures(devel, 'us_politics')
    saveDataset('devel_features', develFeats)
    test = 'labeled/test_ristov1000-1200_damir2000-2800_cleaned.txt'
    testFeats = constructTextFeatures(test, 'us_politics')
    saveDataset('test_features', testFeats)

def createTextLabels():
    devel = 'labeled/devel_200-1000_ristov_corrected.txt'
    test = 'labeled/test_ristov1000-1200_damir2000-2800_cleaned.txt'
    labeledTexts = {'devel':devel, 'test':test}
    labels = labels_rights_final()
    for id in labeledTexts:
        file = labeledTexts[id]
        for lab in labels:
            cl = constructTextClasses(file, [lab])
            name = id+'_'+lab
            saveDataset(name, cl)
        cl = constructTextClasses(file, labels, multiLabel=True)
        name = id+'_multilabel'
        saveDataset(name, cl)

def trainMulticlassClassifier():
    trainfeat = loadDataset('devel_features')
    traincl = loadDataset('devel_multilabel')
    testfeat = loadDataset('test_features')
    testcl = loadDataset('test_multilabel')
    # Set the parameters by cross-validation
    #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
    #                    {'kernel': ['linear'], 'C': [1]}] #  10, 100, 1000
    mlparams = [ {'estimator' : [SVC(C=0.001, kernel='linear'),SVC(C=0.01, kernel='linear'),
                                 SVC(C=0.1, kernel='linear'),SVC(C=1, kernel='linear'),
                                 SVC(C=10, kernel='linear'), SVC(C=100, kernel='linear'),
                                 SVC(C=1000, kernel='linear'), SVC(C=10000, kernel='linear') ] } ]

    score = 'f1_micro' #['precision', 'recall']
    clf = GridSearchCV(OneVsRestClassifier(SVC(C=1)), param_grid=mlparams, cv=5,
                       verbose=100, scoring = score, n_jobs=4)
    clf.fit(trainfeat, traincl)
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    clpred = clf.predict(testfeat)
    print(classification_report(testcl, clpred))
    print 'micro f1 %.2f' % f1_score(testcl, clpred, average='micro',pos_label=None)
    print 'macro f1 %.2f' % f1_score(testcl, clpred, average='macro',pos_label=None)
    print 'micro recall %.2f' % recall_score(testcl, clpred, average='micro',pos_label=None)
    print 'macro recall %.2f' % recall_score(testcl, clpred, average='macro',pos_label=None)
    print 'micro prec %.2f' %  precision_score(testcl, clpred, average='micro',pos_label=None)
    print 'macro prec %.2f' %  precision_score(testcl, clpred, average='macro',pos_label=None)



def trainSingleClassClasifier(label, gridSearch = True):
    allLabels = labels_rights_final()
    if label not in allLabels: raise Exception('label is not among labels_rights')
    trainfeat, traincl = loadDataset('devel_features'), loadDataset('devel_'+label)
    testfeat, testcl =  loadDataset('test_features'), loadDataset('test_'+label)
    # Set the parameters by cross-validation
    #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
    #                    {'kernel': ['linear'], 'C': [1]}] #  10, 100, 1000
    mlparams = [{ #'class_weight': [{0:1,1:1},{0:10,1:1},{0:100,1:1},{0:1000,1:1}],
                 'C': [0.01, 0.1, 1, 10, 100, 1000]}] #  10, 100, 1000
    score = 'f1' #'f1_micro' #['precision', 'recall']
    #print traincl == 0.0
    #print traincl == 1
    if gridSearch:
        cv = StratifiedKFold(traincl, 5)
        clf = GridSearchCV(LinearSVC(), param_grid=mlparams, cv=cv, verbose=100, scoring = score, n_jobs=2)
        clf.fit(trainfeat, traincl)
    else:
        clf = LinearSVC()
        clf.fit(trainfeat, traincl)

    if gridSearch:
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
        print()

    testpred = clf.predict(testfeat)
    print(classification_report(testcl, testpred))
    print 'micro f1 %.2f' % f1_score(testcl, testpred, average='micro', pos_label=None)
    print 'macro f1 %.2f' % f1_score(testcl, testpred, average='macro', pos_label=None)
    print 'micro recall %.2f' % recall_score(testcl, testpred, average='micro', pos_label=None)
    print 'macro recall %.2f' % recall_score(testcl, testpred, average='macro', pos_label=None)
    print 'micro prec %.2f' %  precision_score(testcl, testpred, average='micro', pos_label=None)
    print 'macro prec %.2f' %  precision_score(testcl, testpred, average='macro', pos_label=None)
    return clf

def tresholdSingleClassClasifier(label, labelClass = 0):
    allLabels = labels_rights_final()
    if label not in allLabels: raise Exception('label is not among labels_rights')
    trainfeat, traincl = loadDataset('devel_features'), loadDataset('devel_'+label)
    testfeat, testcl =  loadDataset('test_features'), loadDataset('test_'+label)
    score = 'f1' #'f1_micro' #['precision', 'recall']
    #clf = TresholdLinearClassifier(LinearSVC)
    clf = TresholdLinearClassifier(SVC, kernel='linear')
    #clf = RandomForestClassifier()
    clf.fit(trainfeat, traincl)
    clf.testPredict(trainfeat, traincl)
    clf.testPredict(testfeat, testcl)

    print '\nclassification results'
    clpred = clf.predict(testfeat)
    print(classification_report(testcl, clpred))
    print 'micro f1 %.2f' % f1_score(testcl, clpred, average='micro', pos_label=None)
    print 'macro f1 %.2f' % f1_score(testcl, clpred, average='macro', pos_label=None)
    print 'micro recall %.2f' % recall_score(testcl, clpred, average='micro', pos_label=None)
    print 'macro recall %.2f' % recall_score(testcl, clpred, average='macro', pos_label=None)
    print 'micro prec %.2f' %  precision_score(testcl, clpred, average='micro', pos_label=None)
    print 'macro prec %.2f' %  precision_score(testcl, clpred, average='macro', pos_label=None)
    return clf


from sklearn.pipeline import Pipeline
from experiments.classification.transformers import CorpusTransformer
def trainSingleClassClasifierNew(label, gridSearch = True):
    allLabels = labels_rights_final()
    if label not in allLabels: raise Exception('label is not among labels_rights')
    devel = 'labeled/devel_200-1000_ristov_corrected.txt'
    test = 'labeled/test_ristov1000-1200_damir2000-2800_cleaned.txt'
    devel_texts, devel_labels = getTextsAndLabelsFromFile(devel, label)
    test_texts, test_labels = getTextsAndLabelsFromFile(test, label)
    pipe = Pipeline([('corpus_transform', CorpusTransformer('us_politics')),('svc', LinearSVC())])
    pipe.fit(devel_texts, devel_labels)
    testpred = pipe.predict(test_texts)
    print(classification_report(test_labels, testpred))
    print 'micro f1 %.2f' % f1_score(test_labels, testpred, average='micro', pos_label=None)
    print 'macro f1 %.2f' % f1_score(test_labels, testpred, average='macro', pos_label=None)
    print 'micro recall %.2f' % recall_score(test_labels, testpred, average='micro', pos_label=None)
    print 'macro recall %.2f' % recall_score(test_labels, testpred, average='macro', pos_label=None)
    print 'micro prec %.2f' %  precision_score(test_labels, testpred, average='micro', pos_label=None)
    print 'macro prec %.2f' %  precision_score(test_labels, testpred, average='macro', pos_label=None)
    return pipe


