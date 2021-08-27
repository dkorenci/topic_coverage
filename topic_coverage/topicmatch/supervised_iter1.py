from topic_coverage.topicmatch.supervised_data import dataset, getLabelingContext
from topic_coverage.topicmatch.supervised_iter0 import cvAllClassifiers

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import f1_score, accuracy_score

from topic_coverage.topicmatch.supervised_data import dataset, getLabelingContext
from topic_coverage.topicmatch.supervised_iter0 import cvAllClassifiers


def labelingAgreement(scores=[f1_score, accuracy_score]):
    ''' Measures of agreement on labeled training data '''
    from topic_coverage.topicmatch.labeling_iter1_analysis import calculateIaa, loadLabelings, createCompareData
    from topic_coverage.topicmatch.supervised_data import filesForLabeler, dataFolder
    calculateIaa(filesForLabeler, dataFolder)
    print
    # calc. pairwise classification scores
    cmpPairs = loadLabelings(filesForLabeler, dataFolder)
    pids, labeler2labelings = createCompareData(cmpPairs)
    labelers = labeler2labelings.keys()
    labels = {l: [str(labeler2labelings[l][pid]) for pid in pids] for l in labelers}
    for l1 in labelers:
        for l2 in labelers:
            if l1 != l2:
                print 'true: %s; pred: %s' % (l1, l2)
                for sc in scores:
                    if sc == f1_score:
                        for avg in ['macro', 'micro']:
                            p = {'average' : avg}
                            print '%s: %g' % (sc.__name__+' '+avg, sc(labels[l1], labels[l2], **p))
                    else:
                        print '%s: %g'%(sc.__name__, sc(labels[l1], labels[l2]))
                print


def runSupervised(corpus='uspol'):
    ctx = getLabelingContext(corpus)
    with ctx:
        #cvAllClassifiers(dataset(0.75), 'all-distances', 'value-inv')
        #cvAllClassifiers(dataset(0.75), 'vectors', 'cosine')
        # cvFitAndLearnCurve(gbt(), dataset(0.75), 'all-distances', 'all', 200, 10, average=10,
        #                    score=f1_score)
        #cvFitAndLearnCurve(randomForest(), dataset(0.75), 'all-distances', 'cosine', 100, 10, average=10, score=f1_score)
        #cvFitAndLearnCurve(mlp(), uspolBinLab(),'all-distances', 'cosine', 100, 10, average=10, score=f1_score)
        # cvFitAndLearnCurve(randomForest(), 'all-distances', 'all',
        #                    100, 10, average=10, score=accuracy_score)
        cvAllClassifiers(dataset(0.75, corpus=corpus), 'all-distances', 'all')

if __name__ == '__main__':
    runSupervised('uspol')
    #runSupervised('pheno')
    #labelingAgreement()