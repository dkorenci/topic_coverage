from topic_coverage.topicmatch.labeling_iter1_uspolfinal import  uspolLabelingModelsContex as uspolLabModels
from topic_coverage.topicmatch.labeling_iter1_pheno_schemedevel import phenoModelsLabeling, phenoLabelingModelsContex

from topic_coverage.settings import uspol_pairs_folder

K=5
dataFolder = uspol_pairs_folder
filesForLabeler = {
    lab: ['topicPairs[%d-%d]_%s.txt'%(pi, pi+50, lab) for pi in range(0, 50*(K-1)+1, 50) ]
    for lab in ['damir', 'ristov', 'petar']
}
for lab in ['damir', 'ristov', 'petar']:
    filesForLabeler[lab].append('_testni_%s.txt'%lab)

from topic_coverage.settings import phenotype_pairs_folder
phenoFinalFiles = {
    'barbara': ['_testni_barbara.txt',
                '1_topicPairs[0-50]_barbara.txt', '2_topicPairs[150-200]_barbara.txt', '3_topicPairs[200-250]_barbara.txt',
                '4_topicPairs[250-300]_barbara.txt', '5_topicPairs[300-350]_barbara.txt'],
    'bruno': ['_testni_bruno.txt',
              '1_topicPairs[0-50]_bruno.txt', '2_topicPairs[150-200]_bruno.txt', '3_topicPairs[200-250]_bruno.txt',
              '4_topicPairs[250-300]_bruno.txt', '5_topicPairs[300-350]_bruno.txt'],
    'jelena': ['_testni_jelena.txt',
               '1_topicPairs[0-50]_jelena.txt', '2_topicPairs[150-200]_jelena.txt', '3_topicPairs[200-250]_jelena.txt',
                '4_topicPairs[250-300]_jelena.txt', '5_topicPairs[300-350]_jelena.txt'],
}

def getFolderAndFiles(corpus='uspol'):
    '''
    Return folder and map of labeler -> files with labeled pairs, for a dataset.
    '''
    if corpus == 'uspol':
        labelerFiles = filesForLabeler
        folder = dataFolder
    elif corpus == 'pheno':
        labelerFiles = phenoFinalFiles
        folder = phenotype_pairs_folder
    return folder, labelerFiles

def rawDataset(corpus='uspol'):
    '''
    Parse txt files with labeled topics, for multiple annotators,
    return a list of (Topic, Topic, labels), where labels is a list per-annotator labels.
    '''
    from topic_coverage.topicmatch.data_iter0 import loadDataset, resolveTopic
    folder, labelerFiles = getFolderAndFiles(corpus)
    pid2topics = None
    pid2labels = None
    for i, lab in enumerate(labelerFiles):
        topics = loadDataset(folder, labelerFiles[lab], raw=True)
        if i == 0:
            pid2topics = { pid: (t1, t2) for pid, t1, t2, l in topics }
            pid2labels = {pid: [l] for pid, t1, t2, l in topics}
            pids = set(pid for pid, _, _, _ in topics)
        else:
            assert pids == set(pid for pid, _, _, _ in topics)
            assert pid2topics == { pid: (t1, t2) for pid, t1, t2, _ in topics }
            for pid, _, _, l in topics: pid2labels[pid].append(l)
    labeledPairs = []
    for pid in pids:
        t1, t2 = pid2topics[pid]
        labels = pid2labels[pid]
        labeledPairs.append((resolveTopic(t1), resolveTopic(t2), labels))
    return labeledPairs

def transformLabel(labels, labelAgg=1.0):
    '''
    :param labels: list of topic pair similarity labels, each assigned by one annotator
           labelAgg: strategy for label binarization,
            if labelAgg is a number, label with 1 pairs with label average >= the number
    :return: 0 or 1
    '''
    avg = sum(float(l) for l in labels) / len(labels)
    return 1 if avg >= labelAgg else 0

def dataset(labelAgg=1.0, corpus='uspol', split=False):
    '''
    Load raw dataset and binarize pair lables.
    :return: list of (Topic, Topic, 0/1) or (list of (Topic, Topic), list of labels) if split = True
    '''
    data = [ (t1, t2, transformLabel(lab, labelAgg)) for t1, t2, lab in rawDataset(corpus) ]
    if not split: return data
    else:
        pairs, labs = [(t1, t2) for t1, t2, _ in data], [l for _, _, l in data]
        return pairs, labs

def getLabelingContext(corpus='uspol'):
    if corpus == 'uspol': ctx = uspolLabModels()
    elif corpus == 'pheno': ctx = phenoLabelingModelsContex()
    return ctx

def calculateInterAnnotF1(corpus='uspol', labelAgg=0.75):
    from sklearn.metrics import f1_score
    with getLabelingContext(corpus):
        lpairs = rawDataset(corpus)
        N = len(lpairs[0][2]) # number of labelers
        print N
        f1avg = 0
        for lab in range(N):
            # averages of labelers != lab
            avgs = [ sum(float(l) for i, l in enumerate(labels) if i != lab)/(N-1)
                     for _, _, labels in lpairs ]
            #print avgs
            #print set(avgs)
            # averages -> {0,1}, by labelAgg thresholding
            res = [ int(avg >= labelAgg) for avg in avgs]
            #print res
            #print set(res)
            # labels of the labeler lab
            pred = [ int(labels[lab] >= labelAgg) for _, _, labels in lpairs ]
            f1 = f1_score(res, pred)
            print '%g'%f1
            f1avg += f1
        f1avg /= N
        print 'f1 avg: %g'%f1avg

def printAnnotationModels():
    from labeling_iter1_pheno_schemedevel import phenoModelsLabeling
    newsctx = getLabelingContext('uspol')
    bioctx = getLabelingContext('pheno')
    with newsctx:
        with bioctx:
            for m in newsctx:
                print m.id
            print
            for m in bioctx:
                print m.id

from topic_coverage.resources.pytopia_context import topicCoverageContext
if __name__ == '__main__':
    with topicCoverageContext():
        printAnnotationModels()
        #print filesForLabeler
        #print filesForLabeler['damir']
        #with uspolLabModels(): rawDataset()
        #with phenoModelsLabeling(1, 1, 3, context=True): rawDataset('pheno')
        #calculateInterAnnotF1('pheno', 0.5)