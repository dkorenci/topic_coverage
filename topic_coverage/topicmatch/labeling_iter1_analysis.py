from topic_coverage.topicmatch.labeling_iter1_pheno_schemedevel import phenoModelsLabeling
# labeled pheno and uspol data
from topic_coverage.topicmatch.supervised_data import filesForLabeler, dataFolder, phenotype_pairs_folder, phenoFinalFiles, uspol_pairs_folder

unlabeledPairsFolder = '/home/damir/Dropbox/projekti/doktorat/D1 eksplorativa/mjerenje pokrivenosti/supervised/' \
                 'oznaceni parovi/uspol_ternary_prelim/labeling_iter1_uspol_ternary [svi parovi]/'
labeledPairsUspolSchemedevel = '/home/damir/Dropbox/projekti/doktorat/D1 eksplorativa/mjerenje pokrivenosti/supervised/oznaceni parovi/uspol_schemedevel/'
labeledFiles1 = ['topicPairs[0-50]_schemedevel_damir.txt', 'topicPairs[0-50]_schemedevel_ristov.txt']
labeledFiles2 = ['topicPairs[150-200]_schemedevel_damir.txt', 'topicPairs[150-200]_schemedevel_ristov.txt',
                 'topicPairs[150-200]_schemedevel_jan.txt']
unlabeledFiles=[
    'topicPairs[0-50].txt', 'topicPairs[50-100].txt', 'topicPairs[100-150].txt', 'topicPairs[150-200].txt',
    'topicPairs[200-250].txt', 'topicPairs[250-300].txt', 'topicPairs[300-350].txt', 'topicPairs[350-400].txt',
    'topicPairs[400-450].txt', 'topicPairs[450-500].txt',
]
cmpSet1 = {
    'damir': ['topicPairs[0-50]_schemedevel_damir.txt'],
    'ristov': ['topicPairs[0-50]_schemedevel_ristov.txt']
}
cmpSet2 = {
    'damir': ['topicPairs[150-200]_schemedevel_damir.txt'],
    'ristov': ['topicPairs[150-200]_schemedevel_ristov.txt'],
    'jan': ['topicPairs[150-200]_schemedevel_jan.txt']
}
cmpSetAll = {
    'damir': ['topicPairs[0-50]_schemedevel_damir.txt', 'topicPairs[150-200]_schemedevel_damir.txt'],
    'ristov': ['topicPairs[0-50]_schemedevel_ristov.txt', 'topicPairs[150-200]_schemedevel_ristov.txt']
}

phenoKalib='/home/damir/Dropbox/projekti/doktorat/D1 eksplorativa/mjerenje pokrivenosti/supervised/pheno pairs/iaa/kalibracijski/'
cmpSetPhenoKalib = {
    'barbara': ['barbara.txt'],
    'bruno': ['bruno.txt'],
    'jelena': ['jelena.txt']
}

phenoTest='/home/damir/Dropbox/projekti/doktorat/D1 eksplorativa/mjerenje pokrivenosti/supervised/pheno pairs/iaa/testni/'
cmpSetPhenoTest = {
    'barbara': ['barbara.txt'],
    'bruno': ['bruno.txt'],
    'jelena': ['jelena.txt']
}

uspolAnn2Calib = '/home/damir/Dropbox/projekti/doktorat/D1 eksplorativa/mjerenje pokrivenosti/supervised/oznacavanje uspol stud/oznacavanje/calib/'
uspolAnn2CalibFiles = {
    'damir': ['_kalibracijski_damir.txt'],
    'ristov': ['_kalibracijski_ristov.txt'],
    'petar': ['_kalibracijski_petar.txt'],
}

uspolAnn2Test = '/home/damir/Dropbox/projekti/doktorat/D1 eksplorativa/mjerenje pokrivenosti/supervised/oznacavanje uspol stud/oznacavanje/test/'
uspolAnn2TestFiles = {
    'damir': ['_testni_damir.txt'],
    'ristov': ['_testni_ristov.txt'],
    'petar': ['_testni_petar.txt'],
}

def __initR():
    import rpy2.robjects as ro
    R = ro.r
    R('require(irr)')
    R('options(warn=-1)')
    return R

def loadLabelings(cmpSet, folder=labeledPairsUspolSchemedevel):
    '''
    :param cmpSet: map labeler (string) -> list of files
    :param folder: folder where files reside
    :return: cmpPairs: map labeler -> list of labeled topic pairs
    '''
    cmpPairs = {}
    for labeler, files in cmpSet.iteritems():
        pairs = loadDataset(folder, files, raw=True)
        cmpPairs[labeler] = pairs
    return cmpPairs

def createCompareData(cmpPairs):
    '''
    Verify and transform to format that enables analysis
    :param cmpPairs: map labeler -> list of labeled topic pairs
    :return: list po pair ids, map labeler -> ( map pairId -> label )
    '''
    labeler2labelings = {}; labelerPids = {}
    labelers = cmpPairs.keys()
    for labeler, pairs in cmpPairs.iteritems():
        labeler2labelings[labeler] = { pid:label for pid, t1, t2, label in pairs}
        labelerPids[labeler] = [pid for pid, _, _, _ in pairs]
    # assert all labels have same ids
    for i, labeler in enumerate(labelers):
        assert labelerPids[labeler] == labelerPids[labelers[0]]
    pids = labelerPids[labelers[0]]
    return pids, labeler2labelings

def printMismatches(cmpSet, dists=False, folder=None, ctx=None):
    if dists:
        from topic_coverage.topicmatch.data_iter0 import resolveTopic
        with ctx:
            files = None
            for f in cmpSet.itervalues():
                files = f; break
            pairs = loadDataset(folder, files, raw=True)
            pid2dist = { pid: cosine(resolveTopic(t1).vector, resolveTopic(t2).vector)
              for pid, t1, t2, label in pairs }
    cmpPairs = loadLabelings(cmpSet, folder)
    pids, labeler2labelings = createCompareData(cmpPairs)
    labelers = labeler2labelings.keys()
    mm = 0
    for pid in pids:
        pidLabels = set(labeler2labelings[labeler][pid] for labeler in labelers)
        if len(pidLabels) > 1:
            mm += 1
            print 'mismatch for pair: %s' % pid
            if dists: print 'distance: %g' % pid2dist[pid]
            for l in labelers:
                print l, labeler2labelings[l][pid]
            print
    print 'num pairs: %d, mismatches: %d ' % (len(pids), mm)
    print 'percentage of mismatches: %g' % (float(mm) / len(pids))

def calculateIaa(cmpSet, folder, variables='nominal'):
    '''
    Calc. krippendorph alpha
    :param cmpSet: map of annotator_label -> list of files with labeled pairs
    :param folder: folder where files are stored
    :return:
    '''
    cmpPairs = loadLabelings(cmpSet, folder)
    pids, labeler2labelings = createCompareData(cmpPairs)
    labelers = labeler2labelings.keys()
    r = __initR()
    rstrLabels = []
    print ' '.join(l for l in sorted(labelers))
    for l in labelers:
        labels = [ labeler2labelings[l][pid] for pid in pids ]
        rlabs = ','.join(('%s' % label) for label in labels)
        #print rlabs
        rstrLabels.append(rlabs)
    flatten = ','.join(rlab for rlab in rstrLabels)
    #print flatten
    matrixCode = 'matrix(c(%s), nrow=%d, byrow=TRUE)' % (flatten, len(labelers))
    # call kripp.alpha with the matrix, extract result
    matrix = r(matrixCode)
    kripp = r['kripp.alpha']
    result = kripp(matrix, variables)
    print result
    #for r in result: print r
    #return result[4][0]

def calculateInterAnnotSpearman(cmpSet, folder):
    '''
    Calc. krippendorph alpha
    :param cmpSet: map of annotator_label -> list of files with labeled pairs
    :param folder: folder where files are stored
    :return:
    '''
    from scipy.stats import spearmanr
    cmpPairs = loadLabelings(cmpSet, folder)
    pids, labeler2labelings = createCompareData(cmpPairs)
    labelers = labeler2labelings.keys()
    rstrLabels = []
    print ' '.join(l for l in sorted(labelers))
    for i, li in enumerate(labelers):
        for j, lj in enumerate(labelers):
            if j > i:
                print labelers[i], labelers[j]
                labelsi = [ labeler2labelings[li][pid] for pid in pids ]
                labelsj = [labeler2labelings[lj][pid] for pid in pids]
                corr = spearmanr(labelsi, labelsj)
                print 'spearman %g, p-val %g' % (corr[0], corr[1])

def analyzeUspolLabel4Distance(labeler='damir'):
    from topic_coverage.topicmatch.supervised_data import filesForLabeler, \
        dataFolder as uspolPairsFolder
    from topic_coverage.topicmatch.data_analysis_iter0 import plotClassDistribution, createIntervals
    from topic_coverage.topicmatch.labeling_iter1_uspolfinal import uspolModelsLabeling
    with uspolModelsLabeling(1, 1, 3, context=True):
        pairs = loadDataset(uspolPairsFolder, filesForLabeler[labeler])
        plotClassDistribution(pairs, cosine, createIntervals(0.0, 1.0, 10))

def analyzePhenoLabel4Distance(labeler='bruno'):
    from topic_coverage.topicmatch.data_analysis_iter0 import plotClassDistribution, createIntervals
    with phenoModelsLabeling(1, 1, 3, context=True):
        pairs = loadDataset(phenotype_pairs_folder, phenoFinalFiles[labeler])
        plotClassDistribution(pairs, cosine, createIntervals(0.0, 1.0, 10))

def uspolLabeledAnalysis(fileBlock=None):
    if fileBlock is not None:
        for lab in ['ristov', 'damir']:
            filesForLabeler[lab] = filesForLabeler[lab][fileBlock:fileBlock+1]
        print filesForLabeler
    print 'nominal'
    calculateIaa(filesForLabeler, dataFolder, 'nominal')
    print 'ordinal'
    calculateIaa(filesForLabeler, dataFolder, 'ordinal')

uspolAnn2TestFolder = '/home/damir/Dropbox/projekti/doktorat/D1 eksplorativa/mjerenje pokrivenosti/supervised/oznacavanje uspol stud/oznacavanje/test/'
uspolAnn2TestFiles = {
    #'damir': ['testni_damir.txt'],
    'ristov': ['testni_ristov.txt'],
    #'mislav': ['testni_mislav.txt'],
    'petar': ['testni_petar.txt'],
}
from topic_coverage.topicmatch.labeling_iter1 import *
def uspolAnnotround2Analysis():
    #calculateIaa(uspolAnn2CalibFiles, uspolAnn2Calib, 'nominal')
    #calculateIaa(uspolAnn2CalibFiles, uspolAnn2Calib, 'ordinal')
    calculateIaa(uspolAnn2TestFiles, uspolAnn2TestFolder, 'nominal')
    calculateIaa(uspolAnn2TestFiles, uspolAnn2TestFolder, 'ordinal')
    #printMismatches(uspolAnn2CalibFiles, True, uspolAnn2Calib, iter1UspolModelsTernarny(1, 1, 0, context=True))

def matchingStats(files, folder, pairwise=False):
    '''
    Print IAA stats, per file and for all files.
    Files must be in same order for all annotators.
    :param folder: folder where the files are at
    :param files: map annotator name -> list of file names
    :return:
    '''
    ann = files.keys()
    annPairs = [(ann[i], ann[j]) for i in range(len(ann)) for j in range(i+1, len(ann))]
    print 'ALL FILES', 'ALL-ANNOTATORS'
    calculateIaa(files, folder, 'nominal')
    calculateIaa(files, folder, 'ordinal')
    if pairwise:
        for a1, a2 in annPairs:
            f = { a1 : files[a1], a2 : files[a2] }
            print 'ALL FILES', a1, a2
            calculateIaa(f, folder, 'nominal')
            calculateIaa(f, folder, 'ordinal')
    print '\n'
    mcf = min(len(files[a]) for a in files)
    for i in range(mcf):
        fmap = {a:files[a][i:i+1] for a in files}
        file = files[ann[0]][i:i+1][0]
        print 'FILE', file, 'ALL ANNOTATORS'
        calculateIaa(fmap, folder, 'nominal')
        calculateIaa(fmap, folder, 'ordinal')
        if pairwise:
            for a1, a2 in annPairs:
                print 'FILE', file, a1, a2
                f = {a1: files[a1][i:i+1], a2: files[a2][i:i+1]}
                calculateIaa(f, folder, 'nominal')
                calculateIaa(f, folder, 'ordinal')
        print '\n'

def phenoFinalAnalysis():
    matchingStats(phenoFinalFiles, phenotype_pairs_folder, True)

def uspolFinalAnalysis():
    matchingStats(filesForLabeler, uspol_pairs_folder, True)

if __name__ == '__main__': #with iter1UspolModelsTernarny(1, 1, 0, context=True):
    #printMismatches(cmpSet1)
    #calculateIaa(cmpSet2, labeledPairsUspolSchemedevel, 'ordinal')
    #calculateIaa(filesForLabeler, dataFolder, 'nominal')
    #plotIntervalClassDistCosine(labeledPairsUspolSchemedevel, labeledFiles1+labeledFiles2)
    #analyzeUspolLabel4Distance('ristov')
    #analyzePhenoLabel4Distance('jelena')
    #printMismatches(cmpSetPhenoTest, True, phenoTest, phenoModelsLabeling(1, 1, 3, context=True))
    #calculateIaa(cmpSetPhenoKalib, phenoKalib, 'ordinal')
    #calculateIaa(cmpSetPhenoTest, phenoTest, 'ordinal')
    #calculateInterAnnotSpearman(cmpSetPhenoKalib, phenoKalib)
    #calculateInterAnnotSpearman(filesForLabeler, dataFolder)
    #uspolAnnotround2Analysis()
    #uspolLabeledAnalysis(5)
    phenoFinalAnalysis()
    #uspolFinalAnalysis()
    #analyzeUspolLabel4Distance('ristov')
    #analyzePhenoLabel4Distance('jelena')