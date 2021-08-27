from os import path
from pytopia.context.ContextResolver import resolve
from topic_coverage.resources.pytopia_context import topicCoverageContext
from topic_coverage.experiments.concept_types.subset_topic_model import SubsetTopicModel
from gtar_context.semantic_topics.construct_model import MODEL_ID as GTAR_REFMODEL

from topic_coverage.settings import news_semantic_annot

pairsFolder=news_semantic_annot

def __initR():
    import rpy2.robjects as ro
    R = ro.r
    R('require(irr)')
    R('options(warn=-1)')
    return R

def parseLabeledTopicSample(fname):
    '''
    Parse txt with themes (topics described by words and documents) labeled
    as concrete/abstract and issue/non-issue.
    :return: List of parsed pairs represented as (themeId, list of string labels/tags)
    '''
    import codecs, re
    themes, labels = [], []
    def parseThemeLine(l):
        ''' if equals THEME NNN; return number NNN, else return None '''
        if re.match('^THEME [0-9]+;.*', l):
            #print l
            themeStr = l.split(';')[0]
            themeId = int(themeStr.split()[1])
            assert l.startswith('THEME %d;'%themeId)
            return themeId
        else: return None
    def parseLabels(l):
        for match in re.findall(r'\[[ACIaci]\??\]', l):
            labels.append(str(match)[1])
    def addTheme(themeId): # add parsed pair
        themedata = (themeId, labels)
        themes.append(themedata)
        #print themeId, labels
    start = True; prevTheme = None
    for i, l in enumerate(codecs.open(fname, 'r', 'utf-8').readlines()):
        l = l.strip()
        theme = parseThemeLine(l)
        if theme is not None:
            if not start:
                addTheme(prevTheme)
                labels = [];
            prevTheme = theme
            start = False
        parseLabels(l)
    addTheme(prevTheme)
    return themes

def acbin(labeledThemes):
    ''' Convert themeId -> list of string labels to themeId -> 0 (concrete) or 1 (abstract) '''
    res = {}
    for tid, labs in labeledThemes:
        if 'C' in labs: res[tid] = 0
        elif 'A' in labs: res[tid] = 1
        else:
            print 'theme %d labeled as neither abstract nor concrete', tid, labs
    return res

def issuebin(labeledThemes):
    ''' Convert themeId -> list of string labels to themeId -> 0 (not issue) or 1 (issue) '''
    return { tid : int('I' in labs) for tid, labs in labeledThemes}

def calculateIaa(themeLabelings, variables='nominal'):
    '''
    Calc. krippendorph alpha
    :param themeLabelings: iterable of labeled theme sets, ie maps {themeId->0/1}
    '''
    r = __initR()
    N = len(themeLabelings)
    # check all labelings have the same theme ids
    themeIds = set(themeLabelings[0].keys())
    for tl in themeLabelings:
        assert themeIds == set(tl.keys())
    # create string representations of labeling aligned on same ordering of themeIds
    flatlabs = []
    for tl in themeLabelings:
        flab = ','.join([str(tl[tid]) for tid in tl.keys()])
        print flab
        flatlabs.append(flab)
    alllabsflat = ','.join(fl for fl in flatlabs)
    #print flatten
    matrixCode = 'matrix(c(%s), nrow=%d, byrow=TRUE)' % (alllabsflat, N)
    # call kripp.alpha with the matrix, extract result
    matrix = r(matrixCode)
    kripp = r['kripp.alpha']
    result = kripp(matrix, variables)
    print result

def createModel(themesFiles, labelInclude=[], labelExclude=[], refmodel=GTAR_REFMODEL):
    '''
    Create SubsetTopicModel that corresponds to a subset of ref. model topics
    :param themesFiles: list of txt files with labeled themes, must cover all ref. themes
    :param labelInclude: theme is included if its labels include all of these labels
    :param labelExclude: theme is excluded if its labels include any of these labels
    :return:
    '''
    topicIds = set() ; parsed = []
    for f in themesFiles:
        s = parseLabeledTopicSample(f); parsed.extend(s)
        for tid, _ in s: topicIds.add(tid)
    refmodel = resolve(refmodel)
    assert topicIds == set(refmodel.topicIds())
    labelInclude, labelExclude = set(labelInclude), set(labelExclude)
    def acceptTheme(l):
        l = set(l)
        if labelInclude: incl = labelInclude.issubset(l)
        else: incl = True
        if labelExclude: excl = bool(sum(ex in l for ex in labelExclude))
        else: excl = False
        return incl and not excl
    #print 'FILTER', labelInclude, labelExclude
    #for tid, l in parsed:
    #    if acceptTheme(l): print tid, ','.join(sorted(l))
    submodelTopics = [tid for tid, l in parsed if acceptTheme(l)]
    stm = SubsetTopicModel('uspolThemeSubset', refmodel,
                           submodelTopics,  origTopicIds=True, include=labelInclude, exclude=labelExclude)
    return stm

finalAnnotationFiles = [ 'uspolThemesAnnotated.txt' ]

def uspolThemeSubset(labelInclude = [], labelExclude=[]):
    files = [ path.join(pairsFolder, f) for f in finalAnnotationFiles ]
    model = createModel(files, labelInclude, labelExclude)
    return model

def uspolAbstractThemes(): return uspolThemeSubset(['A'])
def uspolAbstractIssues(): return uspolThemeSubset(['A', 'I'])
def uspolConcreteThemes(): return uspolThemeSubset(['C'])
def uspolConcreteIssues(): return uspolThemeSubset(['C', 'I'])
def uspolIssueThemes(): return uspolThemeSubset(['I'])
def uspolNonissueThemes(): return uspolThemeSubset([], ['I'])

def coverageAnalysis(model):
    from topic_coverage.experiments.coverage.experiment_runner import evaluateCoverage
    from pytopia.context.Context import Context
    print model.id
    print '# themes', model.numTopics()
    with Context('subsetmodel_ctx', model):
        evaluateCoverage(eval='metrics', corpus='uspol', refmodel=model, numModels=10, bootstrap=20000)

def calculateCoverages():
    print 'COV.ABSTRACT'
    coverageAnalysis(uspolAbstractThemes())
    print 'COV.CONCRETE'
    coverageAnalysis(uspolConcreteThemes())
    print 'COV.ISSUE'
    coverageAnalysis(uspolIssueThemes())
    print 'COV.NON-ISSUE'
    coverageAnalysis(uspolNonissueThemes())

def calcIaa():
    labrist = parseLabeledTopicSample(path.join(pairsFolder, 'uspolThemesSampleRistov.txt'))
    labdam = parseLabeledTopicSample(path.join(pairsFolder, 'uspolThemesSampleDamir.txt'))
    print 'ABSTRACT / CONCRETE'
    calculateIaa([acbin(labrist), acbin(labdam)])
    print 'ISSUE / NONISSUE'
    calculateIaa([issuebin(labrist), issuebin(labdam)])

def disagree():
    labrist = parseLabeledTopicSample(path.join(pairsFolder, 'uspolThemesSampleRistov.txt'))
    labdam = parseLabeledTopicSample(path.join(pairsFolder, 'uspolThemesSampleDamir.txt'))
    printDisagreements(labrist, labdam, ['ristov', 'damir'])

def printDisagreements(lab1, lab2, llabels = ['labeler1', 'labeler2']):
    for i, issuelab in enumerate(lab1):
        tid, l1 = issuelab ; tid2, l2 = lab2[i]
        assert tid == tid2
        if set(l1) != set(l2):
            print 'THEME %d: %s - %s  ;  %s - %s'%\
                  (tid, llabels[0], sorted(l1), llabels[1], sorted(l2))


if __name__ == '__main__':
    with topicCoverageContext():
        #calcIaa()
        #disagree()
        calculateCoverages()
