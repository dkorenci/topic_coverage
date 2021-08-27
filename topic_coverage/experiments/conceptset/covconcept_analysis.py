from topic_coverage.resources.pytopia_context import topicCoverageContext

from topic_coverage.modelbuild.modelset_loading import modelset1Families
from topic_coverage.resources.modelsets import *
from topic_coverage.experiments.correlation.experiment_runner import refmod
from topic_coverage.experiments.measure_factory import supervisedTopicMatcher

def coveredTopics(refmodel, model, matcher):
    ''' Calculates and returns a set of ids of covered refmodel topics. '''
    cov = set()
    for rt in refmodel:
        covered = False
        for t in model:
            if matcher(rt, t):
                covered = True
                break
        if covered: cov.add(rt.topicId)
    return cov

def coveredConceptsAnalysis(refmodel, modelset, matcher):
    numRef = refmodel.numTopics()
    print 'ref.set.size: %d' % numRef
    print 'model: ', modelset[0].id
    print 'num.models: %d' % len(modelset)
    # caching of covered topics
    covtopics = {} # model id -> covered reftopics
    def covered(model):
        if model.id not in covtopics:
            covtopics[model.id] = coveredTopics(refmodel, model, matcher)
        return covtopics[model.id]
    # average num. concepts covered by two models
    avgMatch = 0.0 ; numPairs = 0
    avgCov = 0.0
    for i, mi in enumerate(modelset):
        c = covered(mi); avgCov += len(c)
        for j in range(i+1, len(modelset)):
            mj = modelset[j]
            ci = covered(mi)
            cj = covered(mj)
            avgMatch += len(ci.intersection(cj))
            numPairs += 1
    avgMatch /= numPairs; avgCov /= len(modelset)
    print 'avg.concept.match: %g'%avgMatch
    print 'avg.coverage: %g'%avgCov
    # number of concepts covered by all models
    allcov = None
    for m in modelset:
        c = covered(m)
        if allcov is None: allcov = c
        else: allcov = allcov.union(c)
    numCov = len(allcov)
    print 'coverage.by.all.models: %d, %g percent' % (numCov, numCov/float(numRef))

def runCovConceptsAnalysis(corpus='uspol', numModels=10, modelsFolder=prodModelsBuild,
                            families='all', numT=[50, 100, 200], strict=True):
    '''
    :param numModels: number of models loaded per (model family, num topics) combination
    :param modelsFolder: folder with stored models
    :param families: list of model families, or 'all'
    :param numT: list of numbers of topics
    :return:
    '''
    msets, mctx, _ = modelset1Families(corpus, numModels, modelsFolder, families, numT)
    refm = refmod(corpus)
    matcher = supervisedTopicMatcher(corpus, strict, cached=True)
    with mctx:
        for mset in msets:
            coveredConceptsAnalysis(refm, mset, matcher)

if __name__ == '__main__':
    with topicCoverageContext():
        for mf in ['lda', 'alda', 'nmf']:
            for T in [50, 100, 200]:
                runCovConceptsAnalysis(corpus='pheno', families=[mf], numT=[T])
        runCovConceptsAnalysis(corpus='pheno', families=['pyp'], numT=[300])