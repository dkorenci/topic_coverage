'''
Evaluation of coverage measures: supervised matcher-based, CTC, ...
'''

from topic_coverage.resources.pytopia_context import topicCoverageContext
from gtar_context.semantic_topics.construct_model import MODEL_ID as GTAR_REFMODEL
from phenotype_context.phenotype_topics.construct_model import MODEL_DOCS_ID as PHENO_REFMODEL
from pytopia.measure.topic_distance import cosine, hellinger, l1norm
from pyutils.stat_utils.utils import Stats
from topic_coverage.experiments.correlation.measure_correlation_utils import *
from topic_coverage.experiments.coverage.coverage_plots import coverageThresholdCurves, precisionRecallPlot
from topic_coverage.experiments.measure_factory import *
from topic_coverage.modelbuild.modelset_loading import modelset1Families
from topic_coverage.experiments.correlation.experiment_runner import refmod
from topic_coverage.settings import topic_models_folder
from time import time

def coverageScoringExperiment(target, models, metrics, bootstrap=None, timing=None):
    '''
    Evaluate coverage of the target models.
    For each group of models statistics for every metric is displayed.
    :param target: target model containing topics to be covered
    :param models: list of lists of models
    :param metrics: list of coverage-scoring metrics, or a single metric,
            returning a value for a (target, covering) pair of models
    :return:
    '''
    if not hasattr(metrics, '__iter__'): metrics = [metrics]
    if timing: t0 = time()
    for modelset in models:
        print modelset[0].id
        for metric in metrics:
            scores = [ metric(target, model) for model in modelset ]
            print metric.id
            print Stats(scores)
            #print ', '.join('%g'%s for s in scores)
            if bootstrap: bootstrap_average(scores, bootstrap)
        print
    if timing:
        t = time() - t0
        print('experiment finished in %g seconds'%t)

def bootstrap_average(scores, numIter):
    import numpy
    from sklearn.utils import resample
    N = len(scores)
    stats = list()
    for i in range(numIter):
        # prepare train and test sets
        scoreResamp = resample(scores, n_samples=N)
        score = numpy.mean(scoreResamp)
        stats.append(score)
    for alpha in [0.95, 0.99]:
        lowp = ((1.0 - alpha) / 2.0) * 100
        highp = (alpha + ((1.0 - alpha) / 2.0)) * 100
        lower = numpy.percentile(stats, lowp)
        upper = numpy.percentile(stats, highp)
        print(' average : %.1f confidence interval [%.2f, %.2f]' % (alpha * 100, lower, upper))

def evaluateCoverage(corpus='uspol', eval='metrics', method='sup.strict', numModels=10, modelsFolder=topic_models_folder,
                     families='all', numT = [50, 100, 200],
                     min=0.0, max=1.0, intervals=100, distMetric=cosine, addmodels=None, label=None,
                     custlabels=None, refmodel=None, supCovCache=True, ctcCovCache=True, topicMatchCache=True,
                     bootstrap=None, timing=None):
    msets, mctx, labels = modelset1Families(corpus, numModels, modelsFolder, families, numT)
    if refmodel is not None: refmodel = resolve(refmodel)
    else:
        if corpus == 'uspol': refmodel = resolve(GTAR_REFMODEL)
        elif corpus == 'pheno': refmodel = resolve(PHENO_REFMODEL)
    if addmodels is not None:
        for mset, label in addmodels:
            msets.append(mset); labels.append(label)
    with mctx:
        if eval == 'metrics':
            covscorers = []
            if method == 'sup.strict': covscorers.append(
                supervisedModelCoverage(corpus, True, covCache=supCovCache, matchCache=topicMatchCache))
            if method == 'sup.nonstrict': covscorers.append(
                supervisedModelCoverage(corpus, False, covCache=supCovCache, matchCache=topicMatchCache))
            if method == 'ctc.strict': covscorers.append(ctcModelCoverage(True, cached=ctcCovCache))
            if method == 'ctc.nonstrict': covscorers.append(ctcModelCoverage(False, cached=ctcCovCache))
            coverageScoringExperiment(refmodel, msets, covscorers, bootstrap=bootstrap, timing=timing)
        elif eval == 'ctc':
            plotlabel = 'ctc_%s_%s_%d' %(distMetric.__name__, corpus, numModels)
            if label: plotlabel += '_%s'%label
            print plotlabel
            if custlabels: labels = custlabels
            coverageThresholdCurves(refmodel, msets, min=min, max=max, intervals=intervals,
                                    labels=labels, xticks=20, distMetric=distMetric, plotlabel=plotlabel)
        elif eval == 'pr':
            plotlabel = 'pr_%s_%s_%d' %('sup.strict', corpus, numModels)
            coverage = supervisedModelCoverage(corpus, True, covCache=supCovCache, matchCache=topicMatchCache)
            precision = supervisedModelPrecision(corpus, True, covCache=supCovCache, matchCache=topicMatchCache)
            #matcher = supervisedTopicMatcher(corpus, strict=True, cached=topicMatchCache)
            if label: plotlabel += '_%s'%label
            print plotlabel
            if custlabels: labels = custlabels
            #precisionRecallPlot(refmodel, msets, matcher=matcher, labels=labels, plotlabel=plotlabel)
            precisionRecallPlot(refmodel, msets, modelCov=coverage, modelPrec=precision, labels=labels, plotlabel=plotlabel)


def runCovmetricProdbuild(method, corpus, bootstrap=None):
    print 'RUNNING EVALUATION OF ALL PRODUCTION MODELS, corpus: %s, method: %s' % (corpus, method)
    print
    evaluateCoverage(eval='metrics', corpus=corpus, method=method, numModels=10,
                     modelsFolder=topic_models_folder, bootstrap=bootstrap)

def timeCovmetricProdbuild(method, corpus, numModels=10, families='all'):
    evaluateCoverage(eval='metrics', corpus=corpus, method=method, numModels=numModels, families=families,
                     modelsFolder=topic_models_folder, timing=True, bootstrap=None,
                     supCovCache=False, ctcCovCache=False, topicMatchCache=False)

def testPlot():
    evaluateCoverage(eval='pr', corpus='pheno', numModels=3,
                     families=['lda', 'nmf'], numT=[50, 100],
                     custlabels=['LDA 50', 'LDA 100',
                                 'NMF 50', 'NMF 100'])

def prPlotsFull():
    for c in ['uspol', 'pheno']:
        evaluateCoverage(eval='pr', corpus=c, numModels=10,
                 custlabels=['LDA 50', 'LDA 100', 'LDA 200',
                             'aLDA 50', 'aLDA 100', 'aLDA 200',
                             'NMF 50', 'NMF 100', 'NMF 200',
                             'PYP'])

if __name__ == '__main__':
    with topicCoverageContext():
        # evaluation of model coverage
        #runCovmetricProdbuild('sup.strict', 'uspol', 20000)
        #runCovmetricProdbuild('ctc.nonstrict', 'uspol', 20000)
        #runCovmetricProdbuild('sup.strict', 'pheno', 20000)
        #runCovmetricProdbuild('ctc.nonstrict', 'pheno', 20000)
        # plot CDC graph
        evaluateCoverage(eval='ctc', corpus='uspol', numModels=10, distMetric=cosine, min=0.0, max=1.0,
                         custlabels=['LDA 50', 'LDA 100', 'LDA 200',
                                     'aLDA 50', 'aLDA 100', 'aLDA 200',
                                     'NMF 50', 'NMF 100', 'NMF 200',
                                     'PYP'])
        #testPlot()
        #prPlotsFull()
        #timeCovmetricProdbuild('sup.strict', 'uspol') #, numModels=1, families=['lda'])
        #timeCovmetricProdbuild('ctc.nonstrict', 'uspol')
        #timeCovmetricProdbuild('sup.strict', 'pheno')
        #timeCovmetricProdbuild('ctc.nonstrict', 'pheno')