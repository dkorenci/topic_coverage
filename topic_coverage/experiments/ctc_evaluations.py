from topic_coverage.resources import pytopia_context

from pytopia.measure.avg_nearest_dist import AverageNearestDistance, TopicCoverDist
from pytopia.measure.topic_distance import cosine as cosineDist
from stat_utils.utils import Stats
from pytopia.context.ContextResolver import resolve

from gtar_context.semantic_topics.construct_model import MODEL_ID as GTAR_REFMODEL
from phenotype_context.phenotype_topics.construct_model import MODEL_ID as PHENO_REFMODEL
from phenotype_context.phenotype_corpus.construct_corpus import CORPUS_ID as PHENO_CORPUS
from topic_coverage.experiments.experiment_iter1_thresh import baseModelsSets1

import numpy as np

def calculateCTC(refmodel, model, thresholds, distance=cosineDist, Cover=TopicCoverDist):
    '''
    Calculate coverage-threshold curve value for given models and matching method.
    The curve is formed by calculating coverage for each of the thresholds
     and the area under the curve is returned.
    :param refmodel: referent topic model that to be covered
    :param model: covering topic model
    :param thresholds: thresholds the coverage should be calculated for,
                either a list of thresholds or a 3-tuple of arguments for numpy.linspace
    :param distance: measure of topic distance the topic sameness will be based on
    :param Cover: model matcher class with constructor that accepts a distance measure and a threshold
    :return:
    '''
    refmodel, model = resolve(refmodel), resolve(model)
    if isinstance(thresholds, tuple):
        mn, mx, num = thresholds
        thresholds = np.linspace(mn, mx, num)
    # calculate coverages for the thresholds
    cov = [None] * len(thresholds)
    for i, t in enumerate(thresholds):
        cc = Cover(distance, t)
        cov[i] = cc(refmodel, model)
    # calculate area under the curve
    area = 0.0
    for i in range(1, len(thresholds)):
        area += (cov[i-1]+cov[i])*(thresholds[i]-thresholds[i-1])/2.0 # area of the trapeze
    return area

def averageCtc(refmodel, models):
    N = len(models)
    thresholds = np.linspace(0.0, 1.0, 20)
    ctc = [ calculateCTC(refmodel, m, thresholds) for m in models ]
    print Stats(ctc)

def ctc(corpus='uspol', topics=50, model='lda'):
    '''
    :param corpus: 'pheno' or 'uspol'
    :param topics: 50 or 100
    :param model: 'lda' or 'nmf'
    :return:
    '''
    if corpus == 'uspol':
        corpus = 'us_politics'
        refmodel = GTAR_REFMODEL
    elif corpus == 'pheno':
        corpus = PHENO_CORPUS
        refmodel = PHENO_REFMODEL
    else: raise Exception('unknown corpus: %s' % corpus)
    models = baseModelsSets1(corpus=corpus, topics=topics, basemodels=[model])[0]
    averageCtc(refmodel, models)

if __name__ == '__main__':
    ctc('pheno', 100, 'nmf')
