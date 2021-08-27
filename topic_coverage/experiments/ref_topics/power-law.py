'''
Tests for power-law fit of topic size distribution
'''

from pytopia.context.ContextResolver import resolve
from pytopia.context.GlobalContext import GlobalContext
from matplotlib import pyplot as plt

from topic_coverage.experiments.ref_topics.measuring_models import \
        uspolMeasureModel1, uspolMeasureModel2, phenoMeasureModel1, \
        measuringModelsContext

from topic_coverage.experiments.ref_topics.measuring_tools import \
    topicSizes, topicCoverage, TopicsDistEqual, topicsEqualCosine04
from topic_coverage.experiments.ref_topics.plpva import plpva

from topic_coverage.resources import pytopia_context
import numpy as np

# requires corpus_topic_index_builder
def testTopicSizeDistribution(model, topics, thresh=0.1):
    '''
    :param topics: either a list of topics ids or integer i denoting topics
            with ids [i, model.numTopics()-1]
    '''
    #print GlobalContext.get()
    tsize = topicSizes(model, topics, thresh)
    model = resolve(model); corpus = resolve(model.corpus)
    # plot histogram of topic sizes
    if thresh is None:
        bowCorpus = resolve('bow_corpus_builder')(corpus, text2tokens=model.text2tokens,
                                              dictionary=model.dictionary)
        corpusTokens = sum(sum(cnt for _, cnt in bow) for bow in bowCorpus)
    if thresh: vals = tsize.values()
    else: vals = [ float(v)/corpusTokens for v in tsize.values() ]
    #vals = [float(v) for v in vals]
    #print plpva(vals, 1, 'sample', 100)
    print 'sizes computing, calculating plpva'
    from powerlaw import Fit, Power_Law, distribution_fit, distribution_compare
    vals = np.array(vals)
    #
    fit = Fit(vals, discrete=True, xmin=5, verbose=True)
    for dist in ['exponential', 'lognormal', 'gamma']:
        print dist
        print fit.distribution_compare('power_law', dist)
    plaw = Power_Law(data=vals, discrete=True, xmin=5, verbose=True)
    res = distribution_fit(vals, discrete=True, xmin=5)
    for dist in res['fits']:
        print dist
        print res['fits'][dist]
    res = distribution_compare(vals, discrete=True, xmin=5)
    #print fit.KS(vals)
    #print plpva(vals, 5)
    # plot occurence topic probabilities
    if thresh: # probability is a fraction of texts topic occurs in
        cti = resolve('corpus_topic_index_builder')(model=model, corpus=corpus)
        ndocs = float(len(cti))
        probs = [tt/ndocs for _, tt in tsize.iteritems()]
    else: # probability is fraction of corpus text assigned to topic
        probs = [ ttokens/corpusTokens for ttokens in tsize.values() ]
    #print plpva(probs, 0)

if __name__ == '__main__':
    with measuringModelsContext():
        testTopicSizeDistribution(uspolMeasureModel2, 20, 0.1)
    #testTopicSizeDistribution(phenoMeasureModel1, 20, 0.1)
