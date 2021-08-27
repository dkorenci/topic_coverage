'''
Tests and preliminary experiments with measuring referent topic sets
from us_politics and phenotype datasets using LdaGibbsTopicModel
'''

from pytopia.context.ContextResolver import resolve
from pytopia.context.GlobalContext import GlobalContext
from matplotlib import pyplot as plt

from topic_coverage.experiments.ref_topics.measuring_models import \
        uspolMeasureModel1, uspolMeasureModel2, phenoMeasureModel1, \
        measuringModelsContext

from topic_coverage.experiments.ref_topics.measuring_tools import \
    topicSizes, topicCoverage, TopicsDistEqual, topicsEqualCosine04

from topic_coverage.resources import pytopia_context

# requires corpus_topic_index_builder
def plotTopicSizeDistribution(model, topics, thresh=0.1):
    '''
    :param topics: either a list of topics ids or integer i denoting topics
            with ids [i, model.numTopics()-1]
    '''
    measuringModelsContext() # todo use as py context (with keyword)
    #print GlobalContext.get()
    tsize = topicSizes(model, topics, thresh)
    model = resolve(model); corpus = resolve(model.corpus)
    # plot histogram of topic sizes
    fig, axes = plt.subplots(2)
    fig.suptitle('referent topics 4 corpus: %s' % corpus.id, fontsize=16)
    if thresh is None:
        bowCorpus = resolve('bow_corpus_builder')(corpus, text2tokens=model.text2tokens,
                                              dictionary=model.dictionary)
        corpusTokens = sum(sum(cnt for _, cnt in bow) for bow in bowCorpus)

    if thresh: vals = tsize.values()
    else: vals = [ float(v)/corpusTokens for v in tsize.values() ]
    axes[0].hist(vals, bins=100, color='gray')
    if thresh:
        xlab = 'num. documents containing the topic (%g perc. of text in topic)' % (thresh * 100)
    else:
        xlab = 'proportion of corpus text'
    axes[0].set_xlabel(xlab)
    axes[0].set_ylabel('number of topics')
    # plot occurence topic probabilities
    if thresh: # probability is a fraction of texts topic occurs in
        cti = resolve('corpus_topic_index_builder')(model=model, corpus=corpus)
        ndocs = float(len(cti))
        probs = [tt/ndocs for _, tt in tsize.iteritems()]
    else: # probability is fraction of corpus text assigned to topic
        probs = [ ttokens/corpusTokens for ttokens in tsize.values() ]
    probs.sort(reverse=True)
    axes[1].plot(range(len(probs)), probs)
    axes[1].set_xlabel('all topics, by descending probability')
    axes[1].set_ylabel('topic probability')
    #plt.tight_layout(pad=0)
    plt.show()

def topicSizeVsDetectability(covered, topics, thresh, tequal,
                             corpus, model, numTopics, numModels):
    from topic_coverage.experiments.clustered_models_v0 import baseModelSet, modelFolders
    from topic_coverage.modelbuild.modelbuild_iter1 import addModelsToGlobalContext
    measuringModelsContext() # todo use as py context (with keyword)
    addModelsToGlobalContext()
    # get topic sizes
    covered = resolve(covered)
    if isinstance(topics, int): topics = range(topics, covered.numTopics())
    tsize = topicSizes(covered, topics, thresh)
    # measure coverage
    modelset1 = baseModelSet('', modelFolders[corpus, model, numTopics],
                             numModels, returnIds=True)
    ctopics = [ covered.topic(tid) for tid in topics ]
    tcovered = topicCoverage(ctopics, modelset1, tequal)
    fig, axes = plt.subplots()
    xaxis = [ tsize[tid] for tid in topics ]
    yaxis = [ tcovered[tid] for tid in topics ]
    axes.scatter(xaxis, yaxis)
    plt.show()

if __name__ == '__main__':
    plotTopicSizeDistribution(uspolMeasureModel2, 20, None)
    #plotTopicSizeDistribution(phenoMeasureModel1, 20, 0.1)
    # topicSizeVsDetectability(uspolMeasureModel2, 20, 0.1, topicsEqualCosine04,
    #                          'us_politics', 'nmf', 100, 15)
    #topicSizeVsDetectability(phenoMeasureModel1, 20, 0.1, topCosEq, 'pheno_corpus1', 'lda', 100, 15)