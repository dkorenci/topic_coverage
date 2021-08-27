'''
Tests and preliminary experiments with measuring referent topic sets
from us_politics and phenotype datasets using LdaGibbsTopicModel
'''

from pytopia.context.ContextResolver import resolve

# requires corpus_topic_index_builder, corpus_index_builder, bow_corpus_builder
def topicSizes(model, topics, thresh=0.1):
    '''
    Compute sizes of model's topic, either as number of documents topics occur in
    or by amount of text assigned to the topic
    :param topics: either a list of topics ids or integer i denoting topics
            with ids [i, model.numTopics()-1]
    :param thresh: threshold for topic-is-in-document decision, or None in which case
            fraction of topic-assigned text per document is used
    :return: map {topicId->topicSize}
    '''
    # create corpus topic index
    model = resolve(model)
    corpus = resolve(model.corpus)
    if isinstance(topics, int): topics = range(topics, model.numTopics())
    cti = resolve('corpus_topic_index_builder')(model=model, corpus=corpus)
    # calculate topic sizes
    tsize = {tid:0 for tid in topics}
    if thresh is not None:
        for tid in topics:
            #print tid, len(cti.topicTexts(tid))
            topicTexts = sum(1 for _, w in cti.topicTexts(tid) if w > thresh)
            tsize[tid] = topicTexts
    else:
        ci = resolve('corpus_index_builder')(corpus.id)
        bowCorpus = resolve('bow_corpus_builder')(corpus, text2tokens=model.text2tokens,
                                                  dictionary=model.dictionary)
        for i, bow in enumerate(bowCorpus):
            numTokens = sum(cnt for _, cnt in bow)
            topicProps = cti.textTopics(ci[i])
            for tid in topics:
                if topicProps[tid] > 0.03:
                    tsize[tid] += topicProps[tid]*numTokens
    return tsize

class TopicsDistEqual:

    def __init__(self, dist, thresh):
        self._dist = dist
        self._thresh = thresh

    def __call__(self, t1, t2):
        return self._dist(t1.vector, t2.vector) <= self._thresh

from pytopia.measure.topic_distance import cosine as cosineDist
topicsEqualCosine04 = TopicsDistEqual(cosineDist, 0.4)

def topicCoverage(topics, covModels, topicEqual):
    '''
    Calculate coverage of a set of topic by a set of models
    :param topics: list of Topic objects
    :param covModels: list of models
    :param topicEqual: topic equality function on pairs of Topics
    :return: map {topicId -> numEqualTopicFromCovModels}
    '''
    tcovered = { t.topicId:0 for t in topics }
    for t in topics:
        for m in covModels:
            model = resolve(m)
            for ctopic in model:
                if topicEqual(ctopic, t): tcovered[t.topicId] += 1
    return tcovered
