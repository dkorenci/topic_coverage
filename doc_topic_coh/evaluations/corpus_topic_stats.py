from doc_topic_coh.resources import pytopia_context
from pytopia.context.ContextResolver import resolve

import numpy as np
from stat_utils.utils import Stats

def topicDist(corpusId, modelId):
    model = resolve(modelId)
    ctiBuilder = resolve('corpus_topic_index_builder')
    cti = ctiBuilder(corpus=corpusId, model=modelId)
    topics = cti.topicMatrix()
    return topics

def maxTopicStats(corpus, model):
    topics = topicDist(corpus, model)
    maxTop = np.max(topics, axis=1)
    maxInd = np.argsort(topics, axis=1)
    print topics[0]
    print maxInd[0]
    top2top = np.empty(topics.shape[0])
    numTop = topics.shape[1]
    for i in range(len(topics)):
        top2top[i] = topics[i, maxInd[i, numTop-1]]+topics[i, maxInd[i, numTop-2]]
    #print maxInd.shape
    print Stats(maxTop)
    print Stats(top2top)

if __name__ == '__main__':
    maxTopicStats('us_politics', 'uspolM11')
