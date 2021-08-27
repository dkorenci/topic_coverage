from pytopia.context.ContextResolver import resolve

def corpusTopicWeights(t):
    '''
    Create vector of topic proportions in corpus texts.
    Corpus is the corpus used to build the topic's model.
    :param t: Topic-like object
    '''
    model = t.model
    corpus = resolve(model).corpus
    cti = resolve('corpus_topic_index_builder')(corpus, model)
    tmx = cti.topicMatrix()
    return tmx[:, t.topicId]