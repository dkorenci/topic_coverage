from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer
from pytopia.measure.topic_distance import cosine
from doc_topic_coh.dataset.topic_splits import allTopics
from doc_topic_coh.evaluations.iteration5.doc_based_coherence import devTestSplit

from pytopia.context.ContextResolver import resolve

cacheFolder = '/datafast/doc_topic_coherence/experiments/iter5_coherence/function_cache'

def topicLabel(t):
    '''
    :param t: (modelId, topicId)
    '''
    return '%s.%s'%t

# requires corpus_topic_index_builder
def printDocumentTitles(topic, topDocs=10, corpus='us_politics'):
    '''
    :param topic: (modelId, topicId)
    :param topDocs:
    :return:
    '''
    mid, tid = topic
    ctiBuilder = resolve('corpus_topic_index_builder')
    cti = ctiBuilder(corpus=corpus, model=mid)
    wtexts = cti.topicTexts(tid, top=topDocs)
    txtIds = [ id_ for id_, _ in wtexts ]
    corpus = resolve(corpus)
    idTexts = corpus.getTexts(txtIds)
    for txto in idTexts:
        print txto.title

def listTopics(coh, topics, topics2print=10, top=True):
    '''
    List coherence scores and labels of top/bottom topics.
    :param coh: coherence measure
    :param topics: list of labeled topics
    :param topics2print: top words for topic label
    :param top: display top or bottom topics
    :return:
    '''
    res = []
    for t, tl in topics:
        res.append((coh(t), t))
    res.sort(key=lambda p:p[0], reverse=top)
    for i in range(topics2print):
        c = res[i][0]
        topic = res[i][1]
        mi, ti = topic
        model = resolve(mi)
        print '%15s: %s , %.4f' % (topicLabel(topic), model.topic2string(ti, 10), c)
        printDocumentTitles(topic)
        print

def contrastCoherences(coh1, coh2, topics, coh1Top=True, coh2Top=False,
                       sort=None, per1=0.9, per2=0.1, topWords=10):
    '''
    Display topics with good rank by one coherence measure and
     bad ranked by another measure.
    :param per1, per2: percentiles that define what is good and bad rank -
        take per1 percentile by coh1 (or above) and bottom per2 percentile by coh2 (or below)
    :param topics: list of labeled topics
    :param topWords: top words for topic label
    :return:
    '''
    from numpy import percentile
    res1 = [coh1(t) for t, tl in topics]
    res2 = [coh2(t) for t, tl in topics]
    perc1 = percentile(res1, per1*100.0)
    perc2 = percentile(res2, per2*100.0)
    print 'coh_top', coh1.id
    print 'coh_bot', coh2.id
    selected = []
    selector = lambda score, perc, above: score >= perc if above else score <= perc
    selector1 = lambda score: selector(score, perc1, coh1Top)
    selector2 = lambda score: selector(score, perc2, coh2Top)
    for i, t in enumerate(topics):
        if selector1(res1[i]) and selector2(res2[i]):
            topic = t[0]
            selected.append(topic)
    if sort:
        if sort == coh1:
            selected = sorted(selected, key=lambda t: coh1(t), reverse=True)
        else:
            selected = sorted(selected, key=lambda t: coh2(t), reverse=True)
    for topic in selected:
        mi, ti = topic
        model = resolve(mi)
        label = unicode(model.description.topic[ti].label).lower().strip()
        print '%15s: %s , [%s]' % (topicLabel(topic), model.topic2string(ti, topWords), label)

def scorer(params, cache=None):
    if cache: params['cache'] = cache
    return DocCoherenceScorer(**params)()

def analyzeBestDoccoh():
    bestGraph1 = {'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'communicability',
                 'vectors': 'probability', 'threshold': 50, 'weightFilter': [0, 0.95], 'type': 'graph'}
    bestGraph2 = {'distance': cosine, 'weighted': False, 'center': 'mean', 'algorithm': 'closeness',
                  'vectors': 'probability', 'threshold': 50, 'weightFilter': [0, 0.95], 'type': 'graph'}
    bestDist = {'distance': cosine, 'center': 'median', 'vectors': 'probability',
                'exp': 1.0, 'threshold': 50, 'type': 'avg-dist'}
    #cohParams = bestDist
    cohParams = bestGraph1
    cohParams['cache'] = cacheFolder
    coh = DocCoherenceScorer(**cohParams)()
    _, topics = devTestSplit()
    listTopics(coh, topics, 50, True)

def analyzeBestWordcoh():
    wordCohBest = {'type':'c_v', 'standard': True, 'index':'wiki_standard'}
    cohParams = wordCohBest
    cohParams['cache'] = cacheFolder
    coh = DocCoherenceScorer(**cohParams)()
    _, topics = devTestSplit()
    listTopics(coh, topics, 10, False)

def documentVsWordCoherence():
    bestGraph1 = { 'distance': cosine, 'weighted': False, 'center': 'mean',
                    'algorithm': 'communicability', 'vectors': 'tf-idf',
                    'threshold': 50, 'weightFilter': [0, 0.92056], 'type': 'graph' }
    cp = { 'type':'c_p', 'standard': False, 'index': 'wiki_docs', 'windowSize': 70}
    cv = { 'type':'c_v', 'standard': False, 'index': 'wiki_docs', 'windowSize': 110}
    npmi = {'type': 'npmi', 'standard': True, 'index': 'wiki_standard'}
    tfidf = {'type': 'tfidf_coherence'}
    cohDoc = scorer(bestGraph1, cacheFolder)
    cohWord = scorer(cp, cacheFolder)
    dev, test = devTestSplit()
    alltop = allTopics()
    topics = test
    # contrastCoherences(cohDoc, cohWord, topics,
    #                    per1=0.7, per2=0.3, coh1Top=True, coh2Top=False, sort=cohWord)
    # print
    contrastCoherences(cohWord, cohDoc, topics,
                       per1=0.7, per2=0.7, coh1Top=True, coh2Top=True, sort=cohWord)
    # print
    # contrastCoherences(cohWord, cohDoc, topics,
    #                    per1=0.7, per2=0.7, coh1Top=True, coh2Top=True, sort=cohWord)
    # contrastCoherences(cohWord, cohDoc, topics,
    #                    per1=0.3, per2=0.3, coh1Top=False, coh2Top=False, sort=cohWord)

def documentVsWordCoherenceCro():
    from doc_topic_coh.resources.croelect_resources.croelect_resources import corpusId, dictId, text2tokensId
    from doc_topic_coh.evaluations.iteration6.croelect_topics import model123Topics
    bestGraph1 = {'distance': cosine, 'weighted': False, 'center': 'mean',
         'algorithm': 'communicability', 'vectors': 'tf-idf',
         'threshold': 50, 'weightFilter': [0, 0.93172], 'type': 'graph',
        'corpus': corpusId, 'text2tokens': text2tokensId, 'dict': dictId}
    cv = {'type':'c_v', 'standard': False, 'index': 'crowiki_palmetto_index', 'windowSize': 110,
          'corpus': corpusId, 'text2tokens': text2tokensId, 'dict': dictId}
    cacheFolder = '/datafast/doc_topic_coherence/experiments/iter6_coherence/function_cache'
    cohDoc = scorer(bestGraph1, cacheFolder)
    cohWord = scorer(cv, cacheFolder)
    topics = model123Topics
    print 'high doc, low word'
    contrastCoherences(cohDoc, cohWord, topics,
                       per1=0.7, per2=0.3, coh1Top=True, coh2Top=False, sort=cohWord)
    print 'high doc, high word'
    contrastCoherences(cohWord, cohDoc, topics,
                       per1=0.7, per2=0.7, coh1Top=True, coh2Top=True, sort=cohWord)
    print 'low doc, high word'
    contrastCoherences(cohWord, cohDoc, topics,
                       per1=0.7, per2=0.3, coh1Top=True, coh2Top=False, sort=cohWord)
    print 'low doc, low word'
    contrastCoherences(cohWord, cohDoc, topics,
                       per1=0.3, per2=0.3, coh1Top=False, coh2Top=False, sort=cohWord)


from pytopia.nlp.text2tokens.gtar.text2tokens import RsssuckerTxt2Tokens
def destemWords(words, top=True, corpusId='us_politics', text2tokens=RsssuckerTxt2Tokens()):
    itb = resolve('inverse_tokenizer_builder')
    itok = itb(corpusId, text2tokens, True)
    if top:
        print ' '.join(itok.allWords(w)[0] for w in words.split())
    else:
        for w in words.split():
            print w, itok.allWords(w)

if __name__ == '__main__':
    #analyzeBestDoccoh()
    #analyzeBestWordcoh()
    #documentVsWordCoherence()
    #destemWords('pledg frank sell item admir model maker unknown navi suddenli')
    #printDocumentTitles(('uspolM1', 49), topDocs=10)
    documentVsWordCoherenceCro()