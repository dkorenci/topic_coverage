from pytopia.topic_functions.coherence.tfidf_variance import TfidfVarianceCoherence

from doc_topic_coh.dataset.topic_splits import iter0DevTestSplit
from pytopia.resource.esa_vectorizer.EsaVectorizer import EsaVectorizer
from pytopia.topic_functions.coherence.doc_matrix_coh_factory import variance_coherence


def compareCoherences(topics, var1, var2, labeled=True):
    '''
    Compare results of two coherence-calculating functions on a set of topics.
    '''
    print 'Comparing %s and %s' % (var1.id, var2.id)
    for t in topics:
        if labeled: t = t[0] # topic is a pair (topic, label)
        v1, v2 = var1(t), var2(t)
        print 'var1 %.4f, var2 %.4f' % (v1, v2)
        assert v1 == v2

def compareOldNewVariance():
    dev, test = iter0DevTestSplit()
    v1 = TfidfVarianceCoherence(0.1)
    v2 = variance_coherence(0.1, 'corpus_tfidf_builder', 'l2')
    compareCoherences(dev, v1, v2)

def compareOldNewAvgDist():
    from pytopia.topic_functions.coherence.doc_matrix_coh_factory import \
        avg_dist_coherence
    from pytopia.topic_functions.coherence.matrix_norm import MatrixNormCoherence
    from pytopia.measure.topic_distance import cosine, l2, l1
    dev, test = iter0DevTestSplit()
    dists = [l1, l2, cosine]
    for d in dists:
        old = MatrixNormCoherence(d)
        new = avg_dist_coherence(threshold=100, mapperCreator='corpus_tfidf_builder',
                                 distance=d.__name__)
        compareCoherences(dev, old, new)

def testEsaCoherence():
    dev, test = iter0DevTestSplit()
    esa = variance_coherence(5, vectorizerCreator=EsaVectorizer)
    print esa.id
    for t in dev:
        t = t[0]
        c = esa(t)
        print '%.4f' % c

if __name__ == '__main__':
    compareOldNewVariance()
    #compareOldNewAvgDist()
    #testEsaCoherence()

