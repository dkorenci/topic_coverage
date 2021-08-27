from pytopia.topic_functions.coherence.adapt import WordsStringCohAdapter
from pytopia.topic_functions.coherence.document_distribution import DocuDistCoherence
from pytopia.topic_functions.coherence.gaussian_density import GaussCoherence
from pytopia.topic_functions.coherence.graph_density import GraphCCCoherence
from pytopia.topic_functions.coherence.kernel_density import KernelDensityCoherence
from pytopia.topic_functions.coherence.matrix_norm import MatrixNormCoherence
from pytopia.topic_functions.coherence.tfidf_coherence import TfidfCoherence
from pytopia.topic_functions.coherence.tfidf_variance import TfidfVarianceCoherence

from coverexp.coherence.factory import gtarCoherence
from doc_topic_coh.evaluations.tools import topicMeasureAuc
from pytopia.measure.topic_distance import cosine, l1, l2, jensenShannon, kullbackLeibler
from pytopia.resource.esa_vectorizer.EsaVectorizer import EsaVectorizer
from pytopia.topic_functions.coherence.doc_matrix_coh_factory import \
    variance_coherence, graph_coherence
from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer

def palmettoCoherence(measure, topW):
    return WordsStringCohAdapter(coh=gtarCoherence(measure), topW=topW,
                                 id='%s[%d]'%(measure, topW))

from doc_topic_coh.dataset.topic_splits import iter0DevTestSplit, devTestSplit2
from doc_topic_coh.dataset.topic_labels import labelAllTopics, labelingStandard
def calculateAucs():
    dev, test = iter0DevTestSplit()
    allTopic = labelAllTopics(labelingStandard)
    # ['umass', 'uci', 'npmi', 'c_a', 'c_p', 'c_v']
    # theme, theme_noise, theme_mix, theme_mix_noise, noise
    npmi = palmettoCoherence('npmi', 10)
    uci = palmettoCoherence('uci', 10)
    umass = palmettoCoherence('umass', 10)
    c_a = palmettoCoherence('c_a', 10)
    c_p = palmettoCoherence('c_p', 10)
    c_v = palmettoCoherence('c_v', 10)
    var = TfidfVarianceCoherence()
    cosDist = MatrixNormCoherence(cosine)
    l2Dist = MatrixNormCoherence(l2)
    jsDist = MatrixNormCoherence(jensenShannon)
    #coherences = [cosDist, l2Dist, var, umass, uci, npmi]
    ddCos = DocuDistCoherence(cosine)
    ddKL = DocuDistCoherence(kullbackLeibler)
    gauss = GaussCoherence(score='aic')
    graph = GraphCCCoherence(threshold=0.5)
    kernel = KernelDensityCoherence()
    tfidf = TfidfCoherence(10)
    esa = varcoh(threshold=5, mapperCreator=EsaVectorizer, distance='l2', timer=True)
    avgMedian = avgcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='l2', center='median', exp=2.0)
    ltopics = dev
    # [npmi, uci, umass, c_a, c_p, c_v]
    # [ddCos, ddKL, avgMedian, tfidf]
    coherences = [npmi]
    for coh in coherences:
        print topicMeasureAuc(coh, ltopics, ['theme', 'theme_noise'])

from pytopia.topic_functions.coherence.doc_matrix_coh_factory import \
    avg_dist_coherence as avgcoh, variance_coherence as varcoh, distance_or_matrix_coherence
from doc_topic_coh.factory.coherences import pairwiseWord2Vec
def calculateAucsNew():
    ''' Work with new, modular, coherence calculators. '''
    dev, test = iter0DevTestSplit()
    allTopic = labelAllTopics(labelingStandard)
    coherences1 = [
        varcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='l2'),
        varcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='l1'),
        varcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='cosine'),
        avgcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='l2'),
        avgcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='l1'),
        avgcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='cosine'),
    ]
    coherences2 = [
        varcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='l2', center='median'),
        #varcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='l2', exp=2),
        varcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='cosine', center='median'),
        avgcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='l2', center='median'),
        avgcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='cosine', center='median'),
    ]
    from doc_topic_coh.factory.coherences import uspolProbTextVectorsStatic
    uspolProbVec = uspolProbTextVectorsStatic()
    coherences2Prob = [
        varcoh(threshold=100, mapperCreator=uspolProbVec,
               distance='l2', center='median', factory=False),
        #varcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='l2', exp=2),
        varcoh(threshold=100, mapperCreator=uspolProbVec, distance='cosine',
               center='median', factory=False),
        avgcoh(threshold=100, mapperCreator=uspolProbVec, distance='l2',
               center='median', factory=False),
        avgcoh(threshold=100, mapperCreator=uspolProbVec, distance='cosine',
               center='median', factory=False),
    ]
    coherences3 = [
        avgcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='l2',
               center='median', exp=2.0),
    ]
    graphCoh1 = [
        graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
                        distance='cosine', algorithm='closeness', center='mean'),
        graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
                        distance='cosine', algorithm='closeness', center='median'),
        graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
                        distance='l2', algorithm='closeness', center='median'),
        graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
                        distance='l2', algorithm='closeness', center='median'),
    ]
    graphCohBest = [
        graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
                        distance='cosine', algorithm='closeness', center='median'),
    ]
    # cohWord2Vec = [ uspolPairwiseWord2Vec(cosine, 5), uspolPairwiseWord2Vec(cosine, 10),
    #                 uspolPairwiseWord2Vec(cosine, 15), uspolPairwiseWord2Vec(cosine, 20),
    #                 uspolPairwiseWord2Vec(cosine, 50)]
    #from doc_topic_coh.factory.coherences import uspolWord2VecInvTokensStatic
    uspolW2V = None #uspolWord2VecInvTokensStatic()
    # cohTextWord2Vec = [
    #     varcoh(threshold=100, mapperCreator=uspolW2V, distance='cosine', factory=False),
    #     avgcoh(threshold=100, mapperCreator=uspolW2V, distance='cosine', factory=False),
    # ]
    coherences3 = [
        varcoh(threshold='above-random', mapperCreator='corpus_tfidf_builder', distance='l2', center='median'),
        #varcoh(threshold=100, mapperCreator='corpus_tfidf_builder', distance='l2', exp=2),
        varcoh(threshold='above-random', mapperCreator='corpus_tfidf_builder', distance='cosine', center='median'),
        avgcoh(threshold='above-random', mapperCreator='corpus_tfidf_builder', distance='l2', center='median'),
        avgcoh(threshold='above-random', mapperCreator='corpus_tfidf_builder', distance='cosine', center='median'),
    ]
    coherences = coherences1 #coherences2+coherences3
    ltopics = dev
    for coh in coherences:
        print coh.id
        print topicMeasureAuc(coh, ltopics, ['theme', 'theme_noise'])

def calculateAucsGraph():
    #dev, test = iter0DevTestSplit()
    dev, test = devTestSplit2()
    allTopic = labelAllTopics(labelingStandard)
    center = 'mean'
    coh1 = [
        # graph_coherence(threshold=100, mapperCreator='corpus_tfidf_builder',
        #                 distance='cosine', algorithm='closeness', center='median'),
        graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
                        distance='cosine', algorithm='closeness', center=center,
                        weightFilter=[0.0, 0.9], weighted=False),
        graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
                        distance='cosine', algorithm='closeness', center=center,
                        weightFilter=[0.0, 0.9], weighted=True),
        # graph_coherence(threshold=100, mapperCreator='corpus_tfidf_builder',
        #                 distance='cosine', algorithm='closeness', center='median',
        #                 weighted=False),
        graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
                        distance='cosine', algorithm='clustering', center=center,
                        weightFilter=[0.0, 0.9], weighted=False),
        graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
                        distance='cosine', algorithm='clustering', center=center,
                        weightFilter=[0.0, 0.9], weighted=True),
    ]
    coh2 = [
        graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
                        distance='cosine', algorithm='num_connected', center=center,
                        weightFilter=[0.0, 0.75], weighted=False),
        graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
                        distance='cosine', algorithm='num_connected', center=center,
                        weightFilter=[0.0, 0.8], weighted=False),
        graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
                        distance='cosine', algorithm='num_connected', center=center,
                        weightFilter=[0.0, 0.85], weighted=False),
    ]
    comm = [
        graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
                        distance=cosine, algorithm='communicability', center='median',
                        weighted=False, weightFilter=[0,0.95]),
        graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
                        distance=cosine, algorithm='communicability', center='median',
                        weighted=False, weightFilter=[0, 0.9]),
        # graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
        #                 distance=cosine, algorithm='communicability', center='median',
        #                 weighted=False, weightFilter=[0, 0.8]),
        # graph_coherence(threshold=100, mapper='corpus_tfidf_builder',
        #                 distance=cosine, algorithm='communicability', center='median',
        #                 weighted=False, weightFilter=[0, 0.7]),
    ]
    mstCohP = [
        {'distance': l2, 'weighted': True, 'center': 'mean', 'algorithm': 'mst',
         'vectors': 'probability', 'threshold': 100, 'weightFilter': None, 'type': 'graph'},
        {'distance': l1, 'weighted': True, 'center': 'mean', 'algorithm': 'mst',
         'vectors': 'probability', 'threshold': 100, 'weightFilter': None, 'type': 'graph'},
        {'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'mst',
         'vectors': 'probability', 'threshold': 100, 'weightFilter': None, 'type': 'graph'},
        {'distance': l2, 'weighted': True, 'center': 'mean', 'algorithm': 'mst',
         'vectors': 'word2vec', 'threshold': 50, 'weightFilter': None, 'type': 'graph'},
        {'distance': l1, 'weighted': True, 'center': 'mean', 'algorithm': 'mst',
         'vectors': 'word2vec', 'threshold': 50, 'weightFilter': None, 'type': 'graph'},
        {'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'mst',
         'vectors': 'word2vec', 'threshold': 50, 'weightFilter': None, 'type': 'graph'},
    ]
    mstCoh = [DocCoherenceScorer(**p)() for p in mstCohP ]
    coherences = mstCoh
    ltopics = test
    for coh in coherences:
        print coh.id
        print topicMeasureAuc(coh, ltopics, ['theme', 'theme_noise'])

def calculateAucsSvd():
    dev, test = iter0DevTestSplit()
    svdCoh = [
        distance_or_matrix_coherence(type='matrix', threshold=100, mapper='corpus_tfidf_builder',
                                     distance=None, method='mu'),
        distance_or_matrix_coherence(type='matrix', threshold=100, mapper='corpus_tfidf_builder',
                                     distance=None, method='mu0'),
        distance_or_matrix_coherence(type='matrix', threshold=100, mapper='corpus_tfidf_builder',
                                     distance=None, method='mu1'),
    ]
    coherences = svdCoh
    ltopics = test
    for coh in coherences:
        print coh.id
        print topicMeasureAuc(coh, ltopics, ['theme', 'theme_noise'])

def calculateAucsWordVectors(vectors='word2vec', tfidf=False):
    dev, test = devTestSplit2()
    bestDistW2V = [
        {'distance': cosine, 'center': 'mean', 'vectors': vectors, 'tfidf': tfidf,
         'exp': 1.0, 'threshold': 50, 'type': 'avg-dist'},
        {'distance': cosine, 'center': 'median', 'vectors': vectors, 'tfidf': tfidf,
         'exp': 1.0, 'threshold': 50, 'type': 'variance'}
    ]
    bestGraphW2V = [
        {'distance': cosine, 'weighted': True, 'center': 'mean', 'tfidf': tfidf,
         'algorithm': 'closeness', 'vectors': vectors,
         'threshold': 50, 'weightFilter': [0, 0.9], 'type': 'graph'},
        {'distance': cosine, 'weighted': True, 'center': 'mean', 'tfidf': tfidf,
         'algorithm': 'closeness', 'vectors': vectors, 'threshold': 50,
         'weightFilter': None, 'type': 'graph'}
    ]
    params = bestGraphW2V+bestDistW2V
    cohs = [ DocCoherenceScorer(**p)() for p in params ]
    ltopics = test
    for coh in cohs:
        print coh.id
        print topicMeasureAuc(coh, ltopics, ['theme', 'theme_noise'])

if __name__ == '__main__':
    calculateAucsGraph()
    #calculateAucsNew()
    #calculateAucs()
    #calculateAucsSvd()
    #calculateAucsWordVectors('glove', tfidf=False)
    #calculateAucsWordVectors('word2vec', tfidf=False)

