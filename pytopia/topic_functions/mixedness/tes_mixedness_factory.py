'''
Factory of topic mixedness calculating functions based on TopicElementScorer abstraction.
'''

from pytopia.topic_functions.mixedness.clustering_scorer import ClusteringMixednessScorer
from pytopia.resource.corpus_tfidf.CorpusTfidfIndex import CorpusTfidfBuilder
from pytopia.topic_functions.document_selectors import TopDocSelector
from pytopia.topic_functions.word_selector import TopWordSelector
from pytopia.topic_functions.topic_elements_score import TopicElementsScore
from pytopia.measure.topic_distance import cosine, l1, l2


def mixedness(threshold, clusterer, score, mapper=None,
              mapperIsFactory=True, average=1, n_jobs=3, seed=478,
              selected='docs', timer=False):
    '''
    :param threshold: init param for TopDocSelector or TopWordSelector
    :param selected: 'docs' or 'words'
    :param clusterer: wrapped sklearn clusterer
    :param score: function accepting a matrix and clustering labels
    :param mapper: see TopicElementsScore
    :param mapperIsFactory: see TopicElementsScore
    :param average: see ClusteringMixednessScorer
    :param n_jobs: see ClusteringMixednessScorer
    :param seed: see ClusteringMixednessScorer
    :return:
    '''
    if selected == 'docs': select = TopDocSelector(threshold)
    elif selected == 'words': select = TopWordSelector(threshold)
    if mapper is None:
        mapper = CorpusTfidfBuilder()
        mapperIsFactory = True
    scorer = ClusteringMixednessScorer(clusterer, score, average=average,
                                       n_jobs=n_jobs, randomSeed=seed)
    tes = TopicElementsScore(selector=select, mapper=mapper, score=scorer,
                             mapperIsFactory=mapperIsFactory, timer=timer)
    return tes

