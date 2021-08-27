from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer, scorersFromParams
from doc_topic_coh.evaluations.tools import topicMeasureAuc, \
    flattenParams as fp, joinParams as jp
from doc_topic_coh.dataset.topic_splits import iter0DevTestSplit
from pytopia.measure.topic_distance import cosine, l1, l2
from doc_topic_coh.evaluations.experiment import IdDict, IdList, \
    TopicScoringExperiment as TSE

def paramsBestAndBaselines(select=['corpus_vectors', 'word2vec_vectors']):
    ############# baselines
    if 'word_based' in select:
        #todo update with additional paramters for new palmetto coherence class
        blinePalmetto = { 'type': ['npmi', 'uci', 'umass', 'c_a', 'c_p', 'c_v', ] }
        blineOther = { 'type': ['text_distribution', 'pairwise_word2vec', 'tfidf_coherence'] }
        bline = fp(blinePalmetto)+fp(blineOther)
    else: bline = []
    ############# best algorithms
    if 'corpus_vectors' in select:
        ## corpus vectors
        bestMatrixC = [
            { 'type': 'avg-dist', 'distance': cosine, 'center': 'median',
              'vectors': 'probability', 'exp': 1.0, 'threshold': 100, },
            {'type': 'variance', 'distance': l2, 'center': 'mean',
             'vectors': 'probability', 'exp': 2.0, 'threshold': 100, }
                     ]
        bestGraphC = [
            {'type': 'graph', 'distance':cosine, 'weighted': False, 'center': 'median',
             'algorithm': 'closeness', 'vectors': 'probability', 'threshold': 100,
             'weightFilter': [0, 0.9]},
            {'type': 'graph', 'distance': cosine, 'weighted': False, 'center': 'median',
             'algorithm': 'closeness', 'vectors': 'tf-idf', 'threshold': 100,
             'weightFilter': [0, 0.9]},
            { 'type': 'graph', 'distance': cosine, 'weighted': True, 'center': 'median',
             'algorithm': 'closeness', 'vectors': 'probability', 'threshold': 100,
             'weightFilter': [0, 0.9]}
        ]
        bestDensityC = [
            {'type': 'density', 'scoreMeasure': 'aic', 'vectors': 'probability', 'covariance': 'spherical',
             'dimReduce': 20, 'threshold': 100 },
            {'type': 'density', 'scoreMeasure': 'aic', 'vectors': 'probability', 'covariance': 'spherical',
             'dimReduce': 50, 'threshold': 100 },
            {'type': 'density', 'scoreMeasure': 'aic', 'vectors': 'tf-idf', 'covariance': 'spherical',
             'dimReduce': 50, 'threshold': 100 }
        ]
        bestCorpus = bestMatrixC+bestGraphC+bestDensityC
    else: bestCorpus = []

    ## word2vec vectors
    if 'word2vec_vectors':
        bestMatrixW2V = [
            {'type': 'avg-dist', 'distance': cosine, 'center': 'median', 'vectors': 'word2vec', 'exp': 1.0,
             'threshold': 100},
            { 'type': 'avg-dist', 'distance': cosine, 'center': 'mean', 'vectors': 'word2vec', 'exp': 1.0,
             'threshold': 100}
        ]
        bestGraphW2V = [
            {'type': 'graph', 'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'closeness',
             'vectors': 'word2vec', 'threshold': 100, 'weightFilter': [0, 0.5]},
            {'type': 'graph', 'distance': cosine, 'weighted': True, 'center': 'mean', 'algorithm': 'closeness',
             'vectors': 'word2vec', 'threshold': 100, 'weightFilter': [0, 0.9]}
        ]
        bestDensWord2V = [
            {'type': 'density', 'scoreMeasure': 'aic', 'vectors': 'word2vec',
             'covariance': 'diag', 'dimReduce': 10, 'threshold': 50},
            {'type': 'density', 'scoreMeasure': 'aic', 'vectors': 'word2vec',
             'covariance': 'diag', 'dimReduce': 20, 'threshold': 50}
        ]
        bestWord2Vec = bestMatrixW2V+bestGraphW2V+bestDensWord2V
    else: bestWord2Vec = []
    bestAlgo = bestCorpus + bestWord2Vec
    p = IdList(bestAlgo + bline)
    p.id = 'best_algos_and_baselines'
    return p

dev, test = iter0DevTestSplit()
expFolder = '/datafast/doc_topic_coherence/experiments/iter1_coherence/best_word2vec_nostem'

def experimentAllTest():
    tse = TSE(paramSet=paramsBestAndBaselines(['word2vec_vectors']), scorerBuilder=DocCoherenceScorer,
              ltopics=test, posClass=['theme', 'theme_noise'], folder=expFolder,
              cache=True)
    tse.run()
    tse.printResults()

if __name__ == '__main__':
    experimentAllTest()
