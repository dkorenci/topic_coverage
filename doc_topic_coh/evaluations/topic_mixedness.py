from doc_topic_coh.resources import pytopia_context

from doc_topic_coh.evaluations.tools import topicMeasureAuc
from doc_topic_coh.dataset.topic_splits import iter0DevTestSplit
from doc_topic_coh.dataset.topic_labels import labelAllTopics, labelingStandard
from doc_topic_coh.mixedness.factory import *

from sklearn.metrics.pairwise import chi2_kernel

def calculateAucs():
    dev, test = iter0DevTestSplit()
    allTopic = labelAllTopics(labelingStandard)
    ltopics = dev
    #mm = kmeansSilhouette()
    #mm = spectralSilhouette()
    mm = spectralSilhouette(affinity=chi2_kernel, kernel_params={'gamma':1.0})
    print topicMeasureAuc(mm, ltopics, ['theme_mix', 'theme_mix_noise'])

def testSpectral():
    mml = [
        spectralSilhouette('cosine'),
        spectralSilhouette('rbf'),
        spectralSilhouette('sigmoid'),
        spectralSilhouette('polynomial'),
        spectralSilhouette('nearest_neighbors', 5),
        spectralSilhouette('nearest_neighbors', 10),
        spectralSilhouette('nearest_neighbors', 20)
    ]
    dev, test = iter0DevTestSplit()
    lset = [['theme_mix', 'theme_mix_noise']]
    for l in lset:
        runMixednessAuc(mml, dev, l)

def testKmeans():
    mml = [
        kmeansSilhouette('k-means++'),
        kmeansSilhouette('random')
    ]
    dev, test = iter0DevTestSplit()
    lset = [['theme_mix'], ['theme_mix', 'theme_mix_noise']]
    for l in lset:
        runMixednessAuc(mml, dev, l)

def testHAC():
    mml = [
        hacSilhouette('average', 'euclidean'),
        hacSilhouette('complete', 'euclidean'),
        hacSilhouette('ward', 'euclidean'),
        hacSilhouette('average', 'cosine'),
        hacSilhouette('complete', 'cosine'),
        hacSilhouette('average', 'l1'),
        hacSilhouette('complete', 'l1'),
    ]
    dev, test = iter0DevTestSplit()
    lset = [['theme_mix'], ['theme_mix', 'theme_mix_noise']]
    for l in lset:
        runMixednessAuc(mml, dev, l)

def runMixednessAuc(mml, ltopics, labels):
    '''
    :param mml: list of mixedness measures
    :param ltopics: list of labeled topics
    :param labels: labels determining the positive class
    :return:
    '''
    print labels
    for mm in mml:
        res = topicMeasureAuc(mm, ltopics, labels)
        print '%s : %.4f' % (mm.id, res)

from pytopia.topic_functions.mixedness.sklearn_clusterer_wrapper import *
def testMixednessTES():
    '''
    Test mixedness scoring using scorers composed as TopicElementScorers
    '''
    from pytopia.topic_functions.mixedness.tes_mixedness_factory import mixedness
    from sklearn.metrics.cluster import silhouette_score
    dev, test = iter0DevTestSplit()
    labels = ['theme_mix', 'theme_mix_noise']
    n_jobs = 3
    mmspectral = [
        mixedness(100, clusterer=spectral(affinity='cosine'),
                  score=silhouette_score, n_jobs=n_jobs, seed=566),
        mixedness(100, clusterer=spectral(affinity='rbf'),
                  score=silhouette_score, n_jobs=n_jobs, seed=566),
        mixedness(100, clusterer=spectral(affinity='sigmoid'),
                  score=silhouette_score, n_jobs=n_jobs, seed=566),
        mixedness(100, clusterer=spectral(affinity='polynomial'),
                  score=silhouette_score, n_jobs=n_jobs, seed=566),
        mixedness(100, clusterer=spectral(affinity='nearest_neighbors', n_neighbours=5),
                  score=silhouette_score, n_jobs=n_jobs, seed=566),
        mixedness(100, clusterer=spectral(affinity='nearest_neighbors', n_neighbours=10),
                  score=silhouette_score, n_jobs=n_jobs, seed=566),
        mixedness(100, clusterer=spectral(affinity='nearest_neighbors', n_neighbours=20),
                  score=silhouette_score, n_jobs=n_jobs, seed=566),
    ]
    ltopics = dev
    mml = mmspectral
    for mm in mml:
        res = topicMeasureAuc(mm, ltopics, labels)
        print mm.id
        print '%.4f' % res

def testWordMixedness():
    '''
    Test mixedness based on word clustering.
    '''
    from pytopia.topic_functions.mixedness.tes_mixedness_factory import mixedness
    from sklearn.metrics.cluster import silhouette_score
    dev, test = iter0DevTestSplit()
    labels = ['theme_mix', 'theme_mix_noise']
    n_jobs = 3
    from doc_topic_coh.factory.mixedness import uspolInvTokenWord2Vec
    uspolW2V = uspolInvTokenWord2Vec()
    numw = 10
    mmspectral = [
        mixedness(numw, selected='words', mapper = uspolW2V, mapperIsFactory=False,
                  clusterer=spectral(affinity='cosine'),
                  score=silhouette_score, n_jobs=n_jobs, seed=566, timer=False),
        mixedness(numw, selected='words', mapper=uspolW2V, mapperIsFactory=False,
                  clusterer=spectral(affinity='sigmoid'),
                  score=silhouette_score, n_jobs=n_jobs, seed=566, timer=False),
        mixedness(numw, selected='words', mapper=uspolW2V, mapperIsFactory=False,
                  clusterer=spectral(affinity='rbf'),
                  score=silhouette_score, n_jobs=n_jobs, seed=566, timer=False),
        mixedness(numw, selected='words', mapper=uspolW2V, mapperIsFactory=False,
                  clusterer=spectral(affinity='polynomial'),
                  score=silhouette_score, n_jobs=n_jobs, seed=566, timer=False),
    ]
    ltopics = dev
    mml = mmspectral
    for mm in mml:
        print mm.id
        res = topicMeasureAuc(mm, ltopics, labels, tolerant=True)
        print '%.4f' % res

if __name__ == '__main__':
    #calculateAucs()
    #testSpectral()
    #testKmeans()
    #testHAC()
    #testMixednessTES()
    testWordMixedness()