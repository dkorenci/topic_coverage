# -*- coding: utf-8 -*-
from doc_topic_coh.evaluations.topic_coherence import palmettoCoherence
from matplotlib import pyplot as plt
from pytopia.topic_functions.coherence.matrix_norm import MatrixNormCoherence

from pytopia.measure.topic_distance import cosine, l2, jensenShannon
from pytopia.topic_functions.coherence.tfidf_variance import TfidfVarianceCoherence


def plotLabelDistribution(ltopics, coh, measureLabel=None, joinMixed=True):
    '''
    Plot boxplot of coherence values for each topic label / category.
    :param ltopics: list of (topic, label)
    :param coh: pytopia coherence measure
    '''
    fig, ax = plt.subplots()
    labVals = {}
    print 'NUM TOPICS %d'%len(ltopics)
    cnt = 0
    def appJoinMixed(t, l):
        newl = 'theme_mix' if l.startswith('theme_mix') else l
        return t, newl
    if joinMixed: ltopics = [appJoinMixed(t, l) for t, l in ltopics]
    for t, l in ltopics:
        if l not in labVals: labVals[l] = []
        #print topicWords(t, topw)
        labVals[l].append(coh(t))
        cnt+=1
        if cnt % 20 == 0: print '%d processed' % cnt
    labs = labVals.keys(); labs.sort()
    #labs = ['theme', 'theme_noise', 'theme_mix', 'noise']
    labs = ['noise', 'theme_mix', 'theme_noise', 'theme']
    data = [ labVals[l] for l in labs ]
    ttl = coh.id if not measureLabel else measureLabel
    ttl = ttl.upper();
    if ttl == 'C_V': ttl = 'CV'
    ax.boxplot(data)
    lmap = {'noise':u'šum', 'theme':u'tema',
            'theme_mix':u'mješavina tema', 'theme_noise':u'tema i šum'}
    plabs = [lmap[l] for l in labs]
    ax.set_xticklabels(plabs, minor=False)
    plt.ylabel(ttl, fontsize=13)
    #plt.title('%s @ gtar labeled topics' % ttl)
    plt.savefig('%s_labeldist.pdf' % ttl)

from doc_topic_coh.dataset.topic_splits import iter0DevTestSplit
from doc_topic_coh.dataset.topic_labels import labelAllTopics, labelingStandard
def plot():
    from doc_topic_coh.evaluations.scorer_build_data import DocCoherenceScorer
    cacheFolder = '/datafast/doc_topic_coherence/experiments/iter5_coherence/function_cache/'
    params = {
        'npmi': { 'type':'npmi', 'standard': False, 'index': 'wiki_docs', 'windowSize': 10},
        'uci': { 'type':'uci', 'standard': False, 'index': 'wiki_docs', 'windowSize': 10},
        'c_a': { 'type':'c_a', 'standard': False, 'index': 'wiki_docs', 'windowSize': 5},
        'c_v': { 'type':'c_v', 'standard': False, 'index': 'wiki_docs', 'windowSize': 110},
        'c_p': { 'type':'c_p', 'standard': False, 'index': 'wiki_docs', 'windowSize': 70},
    }
    for p in params.itervalues(): p['cache'] = cacheFolder
    allTopics = labelAllTopics(labelingStandard)
    for cohl in ['npmi', 'c_v']:
        cohpar = params[cohl]
        scorer = DocCoherenceScorer(**cohpar)
        coh = scorer()
        plotLabelDistribution(allTopics, coh, cohl)

if __name__ == '__main__':
    plot()