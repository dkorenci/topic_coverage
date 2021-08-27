import codecs
from math import log

from pytopia.utils.print_ import topTopicWords

def cluster2str(cluster):
    '''
    Create string representation of a cluster.
    :param cluster: list of (modelId, topicId) pairs
    '''
    pass

def variationOfInformation(clust1, clust2):
    '''
    Obsolete
    TODO: rewrite
    Calculates variation of information distance metric between two clusterings.
    Clusterings are lists of clusters - lists containing elements.
    '''
    if set(e for cl in clust2 for e in cl) != set(e for cl in clust1 for e in cl) :
        raise Exception('clusters do not contain same elements')
    N = (float)(sum(1 for cl in clust1 for e in cl))
    print N
    sets1, sets2 = [set(cl) for cl in clust1], [set(cl) for cl in clust2]
    vi = 0.0
    for cl1 in clust1:
        s1 = set(cl1); assert len(cl1) == len(s1)
        assert len(cl1)
        for cl2 in clust2:
            s2 = set(cl2); assert len(cl2) == len(s2)
            assert len(cl2)
            r = len(s1.intersection(s2))/N
            p, q = len(s1)/N, len(s2)/N
            if r != 0: vi += r*(log(r/p)+log(r/q))
    vi = -vi
    return vi

def clusterInfo(cl, mmap, topW=20, label=None):
    '''
    Obsolete
    TODO: rewrite
    Generate printable (string) cluster info for a cluster of topics.
    :param cl: cluster, list of (modelId, topicId)
    :param mmap: map modelId -> TopicModel
    :param topW: number of top topic words to represent a topic
    '''
    dict = None; avg = None
    for mi, ti in cl:
        if avg is None: avg = mmap[mi].topicVector(ti).copy()
        else: avg += mmap[mi].topicVector(ti)
    # get dictionary from models (all models must share same dictionary)
    for mi, _ in cl:
        dict = mmap[mi].dictionary()
        break
    res = u'%s ; ' % label if label else u'';
    res += (u'size %d \n' % len(cl))
    if len(cl) > 1:
        res += (u'aggreg:   %s \n' % topTopicWords(avg, dict, topW))
    for i, (mi, ti) in enumerate(cl):
        topic = mmap[mi].topicVector(ti)
        res += (u'   topic: %s%s' %
                (topTopicWords(topic, dict, 20), '\n' if i < len(cl)-1 else ''))
    return res

def loadTopics(topics, models=False, checkDuplicates=False):
    '''
    Create a list of topics from Topics and TopicModels.
    :param topics: list of either Topics, or TopicModels (all topics are loaded)
            or folders containing saved topic models.
    :param checkDuplicates: if True, topic list will contain no duplicates
    :return: list of topics or if models=True, list of topics and a list of models
    '''
    from pytopia.topic_model.TopicModel import isTopicModel
    from pytopia.resource.loadSave import loadResource
    def isFile(o): return isinstance(o, basestring)
    topicList = []
    for o in topics:
        # todo enable o to also be an id
        if isFile(o) or isTopicModel(o):
            if isFile(o): m = loadResource(o)
            else: m = o
            for tid in m.topicIds(): topicList.append(m.topic(tid))
        else: topicList.append(o)
    if checkDuplicates:
        uniq = set()
        dedup = []
        for t in topicList:
            if t.id not in uniq:
                uniq.add(t.id)
                dedup.append(t)
        topicList = dedup
    if models: modelList = list(set(t.model for t in topicList))
    else: modelList = None
    if modelList is not None: return topicList, modelList
    else: return topicList

