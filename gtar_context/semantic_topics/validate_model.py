from pytopia.context.GlobalContext import GlobalContext
from pytopia.context.ContextResolver import resolve

from gtar_context.compose_context import gtarContext
GlobalContext.set(gtarContext())

from gtar_context.semantic_topics.construct_model import MODEL_ID
from gtar_context.semantic_topics.parse_themes import parseAllThemes

import numpy as np

def createTopicId2Vector():
    from gtar_context.orig_models.orig_model_context import modelId2Folder
    rows = 0
    for mid in modelId2Folder.keys():
        m = resolve(mid)
        rows += len(m.topicIds())
    topicIds = []
    matrix = None; rc = 0
    for mid in modelId2Folder.keys():
        m = resolve(mid)
        for tid in m.topicIds():
            t = '%s.%s' % (mid, tid)
            topicIds.append(t)
            vec = m.topicVector(tid)
            if matrix is None:
                matrix = np.zeros((rows, len(vec)), np.float64)
            matrix[rc] = vec
            rc += 1
    return topicIds, matrix

def checkOrigTopics(modelId, inTop=5):
    '''
    Check that topic of the theme model are close to the original topics
     (orig. gtar model topic the theme is based on)
    Orig. topic must be among inTop closest of all original topics, by cosine distance.
    '''
    from scipy.spatial.distance import cdist
    from numpy import argsort
    pthemes = parseAllThemes(); print 'parsed'
    topIds, matrix = createTopicId2Vector()
    thmodel = resolve(modelId)
    topLabelMod = lambda l: l[1:] if l.startswith('~') else l
    mm = 0
    for i, pt in enumerate(pthemes):
        topics = [topLabelMod(t) for t in pt.topics]
        thVec = thmodel.topicVector(i)
        thMatrix = np.array([thVec])
        dists = cdist(thMatrix, matrix, 'cosine')[0]
        sortInd = argsort(dists)
        closest = [ topIds[j] for j in sortInd[:inTop] ]
        if not set(closest).issuperset(topics):
            mm += 1
            print 'MISMATCH'
            print pt.label
            print thmodel.topic2string(i)
            print topics
            print closest
    print 'NUM. MISMATCHES: %d' % mm

if __name__ == '__main__':
    checkOrigTopics(MODEL_ID, inTop=5)
    #createTopicId2Vector()