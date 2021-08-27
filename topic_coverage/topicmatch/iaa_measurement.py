from topic_coverage.topicmatch.pair_labeling import parseLabelingFile

def __initR():
    import rpy2.robjects as ro
    R = ro.r
    R('require(irr)')
    R('options(warn=-1)')
    return R

def krippendorfAlpha(l1, l2):
    '''
    Calculate Krippendorf's alpha IAA measure for two labelings of the same set of topics
    :param l1, l2: lists of (pairId, topic1, topic2, label)
    :return: krippendorf alpha
    '''
    r = __initR()
    def idset(labeling): return set(l[0] for l in labeling)
    assert idset(l1) == idset(l2)
    # align labeles by pair id
    lmap = {pid:[l, None] for pid, _, _, l in l1}
    for pid, _, _, l in l2: lmap[pid][1] = l
    # create R matrix
    r1 = ','.join(('%d' % l[0]) for l in lmap.itervalues())
    r2 = ','.join(('%d' % l[1]) for l in lmap.itervalues())
    matrixCode = 'matrix(c(%s,%s), nrow=2, byrow=TRUE)' % (r1, r2)
    # call kripp.alpha with the matrix, extract result
    matrix = r(matrixCode)
    kripp = r['kripp.alpha']
    result = kripp(matrix, 'nominal')
    return result[4][0]


def iaaIter1():
    annot1 = '/home/damir/Dropbox/projekti/doktorat/D1 eksplorativa/iteracija3/mjerenje istosti tema/iaa4real_topicPairs[100-150]_damir.txt'
    annot2 = '/home/damir/Dropbox/projekti/doktorat/D1 eksplorativa/iteracija3/mjerenje istosti tema/iaa4real_topicPairs[100-150]_ristov.txt'
    l1 = parseLabelingFile(annot1)
    l2 = parseLabelingFile(annot2)
    print '%g' % krippendorfAlpha(l1, l2)

if __name__ == '__main__':
    iaaIter1()