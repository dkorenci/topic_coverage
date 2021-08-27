import codecs, random
from pyutils.file_utils.location import FolderLocation as loc
from pytopia.context.ContextResolver import resolve

def createLabelingFiles(outfolder, sampleId, pairsFile, intervals, sizes,
                        rndseed=426, filesize=-1, docs=False):
    '''
    Create a set of text files with topic pairs, for labeling of topic match (sameness).
    Sample pairs from the base set based on sampling params, shuffle them
     and write into one or more labeling files.
    :param outfolder: folder to output text files to
    :param sampleId: string id describing the sampling performed
    :param pairsFile: file containing pickled base set of all pairs
    :param intervals: distance intervals from which to sample pairs
    :param sizes: sizes of per-interval samples
    :param rndseed: seed for interval sampling and shuffling
    :param filesize: split output into files with this number of pairs each, -1 for single file
    :param docs: weather to include documents in topic text representation
    :return:
    '''
    from topic_coverage.topicmatch.distance_sampling import loadDistances, \
        samplePairsByDistInterval
    # id of the generated labeling sample
    sampleId = 'labelingSample_[topicPairs=%s]_[sampling=%s]' % (pairsFile, sampleId)
    print 'Creating labeling files from sample: %s' % sampleId
    tpairs = loadDistances(pairsFile)
    print 'number of all pairs: %d' % len(tpairs)
    sample = samplePairsByDistInterval(tpairs, intervals, sizes, rndseed)
    # create shuffled list of topic pairs from sample
    samplePairs = []
    for ival in sample: samplePairs.extend(sample[ival])
    random.seed(rndseed)
    random.shuffle(samplePairs)
    print 'sample size: %d' % len(samplePairs)
    # divide into per-file chunks and create files
    start = 0; filesize = len(samplePairs) if filesize == -1 else filesize
    outfolder = loc(outfolder)
    while start < len(samplePairs):
        fileSample = samplePairs[start: start+filesize]
        fileName = outfolder('topicPairs[%d-%d]'%(start, start+filesize))
        createLabelingFile(fileName, fileSample, sampleId, docs)
        start += filesize
    return samplePairs

def createLabelingFile(fname, topicPairs, sampleId, docs=False):
    f = codecs.open(fname+'.txt', 'w', 'utf-8')
    # write header
    f.write('SAMPLE_ID: %s \n\n' % sampleId)
    pairdelim = u'------------------------------------------\n'  # delimiter
    for tp in topicPairs:
        f.write(pairdelim)
        f.write(pairLabelingText(tp, docs))
        f.write('\n')
    f.close()

matchLabel = u'MATCH:'
pairIdLabel = u'PAIR_ID:'
topicIdLabel = u'TID:'
topicIdDelim = u'$$$'
def pairLabelingText(tpair, docs=False):
    '''Create text for labeling of a single pair
    :param tpair: (pairId, topic1, topic2, ...)
    :param docs: if true include top topic documents in the pair description
    '''
    pid, t1, t2, _ = tpair
    t = matchLabel + u' \n\n' # line for writing the label
    # readable topic representations
    pid, t1, t2 = tpair[0], tpair[1], tpair[2]
    t += u'%s\n\n' % topicLabelText(t1, docs)
    t += u'%s\n\n' % topicLabelText(t2, docs)
    # pair, model and topic indexes, for identification
    t += pairIdLabel + ' %d\n' % pid
    def topicIdStr(t): return u'%s %s %s' % (str(t[0]), topicIdDelim, str(t[1]))
    t += topicIdLabel + ' %s\n' % topicIdStr(t1)
    t += topicIdLabel + ' %s\n' % topicIdStr(t2)
    return t

# requires 'corpus_topic_index_builder'
def topicLabelText(topic, docs=False):
    '''
    Topic text representation for human labelers to assess sameness of two topics.
    :param topic: (modelId, topicId)
    :param docs: if true include top topic documents in the pair description
    :return:
    '''
    txt = 'WORDS: %s' % topicWords(topic)
    if docs:
        txt += '\n'+topicDocs(topic)
    return txt

def topicWords(topic):
    '''
    String of top topic words
    :param topic: (modelId, topicId)
    '''
    mid, tid = topic
    model = resolve(mid)
    return model.topic2string(tid, 15)

def topicDocs(topic):
    '''
    String representing top topic documents.
    :param topic: (modelId, topicId)
    TODO solve fetching and representing phenotype corpora texts
    '''
    mid, tid = topic
    model = resolve(mid)
    cti = resolve('corpus_topic_index_builder')(corpus=model.corpus, model=model)
    ids = [id for id, w in cti.topicTexts(tid, top=15) if w != 0]
    corpus = resolve(model.corpus)
    txt = '\n'.join(textLabel(txto) for txto in corpus.getTexts(ids))
    return txt

def textLabel(txto):
    '''
    Create a short label (title, start of text, ...) for a Text object
    '''
    l = txto.title if hasattr(txto, 'title') else ''
    if l is None or l.strip() == '': l = txto.text[:150] # if no title label is start of text
    l = u' '.join(l.split()) # collapse and convert all whitespaces to blanks
    l = '%7s %s' % (str(txto.id), l)
    #l = '%s\n   %s' % (str(txto.id), l)
    return l

def parseLabelingFolder(folder):
    '''
    Parse all the labeling files within the folder and return
        list with all the parsed pairs.
    '''
    pairs = []
    for f in loc(folder).files():
        pairs.extend(parseLabelingFile(f))
    return pairs

def parseLabelingFile(fname, nonlabeled=True):
    '''
    Parse txt file with labeled pairs, each pair represented as text with pairLabelingText
    :param nonlabeled: if True, pair labels can be empty (unannotated pairs)
    :return: List of parsed pairs represented as (pairId, topic1, topic2, label)
    '''
    start, pairs = True, []
    label, tids, pid = None, [], None
    def linecontent(line, label): return line[len(label):].strip()
    def parselabelvalue(value):
        v = value.replace('?', '').strip()
        if v == '': return None
        return float(v)
    def addPair(): # add parsed pair
        if start: return
        if not nonlabeled: assert label is not None
        assert pid
        assert len(tids) == 2
        pairdata = (pid, tids[0], tids[1], label)
        pairs.append(pairdata)
    # print
    # print fname
    for i, l in enumerate(codecs.open(fname, 'r', 'utf-8').readlines()):
        l = l.strip()
        # print i, l
        if l.startswith(matchLabel): # start of new topic pair segment
            addPair(); start = False
            label, tids, pid = None, [], None # reset parsed data
            # parse label
            label = linecontent(l, matchLabel)
            label = parselabelvalue(label)
        if l.startswith(pairIdLabel):
            pid = int(linecontent(l, pairIdLabel))
        if l.startswith(topicIdLabel):
            t = linecontent(l, topicIdLabel).split(topicIdDelim)
            mid = t[0].strip()
            tid = int(t[1].strip())
            t = (mid, tid)
            tids.append(t)
    addPair() # add last pair
    return pairs
