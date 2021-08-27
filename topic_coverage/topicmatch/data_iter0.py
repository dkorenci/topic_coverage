from os import path
from topic_coverage.modelbuild.modelbuild_iter1 import modelsContext
from pytopia.context.ContextResolver import resolve

folder = '/home/damir/Dropbox/projekti/doktorat/D1 eksplorativa/mjerenje pokrivenosti/' \
         'supervised/oznaceni parovi/uspol_binary_prelim'
fnames = [
    'topicPairs[150-200]_ristov_done.txt', 'topicPairs[200-250]_ristov_done.txt',
    'topicPairs[250-300]_ristov_done.txt', 'topicPairs[300-350]_damir_done.txt',
    'topicPairs[350-400]_damir_done.txt', 'topicPairs[400-450]_damir_done.txt'
    ]

def resolveTopic(topic):
    mid, tid = topic
    model = resolve(mid)
    return model[tid]

def labelProportions(data):
    labels = sorted(set(l for _, _, l in data))
    N = float(len(data))
    cnt = { l : sum(dl == l for _, _, dl in data) for l in labels }
    print 'num datapoints: %d' % N
    for l in labels:
        print ' label %s : %d , %g' % (l, cnt[l], cnt[l]/N)

def loadDataset(folder, fnames=None, nonlabeled=False, resolve=True, raw=False):
    '''
    Load dataset of annotated topic pairs:
    parse annotations, load models, return a list of (Topic, Topic, label)
    '''
    from topic_coverage.topicmatch.pair_labeling import parseLabelingFile
    from pyutils.file_utils.location import FolderLocation
    if not fnames: lpairFiles = FolderLocation(folder).files()
    else:
        if not isinstance(fnames, list): fnames = [fnames]
        lpairFiles = [path.join(folder, f) for f in fnames]
    rawdata = [ pair for f in lpairFiles for pair in parseLabelingFile(f, nonlabeled=nonlabeled) ]
    if raw: return rawdata
    if resolve: data = [ (resolveTopic(t1), resolveTopic(t2), label) for _, t1, t2, label in rawdata ]
    else: data = [(t1, t2, label) for _, t1, t2, label in rawdata]
    return data

if __name__ == '__main__':
    #with modelsContext(): loadDataset(folder, fnames)
    #labelProportions(loadDataset(folder, fnames, resolve=False))
    for f in fnames:
        labelProportions(loadDataset(folder, [f], resolve=False))