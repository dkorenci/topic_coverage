from doc_topic_coh.resources import pytopia_context
from pytopia.context.ContextResolver import resolve

from coverexp.utils.settings_loader import dataFolder
#from pytopica.utils.print_ import topTopicWordsString
from agenda.themes import TableParser

from os import path

gtarModelIds = [ 'uspolM0', 'uspolM1', 'uspolM2', 'uspolM10', 'uspolM11' ]

# folder containing above models
baseFolder = dataFolder().subfolder('gtar_original_resources')

def getAllTopics():
    '''
    Return all topics of GtAR models as (modelId, topicId).
    '''
    return [ (mid, tid) for mid in gtarModelIds for tid in sorted(resolve(mid).topicIds()) ]

def getTopicFeatures():
    '''
    :return: (topic, features) for all GtAR topics.
    '''
    return [ (t, topicFeatures(t)) for t in getAllTopics() ]

__tableParse = None
def tableParse():
    global __tableParse
    if __tableParse is None:
        from doc_topic_coh.settings import gtar_labeled_topics
        __tableParse = \
            TableParser(table=gtar_labeled_topics, dataRows=[(3,76), (84, 142)]).parse()
    return __tableParse

def topicFeatures(topic):
    '''
    Extract features of a topic from topic - semantic topic table and topic description.
    :param topic: (modelId, topicId)
    '''
    parse = tableParse()
    mid, tid = topic
    topicLabel = '%s.%d' % (mid, tid)
    f = {}
    ptopic = parse.getTopic(topicLabel)
    # num_themes
    f['num_themes'] = len(ptopic.themes)
    # table_mixed
    f['table_mixed'] = ptopic.mixed
    model = resolve(mid)
    l = str(model.description.topic[tid].label).lower().strip()
    # label_mixed
    if l.startswith('mix:') or l.startswith('mixture:'): f['label_mixed'] = True
    else: f['label_mixed'] = False
    # label_mixonly
    if l == 'mix' or l == 'mixture': f['label_mixonly'] = True
    else: f['label_mixonly'] = False
    # label_noise
    if l.endswith('et al'): f['label_noise'] = True
    else: f['label_noise'] = False
    # stopwords
    f['stopwords'] = (l == 'stopwords')
    return f

def printUnthemedTopics():
    ''' Print topics without associed themes. '''
    for t, f in getTopicFeatures():
        mid, tid = t
        if f['num_themes'] == 0:
            print '%s.%s' % (mid, tid)

if __name__ == '__main__':
    printUnthemedTopics()
    #print getTopicFeatures()
    #print labelAllTopics(isCoherentTopic)




