# -*- coding: utf-8 -*-

from doc_topic_coh.resources import pytopia_context

from agenda.themes import TableParser
from pytopia.context.ContextResolver import resolve

from doc_topic_coh.dataset.topic_labels import topicLabelStats
from doc_topic_coh.evaluations.experiment import IdList

from doc_topic_coh.settings import croelect_labeled_topics_clean, croelect_labeled_topics_improve
        #,croelect_labeled_topics, croelect_labeled_topics_clean

#croelect_labeled_topics = croelect_labeled_topics_clean
croelect_labeled_topics = croelect_labeled_topics_improve

__tableParseRaw, __tableParseClean = None, None
def tableParse(topics='improved'):
    global __tableParseRaw, __tableParseClean
    if topics == 'raw':
        if __tableParseRaw is None:
            __tableParseRaw = \
                TableParser(table=croelect_labeled_topics, dataRows=[(2, 96)], # 106
                            themeCol='A', topicCol='E').parse()
        return __tableParseRaw
    else:
        if __tableParseClean is None:
            __tableParseClean = \
                TableParser(table=croelect_labeled_topics, dataRows=[(2, 96)],
                            themeCol='A', topicCol='D').parse()
        return __tableParseClean

__cleanTableTopics = None
def cleanTableTopics():
    global __cleanTableTopics
    if __cleanTableTopics is None:
        pclean = tableParse('clean')
        __cleanTableTopics = list(pclean.topics.keys())
    return __cleanTableTopics

def topicTabLabel(t):
    '''
    Convert pytopia (mid, tid) topic to label used in annotation excel tables
    '''
    return '%s.%d'%(t[0][9:],t[1])

def topicFeatures(topic, table=None):
    '''
    Extract features of a topic from topic - semantic topic table and topic description.
    :param topic: (modelId, topicId)
    '''
    mid, tid = topic
    model = resolve(mid)
    assert mid.startswith('croelect_')
    mid = mid[9:]
    # if table is None:
    #     l = topicTabLabel(topic)
    #     if l in cleanTableTopics(): return topicFeatures(topic, 'clean')
    #     else:
    #         f = topicFeatures(topic, 'raw')
    #         print l, f
    # parse = tableParse(table)
    parse = tableParse()
    topicLabel = '%s.%d' % (mid, tid)

    #print topicLabel
    f = {}
    ptopic = parse.getTopic(topicLabel)
    #print ptopic.themes
    # num_themes
    f['num_themes'] = len(ptopic.themes)
    # table_mixed
    f['table_mixed'] = ptopic.mixed
    l = unicode(model.description.topic[tid].label).lower().strip()
    # label_noiseonly
    if l == u'šum': f['label_noiseonly'] = True
    else: f['label_noiseonly'] = False
    # label_noise
    if  u'šum' in l: f['label_noise'] = True
    else: f['label_noise'] = False
    return f

def labelingStandard(f, format='string'):
    '''
    Label croelect topics as one of these mutually exclusive categories:
    theme, theme_noise, theme_mix, theme_mix_noise, noise
    :param f: topic features
    :param format: map or string
    '''
    numThemes = f['num_themes']
    c = {}
    if numThemes == 1:
        if f['label_noise']: c['theme_noise'] = 1
        else: c['theme'] = 1
    elif numThemes == 0: c['noise'] = 1
    elif numThemes > 1:
        if f['label_noise']:
            c['theme_mix_noise'] = 1
        else: c['theme_mix'] = 1
    else: raise Exception('illegal number of themes: %s' % str(numThemes))
    if f['label_noiseonly']: c = {'noise':1}
    if format == 'string':
        for l, v in c.iteritems():
            if v == 1: return l
        return None
    return c

croelectModelIds = [ 'croelect_model1', 'croelect_model2', 'croelect_model3' ]
def getAllTopics():
    '''
    Return all topics of GtAR models as (modelId, topicId).
    '''
    all = [ (mid, tid) for mid in croelectModelIds for tid in sorted(resolve(mid).topicIds()) ]
    #clean = [t for t in all if topicTabLabel(t) in cleanTableTopics()]
    return all

def labelTopic(topic, features2categories):
    '''
    :param topic: (modelId, topicId)
    :param features2categories: callable that accepts a feature map
            and creates category map
    :return: category map
    '''
    return features2categories(topicFeatures(topic))

def labelAllTopics(features2categories):
    '''
    Label all the GtAR topics.
    :param features2categories: callable that accepts a feature map
            and creates category map
    :return: list of (topic, categories)
    '''
    return [ (t, labelTopic(t, features2categories)) for t in getAllTopics()]


def getTopics(modelIds = [ 'croelect_model1', 'croelect_model2', 'croelect_model3' ]):
    '''
    Return topics of specified croelect models.
    '''
    all = [ (mid, tid) for mid in modelIds for tid in sorted(resolve(mid).topicIds()) ]
    print len(all)
    #clean = [t for t in all if topicTabLabel(t) in cleanTableTopics()]
    return all

def labeledTopics(modelIds):
    ltopics = [(t, labelTopic(t, labelingStandard)) for t in getTopics(modelIds)]
    top = IdList(ltopics); top.id = 'croelect_topics_%s' % ('_'.join(modelIds))
    return top

def allTopics():
    ltopics = labelAllTopics(labelingStandard)
    alltop = IdList(ltopics); alltop.id = 'all_topics_croelect'
    return alltop

def analyzeParses():
    praw = tableParse('raw')
    pclean = tableParse('clean')
    traw = praw.topics.keys()
    tclean = pclean.topics.keys()
    print 'lengths raw %d clean %d' % (len(traw), len(tclean))
    print 'raw'
    for t in traw: print t
    print 'clean'
    for t in tclean: print t
    diff = [t for t in traw if t not in tclean]
    print 'clean missing topics'
    print diff

def printAllTopics():
    for t in allTopics(): print t

def topicStats():
    alltopics = labeledTopics(['croelect_model1', 'croelect_model2', 'croelect_model3', 'croelect_model4'])
    topicLabelStats(alltopics)

if __name__ == '__main__':
    #topicFeatures(('croelect_model1', 3))
    #analyzeParses()
    #print cleanTableTopics()
    #printAllTopics()
    #for t in getAllTopics():
    #     print '%s.%d'%(t[0][9:],t[1]), topicFeatures(t)
    topicStats()