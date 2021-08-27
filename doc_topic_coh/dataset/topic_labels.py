from doc_topic_coh.dataset.topic_features import *

def labelingCoherent(f):
    '''
    Create category labeling describing if topic
    corresponds to a single theme (semantic topic).
    :param f: topic features
    '''
    c = {}
    if f['num_themes'] == 1 and not \
        (f['table_mixed'] | f['label_mixed'] | f['label_noise']):
        c['coherent'] = 1
    else: c['coherent'] = 0
    return c

def labelingStandard(f, format='string'):
    '''
    Label topics as one of these mutually exclusive categories:
    theme, theme_noise, theme_mix, theme_mix_noise, noise
    :param f: topic features
    :param format: map or string
    '''
    numThemes = f['num_themes']
    c = {}
    if numThemes == 1:
        if f['table_mixed'] | f['label_mixed'] | f['label_noise']:
            c['theme_noise'] = 1
        else: c['theme'] = 1
    elif numThemes == 0: c['noise'] = 1
    elif numThemes > 1:
        if f['label_noise']:
            c['theme_mix_noise'] = 1
        else: c['theme_mix'] = 1
    else: raise Exception('illegal number of themes: %s' % str(numThemes))
    if format == 'string':
        for l, v in c.iteritems():
            if v == 1: return l
        return None
    return c

def labelTopic(topic, features2categories):
    '''
    :param topic: (modelId, topicId)
    :param features2categories: callable that accepts a feature map
            and creates category map
    :return: category map
    '''
    return features2categories(topicFeatures(topic))

def topicLabelStats(ltopics):
    '''
    Display topic/label distribution
    :param ltopics: list of (topic, label)
    :return:
    '''
    lcnt = {}; N = len(ltopics)
    for _, labeling in ltopics:
        if isinstance(labeling, dict):
            labels = [ l for l, v in labeling.iteritems() if v == 1 ]
        else: labels = [labeling]
        for lab in labels:
            if lab in lcnt: lcnt[lab] += 1
            else: lcnt[lab] = 1
    print 'number of topics:', N
    for l in sorted(lcnt.keys()):
        print l, '%.4f'%(float(lcnt[l])/N), '%4d'%lcnt[l]

def labelAllTopics(features2categories):
    '''
    Label all the GtAR topics.
    :param features2categories: callable that accepts a feature map
            and creates category map
    :return: list of (topic, categories)
    '''
    return [ (t, labelTopic(t, features2categories)) for t in getAllTopics()]
