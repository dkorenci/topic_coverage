from doc_topic_coh.dataset.topic_labels import *

from sklearn.model_selection import train_test_split
from doc_topic_coh.evaluations.experiment import IdList

import pickle
from os import path

def iter0DevTestSplit(printStats=False):
    id = 'iter0DevTestSplit'
    res = loadById(id)
    if res is not None: return res
    else:
        ltopics = labelAllTopics(labelingStandard)
        topics, labels = [ t for t, _ in ltopics ], [ l for _, l in ltopics ]
        split = train_test_split(ltopics, train_size=80, stratify=labels, random_state=8825)
        dev = IdList(split[0]); dev.id = 'dev'
        test = IdList(split[1]); test.id = 'test'
        saveById(id, (dev, test))
        if printStats:
            topicLabelStats(ltopics)
            topicLabelStats(dev)
            topicLabelStats(test)
        return dev, test

def devTestSplit2(printStats=False):
    id = 'devTestSplit2'
    res = loadById(id)
    if res is not None: return res
    else:
        ltopics = labelAllTopics(labelingStandard)
        topics, labels = [ t for t, _ in ltopics ], [ l for _, l in ltopics ]
        split = train_test_split(ltopics, train_size=120, stratify=labels, random_state=897423)
        dev = IdList(split[0]); dev.id = 'dev2'
        test = IdList(split[1]); test.id = 'test2'
        saveById(id, (dev, test))
        if printStats:
            topicLabelStats(ltopics)
            topicLabelStats(dev)
            topicLabelStats(test)
        return dev, test

def devTestSplit():
    return topicSplit(devSize=120, rndseed=9984)

def smallTestSample(size=10, rndseed=329):
    import random
    at = allTopics()
    random.seed(rndseed)
    random.shuffle(at)
    sample = IdList(at[:size])
    sample.id = 'test_sample_[size=%d]_[seed=%d]' % (size, rndseed)
    return sample

def topicSplit(devSize=120, rndseed=78298):
    id = 'topic_split_[devSize=%d]_[seed=%d]' % (devSize, rndseed)
    res = loadById(id)
    if res is not None: return res
    ltopics = allTopics()
    topics, labels = [ t for t, _ in ltopics ], [ l for _, l in ltopics ]
    split = train_test_split(ltopics, train_size=devSize,
                             stratify=labels, random_state=rndseed)
    dev = IdList(split[0]); dev.id = 'dev_%s'%id
    test = IdList(split[1]); test.id = 'test_%s'%id
    saveById(id, (dev, test))
    return dev, test

_saveFolder = path.join(path.dirname(__file__), 'saved_datasets')
def resourceFname(id):
    fname = '%s.pickle' % str(id)
    return path.join(_saveFolder, fname)

def saveById(id, obj):
    ''' Pickle object to the dataset folder. '''
    pickle.dump(obj, open(resourceFname(id), 'wb'))

def loadById(id):
    ''' Unpickle object from the dataset folder. '''
    fname = resourceFname(id)
    if path.exists(fname):
        return pickle.load(open(fname, 'rb'))
    else: return None

def allTopics():
    id = 'all_topics'
    res = loadById(id)
    if res is not None: return res
    else:
        ltopics = labelAllTopics(labelingStandard)
        alltop = IdList(ltopics); alltop.id = id
        saveById(id, alltop)
        return alltop

def iter0TrainTestSplit(printStats=False):
    ltopics = labelAllTopics(labelingStandard)
    topics, labels = [ t for t, _ in ltopics ], [ l for _, l in ltopics ]
    split = train_test_split(ltopics, train_size=200, stratify=labels, random_state=123)
    train = split[0]
    test = split[1]
    if printStats:
        topicLabelStats(ltopics)
        topicLabelStats(train)
        topicLabelStats(test)
    return train, test

if __name__ == '__main__':
    #print topicSplit()
    topicLabelStats(allTopics())
    #iter0DevTestSplit()
    #devTestSplit2()

