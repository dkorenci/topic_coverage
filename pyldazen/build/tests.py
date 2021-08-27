#import pyximport;
#py_importer, pyx_importer = pyximport.install()
from pyldazen.build.LdaGibbsInferer import *
#pyximport.uninstall(py_importer, pyx_importer)
from pyldazen.build.LdaGibbsBuildData import LdaGibbsBuildData
from pyldazen.build.LdaGibbsBuilder import LdaGibbsBuilder

from croelect.resources.resource_builds import resourceBuilder
import numpy as np

def printModelTopics(data, dictionary, iterations, step, topW = 15, numFixed=10, numFixedWords=5):
    '''
    Run LdaGibbsInferer and inspect topics
    :param data: LdaGibbsBuildData
    :param dictionary: int -> token mapping
    :param iterations: number of gibbs markov chain samples to generate
    :param step: print topic every this many interation
    :param topW: represent each topic by topW top words
    :return:
    '''
    # prepare fixed topics matrix
    fixedTopics = np.zeros((numFixed, len(dictionary)), dtype=np.float64)
    for i in range(numFixed):
        for j in range(numFixedWords):
            fixedTopics[i, i*numFixedWords+j] = 1.0
        fixedTopics[i] /= numFixedWords
    data.fixedTopics = fixedTopics
    data.Tf = numFixed
    inferer = LdaGibbsBuilder.createInferer(data)
    inferer.startInference()
    for i in range(1, iterations+1):
        inferer.runInference(1)
        if i % step == 0:
            topics = inferer.calcTopicMatrix()
            print 'ITERATION %d' % i
            for t in range(inferer.numTopics()):
                if t < numFixed: topic = fixedTopics[t]
                else: topic = topics[t-numFixed]
                top_indices = np.argsort(topic)[::-1][:topW] # get indices in sorted order, reverse, take first topN
                words = [ dictionary[i] for i in top_indices ]
                print u'topic %d: %s' % (t, ' '.join(['%s %.4f'%(w,topic[i])
                                                      for w, i in zip(words, top_indices)]))
    m = inferer.calcDocTopicMatrix()
    print type(m), m.shape
    print m
    inferer.finishInference()

def topicTest1():
    '''
    Print inferred topics for cro news corpus.
    '''
    rb = resourceBuilder()
    corpus = rb.loadBowCorpus('iter0_cronews_agendadedup2')
    dict = rb.loadDictionary('iter0_cronews_agendadedup2')
    corpus = corpus[:1000]
    data = LdaGibbsBuildData(100, None, corpus, 1.0, 0.01)
    printModelTopics(data, dict, 20, 5)

if __name__ == '__main__':
    topicTest1()