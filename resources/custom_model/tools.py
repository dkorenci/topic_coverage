import copy

from models.interfaces import TopicModel
from models.description import *

import numpy as np
from os import path
import codecs

def loadThemeWords(folder, topicLabel):
    '''
    Load list of words from txt file with same name as the label,
    residing in folder.
    '''
    file = path.join(folder, '%s.txt'%topicLabel)
    words = []
    for l in codecs.open(file, encoding='utf-8').readlines():
        line = l.strip();
        if line == '' or line.startswith('#'): continue
        if '#' in line: line = line[0:line.find('#')].strip()
        words.append(line)
    return words

def createCustomTopic(opts, words, dictionary, stats = False):
    ' Create topic prior vector form options '
    if opts.strategy == 'per_word' : return customTopicPerWord(opts, words, dictionary, stats)
    elif opts.strategy == 'divide_mass' : return customTopicDivideMass(opts, words, dictionary, stats)
    else: raise Exception('undefined topic customization strategy')

def customTopicPerWord(opts, words, dictionary, stats = False):
    '''
    Create topic prior vector form options, so that expectation for each word in words
    is opts.defWordProb, and prior value for the rest of the words is opts.nondefWordPrior
    :param opts: PriorOptions
    :param words: list of words for which prior probabilities will be defined
    :return array of size opts.numWords with priors set
    '''
    Nw = len(words)
    if opts.defWordProb * Nw < 1.0 : dwprob = opts.defWordProb
    else: dwprob = opts.probMass / float(Nw)
    defWordPrior = ( dwprob * (opts.numWords - Nw) * opts.nondefWordPrior ) \
                / (1.0 - dwprob* Nw)
    if stats:
        D = len(dictionary); W =  len(words)
        print 'num words in dictionary: %d' % D
        print 'num seed words: %d' % W
        print 'defined word prior: %.4f ' % defWordPrior
        undefWordProb = opts.nondefWordPrior/(defWordPrior * len(words) + opts.nondefWordPrior * (D-W))
        print 'undef word prob: %.10f ' % undefWordProb

    return create_prior_vector(opts.numWords, dictionary, words, defWordPrior, opts.nondefWordPrior)

def customTopicDivideMass(opts, words, dictionary, stats = False):
    '''
    Create topic prior vector form options, so that probability mass given in options
    is equally divided between prior words.
    :param opts: PriorOptions
    :param words: list of words for which prior probabilities will be defined
    :return array of size opts.numWords with priors set
    '''
    o = copy.copy(opts)
    Nw = len(words)
    o.defWordProb = o.probMass / float(Nw)
    return customTopicPerWord(o, words, dictionary)

def createNoncustomTopic(opts):
    ' :param opts: PriorOptions '
    return np.repeat(opts.noncustPrior, opts.numWords)

def create_prior_vector(N, dict, words, wordPerc, perc):
    '''
    create vector of dimension N, with wordPerc value at word positions
     and perc values at the rest of positions
    '''
    rtopic = np.repeat(perc, N)
    for w in words:
        if isinstance(w, basestring):
            rtopic[dict.token2id[w]] = wordPerc
        else:
            rtopic[w] = wordPerc
    return rtopic

def createDescriptionsFromLabels(labels, numTopics):
    'create model descriptions where first len(labels) topic correspond to labels'
    d = empty_description(numTopics)
    for i, l in enumerate(labels):
        d.topic[i].label = l
    return d