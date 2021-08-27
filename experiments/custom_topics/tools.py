import copy

from models.interfaces import TopicModel
from pymedialab_settings.settings import labeled_models_folder as LMF
from resources.resource_builder import *
from utils.utils import normalize_path
from models.description import *


def createModelMap():
    'create modelId -> TopicModel map'
    mmap = {}
    mmap['uspolM0'] = TopicModel.load(LMF+'uspolM0_234_ldamodel_T50_A1.000_Eta0.010_Off1.000'
                                  '_Dec0.500_Chunk1000_Pass10_label_seed345556')
    mmap['uspolM1'] = TopicModel.load(LMF+'uspolM1_234_ldamodel_T50_A1.000_Eta0.010_Off1.000'
                                  '_Dec0.500_Chunk1000_Pass10_label_seed877312')
    mmap['uspolM2'] = TopicModel.load(LMF+'uspolM2_234_ldamodel_T50_A1.000_Eta0.010_Off1.000'
                                  '_Dec0.500_Chunk1000_Pass10_label_seed8903')
    mmap['uspolM10'] = TopicModel.load(LMF+'uspolM10_045_ldamodel_T100_A0.500_Eta0.010_Off1.000'
                                   '_Dec0.500_Chunk1000_Pass10_label_seed345556')
    mmap['uspolM11'] = TopicModel.load(LMF+'uspolM11_045_ldamodel_T100_A0.500_Eta0.010_Off1.000'
                                   '_Dec0.500_Chunk1000_Pass10_label_seed133890')
    return mmap

def loadThemeWords(topic_label, topic_set_folder):
    folder = normalize_path(themes_folder + topic_set_folder)
    file = folder+topic_label+'.txt'
    words = []
    for l in open(file).readlines():
        line = l.strip();
        if line == '' or line.startswith('#'): continue
        if '#' in line: line = line[0:line.find('#')].strip()
        words.append(line)
    return words

def getDefiningWords(topic_labels, topic_set_folder):
    'return map of theme_label -> list of theme words '
    return { label : loadThemeWords(label, topic_set_folder) for label in topic_labels }

def checkDefindedTopicWords(defined_topics, dictionary):
    '''
    check that all words in defined topics exitst in the dictionary
    :param defined_topics: map of topic_label -> list of words
    :param dict: gensim Dictionary
    '''
    for t in defined_topics:
        if defined_topics[t] is None: print t
        for w in defined_topics[t]:
            if w not in dictionary.token2id.keys():
                raise ValueError('word %s from theme %s is not in the dictionary' % (w, t))

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