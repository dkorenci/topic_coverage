from experiments.grid_search.engine import *
from experiments.custom_topics.options import PriorOptions
from tools import *
from models.description import save_description_to_file
from pymedialab_settings.settings import object_store
import experiments.labeling.labelsets as labelsets

def experiment1():
    corpusId = 'us_politics'
    themes = ['surveillance', 'intelligence', 'cybersecurity', 'hillary_clinton_email']
    defined_topics = { theme : loadThemeWords(theme) for theme in themes }
    dictionary = loadDictionary(corpusId)
    checkDefindedTopicWords(defined_topics, dictionary)
    nonPriorOpts = [ModelOptions(num_topics=100, alpha=0.5, alpha_init=None, eta=0.01, offset=1.0,
                            decay=0.5, chunksize=1000, passes=4)]
    opts = []; N = len(dictionary)
    for o in nonPriorOpts: # create cross product of nonPriorOpts with prior creation options
        T = o.num_topics
        prior = 0.01
        beta_ps = [0.001] #, 0.01, 0.0001]
        p = 0.03
        for beta_p in beta_ps:
            topic_priors = np.empty((T, N))
            for t in range(T):
                if t < len(themes): # set custom priors, for words in a theme with index t
                    topic_words = defined_topics[themes[t]]
                    Nw = len(topic_words)
                    beta_w = ( p * (N - Nw) * beta_p ) / (1.0 - p*Nw)
                    print t, Nw, beta_w, beta_p
                    topic_priors[t, ] = create_prior_vector(N, dictionary, topic_words, beta_w, beta_p)
                else: # set generic priors
                    topic_priors[t, ] = np.repeat(prior, N)
            newOpt = copy.copy(o)
            newOpt.eta = topic_priors
            newOpt.label = 'customTopics_p%.3fprior%.3f_betap%.5f' % (p, prior, beta_p)
            opts.append(newOpt)

    gridSearchParallel(folder='grid_search_us_politics_customtest2', corpusId='us_politics',
                       options=opts, processes=2, testSize = 2000,
                       seed=345556, propagateSeed=True, shuffleOpts=True, label='', evalPasses=False)

def experiment2():
    '''
    Test priors for 'civil rights' group of themes. Use new functionality.
    '''
    corpusId = 'us_politics'
    themes = ['abortion', 'cybersecurity', 'gun rights', 'immigration', 'medicare',
        'police shootings', 'student debt', 'chapel hill', 'death penalty',
        'hillary_clinton_email', 'intelligence', 'net neutrality',
        'surveillance', 'gay rights', 'homelessness',
        'marijuana', 'obamacare', 'selma', 'vaccination']
    dictionary = loadDictionary(corpusId)
    defined_topics = getDefiningWords(themes, 'civil_rights_themes')
    checkDefindedTopicWords(defined_topics, dictionary)
    nonPriorOpts = [ModelOptions(num_topics=100, alpha=0.5, alpha_init=None, eta=0.01, offset=1.0,
                            decay=0.5, chunksize=1000, passes=1)]
    opts = []; N = len(dictionary)
    for o in nonPriorOpts: # create cross product of nonPriorOpts with prior creation options
        T = o.num_topics
        nondefPriors = [0.001]
        for nondef in nondefPriors:
            topic_priors = np.empty((T, N))
            priorOpts = PriorOptions(numTopics=o.num_topics, numWords=N,
                                     defWordProb=0.03, nondefPrior=nondef, noncustPrior=0.01)
            for t in range(T):
                if t < len(themes): # set custom priors, for words in a theme with index t
                    topic_words = defined_topics[themes[t]]
                    topic_priors[t, ] = createCustomTopic(priorOpts, topic_words, dictionary)
                else: # set generic priors
                    topic_priors[t, ] = createNoncustomTopic(priorOpts)
            newOpt = copy.copy(o)
            newOpt.eta = topic_priors
            newOpt.label = str(priorOpts)
            opts.append(newOpt)

    gridSearchParallel(folder='grid_search_us_politics_customtest3', corpusId='us_politics',
                       options=opts, processes=2, testSize = 2000,
                       seed=12856, propagateSeed=True, shuffleOpts=True, label='', evalPasses=False)

def experiment_rights1():
    '''
    complete rights agenda, first try
    '''
    corpusId = 'us_politics'
    themes = [
               'civil rights movement', 'gay rights',  'police brutality',
               'chapel hill', 'fraternity racism', 'reproductive rights',
               'violence against women', 'death penalty', 'surveillance',
               'gun rights', 'net neutrality', 'marijuana', 'vaccination'
            ]
    dictionary = loadDictionary(corpusId)
    defined_topics = getDefiningWords(themes, 'civil_rights_themes')
    checkDefindedTopicWords(defined_topics, dictionary)
    nonPriorOpts = [ModelOptions(num_topics=100, alpha=0.5, alpha_init=None, eta=0.01, offset=1.0,
                            decay=0.5, chunksize=1000, passes=5)]
    opts = []; N = len(dictionary)
    for o in nonPriorOpts: # create cross product of nonPriorOpts with prior creation options
        T = o.num_topics
        nondefPriors = [0.001]
        for nondef in nondefPriors:
            topic_priors = np.empty((T, N))
            priorOpts = PriorOptions(numTopics=o.num_topics, numWords=N,
                                     defWordProb=0.03, nondefPrior=nondef, noncustPrior=0.01)
            for t in range(T):
                if t < len(themes): # set custom priors, for words in a theme with index t
                    topic_words = defined_topics[themes[t]]
                    topic_priors[t, ] = createCustomTopic(priorOpts, topic_words, dictionary)
                else: # set generic priors
                    topic_priors[t, ] = createNoncustomTopic(priorOpts)
            newOpt = copy.copy(o); newOpt.eta = topic_priors; newOpt.label = str(priorOpts)
            opts.append(newOpt)

    gsFolder='grid_search_us_politics_rights1'
    gridSearchParallel(folder=gsFolder, corpusId='us_politics',
                       options=opts, processes=2, testSize = 2000,
                       seed=28556, propagateSeed=True, shuffleOpts=True, label='', evalPasses=False)

    # generate model descriptions from theme labels (for each number of topics)
    for numTopics in [o.num_topics for o in nonPriorOpts]:
        d = createDescriptionsFromLabels(themes, numTopics)
        save_description_to_file(normalize_path(object_store+gsFolder)+'description%d.xml'%numTopics, d)

def experiment_rights2():
    '''
    complete rights agenda, first try
    '''
    corpusId = 'us_politics'
    themes = labelsets.labels_rights()
    defThemes = ['intelligence', 'cybersecurity']
    for t in defThemes: themes.append(t)
    dictionary = loadDictionary(corpusId)
    defined_topics = getDefiningWords(themes, 'civil_rights_themes')
    checkDefindedTopicWords(defined_topics, dictionary)
    nonPriorOpts = [ModelOptions(num_topics=100, alpha=0.5, alpha_init=None, eta=0.01, offset=1.0,
                            decay=0.5, chunksize=1000, passes=5)]
    opts = []; N = len(dictionary)
    for o in nonPriorOpts: # create cross product of nonPriorOpts with prior creation options
        T = o.num_topics
        nondefPriors = [0.001]
        for nondef in nondefPriors:
            topic_priors = np.empty((T, N))
            priorOpts = PriorOptions(numTopics=o.num_topics, numWords=N,
                                     defWordProb=0.03, nondefPrior=nondef, noncustPrior=0.01)
            for t in range(T):
                if t < len(themes): # set custom priors, for words in a theme with index t
                    topic_words = defined_topics[themes[t]]
                    topic_priors[t, ] = createCustomTopic(priorOpts, topic_words, dictionary)
                else: # set generic priors
                    topic_priors[t, ] = createNoncustomTopic(priorOpts)
            newOpt = copy.copy(o); newOpt.eta = topic_priors; newOpt.label = str(priorOpts)
            opts.append(newOpt)

    gsFolder='grid_search_us_politics_rights2'
    gridSearchParallel(folder=gsFolder, corpusId='us_politics',
                       options=opts, processes=2, testSize = 2000,
                       seed=28556, propagateSeed=True, shuffleOpts=True, label='', evalPasses=False)

    # generate model descriptions from theme labels (for each number of topics)
    for numTopics in [o.num_topics for o in nonPriorOpts]:
        d = createDescriptionsFromLabels(themes, numTopics)
        save_description_to_file(normalize_path(object_store+gsFolder)+'description%d.xml'%numTopics, d)




def createDescriptionTest():
    themes = [
           'civil rights movement', 'gay rights',  'police brutality',
           'chapel hill', 'fraternity racism', 'reproductive rights',
           'violence against women', 'death penalty', 'surveillance',
           'gun rights', 'net neutrality', 'marijuana', 'vaccination'
        ]
    d = createDescriptionsFromLabels(themes, 100)
    folder='grid_search_us_politics_rights1/'
    save_description_to_file(object_store+folder+'description.xml', d)