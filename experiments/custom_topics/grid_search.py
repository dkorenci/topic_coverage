from experiments.grid_search.engine import *
from experiments.custom_topics.options import PriorOptions
from tools import *
from models.description import save_description_to_file
from pymedialab_settings.settings import object_store
import experiments.labeling.labelsets as labelsets

def custom_topics_gridsearch(defThemes, nonPriorOpts, priorOptions, gsFolder, seed,
                             numProcesses = 1, testSize = 2000, corpusId = 'us_politics'):
    themes = labelsets.labels_rights()
    for t in defThemes: themes.append(t)
    dictionary = loadDictionary(corpusId)
    defined_topics = getDefiningWords(themes, 'civil_rights_themes')
    checkDefindedTopicWords(defined_topics, dictionary)
    opts = []; N = len(dictionary)
    for npo in nonPriorOpts: # create cross product of nonPriorOpts with prior creation options
        T = npo.num_topics
        for po in priorOptions:
            topic_priors = np.empty((T, N))
            # merge non prior options into prior options
            priorOpts = copy.copy(po)
            # todo: set noncustPrior to npo.eta?
            priorOpts.numTopics = npo.num_topics; priorOpts.numWords = N
            # priorOpts = PriorOptions(numTopics=npo.num_topics, numWords=N,
            #                          defWordProb=po.defWordProb, nondefPrior=po.nondefWordPrior,
            #                          noncustPrior=po.noncustPrior)
            for t in range(T):
                if t < len(themes): # set custom priors, for words in a theme with index t
                    topic_words = defined_topics[themes[t]]
                    topic_priors[t, ] = createCustomTopic(priorOpts, topic_words, dictionary)
                else: # set generic priors
                    topic_priors[t, ] = createNoncustomTopic(priorOpts)
            newOpt = copy.copy(npo); newOpt.eta = topic_priors; newOpt.label = str(priorOpts)
            opts.append(newOpt)

    gridSearchParallel(folder=gsFolder, corpusId=corpusId,
                       options=opts, processes=numProcesses, testSize = testSize,
                       seed=seed, propagateSeed=True, shuffleOpts=True, label='', evalPasses=False)

    # generate model descriptions from theme labels (for each number of topics)
    for numTopics in [npo.num_topics for npo in nonPriorOpts]:
        d = createDescriptionsFromLabels(themes, numTopics)
        save_description_to_file(normalize_path(object_store+gsFolder)+'description%d.xml'%numTopics, d)

def grid_search_test():
    numTopics = [50, 100]
    nonPriorOpts = []
    for T in numTopics:
        nonPriorOpts.append(ModelOptions(num_topics=T, alpha=50.0/T, alpha_init=None, eta=0.01, offset=1.0,
                            decay=0.5, chunksize=1000, passes=2))
    defThemes = ['intelligence', 'cybersecurity', 'prisons', 'violent crimes']
    priorOpts = [
        PriorOptions(nondefPrior=0.001, noncustPrior=0.01, probMass=0.8, strategy='divide_mass'),
        PriorOptions(defWordProb=0.03, nondefPrior=0.001, noncustPrior=0.01, probMass=0.6, strategy='per_word')
    ]
    custom_topics_gridsearch(defThemes, nonPriorOpts, priorOpts,
                             gsFolder = 'grid_search_us_politics_rights4X', seed = 28556, numProcesses=2)

def grid_search1():
    # currently customized form printing prior statistics
    numTopics = [100] #, 120, 150]
    nonPriorOpts = []
    for T in numTopics:
        nonPriorOpts.append(ModelOptions(num_topics=T, alpha=50.0/T, alpha_init=None, eta=0.01, offset=1.0,
                            decay=0.5, chunksize=1000, passes=10))
    # auxilliary themes
    defThemes = ['intelligence', 'cybersecurity', 'prisons', 'violent crimes']
    # prior construction options
    priorOpts = []
    customTopicNondefWordPrior = [0.001] # [0.0005, 0.001, 0.002, 0.005]
    perWordProb =  [0.03] #[0.01, 0.03, 0.06, 0.12]
    probMass = [0.1, 0.2, 0.3, 0.5, 0.8]
    for pwp in perWordProb:
        for ndp in customTopicNondefWordPrior:
            priorOpts.append(
                PriorOptions(defWordProb=pwp, nondefPrior=ndp, noncustPrior=0.01,
                             probMass=0.95, strategy='per_word')
            )
    # for mass in probMass:
    #     for ndp in customTopicNondefWordPrior:
    #         priorOpts.append(
    #             PriorOptions(nondefPrior=ndp, noncustPrior=0.01, probMass=mass, strategy='divide_mass')
    #         )
    custom_topics_gridsearch(defThemes, nonPriorOpts, priorOpts,
                             gsFolder = 'grid_search_us_politics_rights5big', seed = 28556, numProcesses=3)
    # [
    #     PriorOptions(nondefPrior=0.001, noncustPrior=0.01, probMass=0.8, strategy='divide_mass'),
    #     PriorOptions(defWordProb=0.03, nondefPrior=0.001, noncustPrior=0.01, probMass=0.6, strategy='per_word')
    # ]

def grid_search2():
    'same as 1, but with 50 topics only'
    numTopics = [50]
    nonPriorOpts = []
    for T in numTopics:
        nonPriorOpts.append(ModelOptions(num_topics=T, alpha=50.0/T, alpha_init=None, eta=0.01, offset=1.0,
                            decay=0.5, chunksize=1000, passes=10))
    # auxilliary themes
    defThemes = ['intelligence', 'cybersecurity', 'prisons', 'violent crimes']
    # prior construction options
    priorOpts = []
    customTopicNondefWordPrior = [0.0005, 0.001, 0.002, 0.005]
    perWordProb = [0.01, 0.03, 0.06, 0.12]
    probMass = [0.1, 0.2, 0.3, 0.5, 0.8]
    for pwp in perWordProb:
        for ndp in customTopicNondefWordPrior:
            priorOpts.append(
                PriorOptions(defWordProb=pwp, nondefPrior=ndp, noncustPrior=0.01,
                             probMass=0.95, strategy='per_word')
            )
    for mass in probMass:
        for ndp in customTopicNondefWordPrior:
            priorOpts.append(
                PriorOptions(nondefPrior=ndp, noncustPrior=0.01, probMass=mass, strategy='divide_mass')
            )
    custom_topics_gridsearch(defThemes, nonPriorOpts, priorOpts,
                             gsFolder = 'grid_search_us_politics_rights5big', seed = 28556, numProcesses=3)

def grid_search3():
    'refinement of the best model'
    numTopics = [150]
    nonPriorOpts = []
    for T in numTopics:
        nonPriorOpts.append(ModelOptions(num_topics=T, alpha=50.0/T, alpha_init=None, eta=0.01, offset=1.0,
                            decay=0.5, chunksize=1000, passes=10))
    # auxilliary themes
    # defThemes = ['intelligence', 'cybersecurity', 'prisons', 'violent crimes',
    #              'black rights', 'racial issues', 'children', 'woman rights',
    #              'it technology', 'drugs', 'ebola']
    defThemes = ['intelligence', 'cybersecurity', 'prisons', 'violent crimes',
                 #'black rights', 'racial issues', 'children',
                 'it technology', 'drugs', 'woman rights']
    # prior construction options
    priorOpts = []
    customTopicNondefWordPrior = [0.005]
    perWordProb = [0.01]
    for pwp in perWordProb:
        for ndp in customTopicNondefWordPrior:
            priorOpts.append(
                PriorOptions(defWordProb=pwp, nondefPrior=ndp, noncustPrior=0.01,
                             probMass=0.95, strategy='per_word')
            )
    custom_topics_gridsearch(defThemes, nonPriorOpts, priorOpts,
                             gsFolder = 'grid_search_us_politics_rights5best_refine2', seed = 28556, numProcesses=1)