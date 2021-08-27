from resources.custom_model.tools import *
from models.description import save_description_to_file
from models.label import modelLabel
from utils.utils import unpinProcess

from multiprocessing import Pool
import random

class CustomizedModelBuilder():
    '''
    Builds models with topics user-defined via prior word lists.
    '''

    def __init__(self, resBuilder):
        '''
        :param resBuilder: ResourceBuilder
        '''
        self.resBuilder = resBuilder

    from experiments.custom_topics.eksperiments import experiment_rights2
    def buildModels(self, corpusId, wordlistFolder, topicLabels,
                    modelBuildOpts, priorBuildOpts, numThreads=1,
                    buildIndex=False, label=''):
        '''
        :param outputFolder: folder to save constructed models
        :param wordlistFolder: folder with word lists, one file per topic label
        :param topicLabels: topic labels, corresponding word list files must exist
        :param modelBuildOpts: non-prior related model building options, list of ModelOptions
        :param priorBuildOpts: options for constructing the prior,
            either list of PriorOptions objects or dict topicLabel -> PriorOptions
        :param label: additional label to add to models
        :param
        '''
        self.corpusId = corpusId
        self.dictionary = self.resBuilder.loadDictionary(corpusId)
        self.priorWords = {label: loadThemeWords(wordlistFolder, label) for label in topicLabels}
        self.__checkPriorWords()
        self.perTopicOptions = isinstance(priorBuildOpts, dict)
        self.__createBuildOptions(topicLabels, modelBuildOpts, priorBuildOpts, label)
        opts = priorBuildOpts if self.perTopicOptions else None
        self.__buildModels(numThreads, buildIndex, opts)
        # todo: move description creation to ModelBuildCallable
        self.__createModelDescriptions(topicLabels)

    def __buildModels(self, numThreads, buildIndex, opts):
        '''
        Build models in parallel, one for each build option, save to outFolder.
        '''
        pool = Pool(numThreads)
        random.shuffle(self.customBuildOpts)  # load balancing
        pool.map(ModelBuildCallable(self.resBuilder, self.corpusId, buildIndex, opts),
                    self.customBuildOpts)

    def __createModelDescriptions(self, topicLabels):
        '''
        For the models built, create decription files with defined topic labels
        '''
        for opt in self.customBuildOpts:
            d = createDescriptionsFromLabels(topicLabels, opt.num_topics)
            save_description_to_file(path.join(self.resBuilder.modelFolder(self.corpusId, opt),
                                               'description.xml'), d)

    def __createBuildOptions(self,  topicLabels, modelBuildOpts, priorBuildOpts, label):
        '''
        For each pair of mbOpt and pbOpt, construct topic prior array based on pbOpt,
         and create new model build option object based on mbOpt and the array.
        '''
        N = len(self.dictionary)
        self.customBuildOpts = []
        for mbo in modelBuildOpts:
            T = mbo.num_topics
            if not self.perTopicOptions:
                # create build options for cross product of
                # modelBuildOpts and priorBuildOpts
                for pbo in priorBuildOpts:
                    pbo.numWords, pbo.numTopics = N, T
                    topicPriors = np.empty((T, N))
                    for t in range(T):
                        if t < len(topicLabels):  # set custom priors, for words in a theme with index t
                            topicWords = self.priorWords[topicLabels[t]]
                            topicPriors[t,] = createCustomTopic(pbo, topicWords, self.dictionary)
                        else:  # set generic priors
                            topicPriors[t,] = createNoncustomTopic(pbo)
                    newOpt = copy.copy(mbo);
                    newOpt.eta = topicPriors
                    newOpt.label = str(pbo)+(u'_%s'%label) if label else ''
                    self.customBuildOpts.append(newOpt)
            else:
                topicPriors = np.empty((T, N))
                for t in range(T):
                    if t < len(topicLabels):
                        if topicLabels[t] in priorBuildOpts:
                            pbo = priorBuildOpts[topicLabels[t]]
                        else: raise Exception(
                            u'Build options for topic "%s" do not exist'%topicLabels[t])
                        topicWords = self.priorWords[topicLabels[t]]
                        pbo.numWords, pbo.numTopics = N, T
                        topicPriors[t,] = createCustomTopic(pbo, topicWords, self.dictionary)
                    else:  # set generic priors
                        topicPriors[t,] = createNoncustomTopic(pbo)
                newOpt = copy.copy(mbo);
                newOpt.eta = topicPriors
                newOpt.label = 'per_topic_options'+(u'_%s'%label) if label else ''
                self.customBuildOpts.append(newOpt)

    def __checkPriorWords(self):
        '''check that all prior words exist in the dictionary'''
        for topic, words in self.priorWords.items():
            for w in words:
                if w not in self.dictionary.token2id.keys():
                    raise ValueError('word %s from theme %s is not in the dictionary' % (w, topic))

import codecs
from os import path

class ModelBuildCallable:
    def __init__(self, resBuilder, corpusId, buildIndex=False, opts=None):
        '''
        :param resBuilder: ResourceBuilder
        :param buildIndex: if True build TopicIndex
        :param opts: None or map of topic label -> per-topic prior build option
                    which will be saved in file for every model
        '''
        self.resBuilder, self.corpusId = resBuilder, corpusId
        self.buildIndex = buildIndex; self.opts = opts

    def __call__(self, opt):
        print 'START building model %s' % modelLabel(opt)
        modelFolder = self.resBuilder.buildModel(self.corpusId, opt, overwrite=False)
        if self.opts:
            optstr = u'\n'.join(u'%s: %s'%(l, str(o)) for l, o in self.opts.items())
            f = codecs.open(path.join(modelFolder, 'topic_priors.txt'), 'w', 'utf8')
            f.write(optstr); f.close()
        print 'done building model %s' % modelLabel(opt)
        if self.buildIndex:
            print 'start building index %s' % modelLabel(opt)
            self.resBuilder.buildTopicIndex(modelFolder, self.corpusId, overwrite=False)
            print 'done building index %s' % modelLabel(opt)
        print 'DONE %s' % modelLabel(opt)
