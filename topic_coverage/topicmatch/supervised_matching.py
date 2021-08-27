from pytopia.tools.IdComposer import IdComposer

from topic_coverage.topicmatch.supervised_iter0 import featExtract, metricSet, corpusTopicWeights
from topic_coverage.modelbuild.modelbuild_iter1 import modelsContext
from topic_coverage.modelbuild.modelbuild_docker_v1 import loadBuild1Modelset
from phenotype_context.phenotype_topics.construct_model import MODEL_ID as PHENO_MODEL
from topic_coverage.topicmatch.labeling_iter1_uspolfinal import  uspolLabelingModelsContex as uspolLabModels
from pytopia.measure.avg_nearest_dist import TopicCoverDist
from pytopia.measure.topic_distance import cosine

from topic_coverage.topicmatch.supervised_iter0 import optMatchModel, logistic, uspolBinLab, gbt
from topic_coverage.topicmatch.supervised_iter1 import dataset

from pytopia.context.ContextResolver import resolve

uspolRefmodel = resolve('gtar_themes_model')
phenoRefmodel = resolve(PHENO_MODEL)

class SupervisedMatcherV1(IdComposer):
    '''
    Topic equality judgements based on a supervised model.
    Using old feature extraction not embedded within the model.
    '''

    def __init__(self, model, featSet='all-distances', metrics='cosine', id=None):
        self._model = model
        self._fset, self._metrics = featSet, metricSet(metrics)
        # todo remove this duplication of atts
        if id is None:
            self.model, self.featSet, self.metrics = model, featSet, metrics
            IdComposer.__init__(self)
        else:
            IdComposer.__init__(self); self.id = id
            self.model, self.featSet, self.metrics = model, featSet, metrics

    def __call__(self, t1, t2):
        #print t1.model, t1.topicId
        #print t2.model, t2.topicId
        feats = featExtract(t1, t2, self._fset, self._metrics)
        predict = self._model.predict([feats])
        #print type(predict), predict
        return predict[0]

class SupervisedTopicMatcher(IdComposer):
    '''
    Topic equality judgements based on a supervised model.
    Adapter that enables calling sklearn model with a pair of topics.
    '''

    def __init__(self, model):
        self.model = model
        IdComposer.__init__(self)

    def __call__(self, t1, t2):
        #print t1.model, t1.topicId
        #print t2.model, t2.topicId
        predict = self.model.predict([(t1, t2)])
        #print type(predict), predict
        return predict[0]

class TopicmatchModelCoverage(IdComposer):
    '''
    Calculator of coverage score based on a matcher, a callable that
    returns, for a pair of topics, 0 (topics don't match) or 1 (topics match).
    '''

    def __init__(self, matcher):
        '''
        :param matcher: callable receiving two topics and returning a 0/1 match judgements
        '''
        self._matcher = matcher
        self.matcher = matcher
        IdComposer.__init__(self)

    def __call__(self, target, model):
        ''' Calculate coverage of target topics by model topics. '''
        matched = 0.0
        for tt in target:
            for mt in model:
                if self._matcher(tt, mt):
                    matched += 1
                    break
        return matched / target.numTopics()

class TopicmatchModelPrecision(IdComposer):
    '''
    Calculator of precision score based on a topic matcher, a callable that
    returns, for a pair of topics, 0 (topics don't match) or 1 (topics match).
    '''

    def __init__(self, matcher):
        '''
        :param matcher: callable receiving two topics and returning a 0/1 match judgements
        '''
        self._matcher = matcher
        self.matcher = matcher
        IdComposer.__init__(self)

    def __call__(self, target, model):
        ''' Calculate coverage of target topics by model topics. '''
        usedTopics = set()
        for tt in target:
            for mt in model:
                if self._matcher(tt, mt):
                    usedTopics.add(mt.id)
                    break
        return len(usedTopics) / float(model.numTopics())

def optSupervisedMatcher(iter=0, corpus='uspol', model='logreg'):
    if corpus == 'uspol':
        if iter == 0:
            fset, metric = 'all-distances', 'cosine'
            with modelsContext():
                optmodel = optMatchModel(logistic(), uspolBinLab(), fset, metric)
            return SupervisedMatcherV1(optmodel, fset, metric, id='optLogisticRegV1_iter%d' % iter)
        elif iter == 1:
            fset, metric = 'all-distances', 'all'
            with uspolLabModels():
                if model == 'logreg':
                    optmodel = optMatchModel(logistic(), dataset(0.75), fset, metric)
                elif model == 'gbt':
                    optmodel = optMatchModel(gbt(), dataset(0.75), fset, metric)
            if model == 'logreg':
                return SupervisedMatcherV1(optmodel, fset, metric, id='optLogisticRegV1_iter%d' % iter)
            elif model == 'gbt':
                return SupervisedMatcherV1(optmodel, fset, metric, id='optGbtV1_iter%d' % iter)
    elif corpus == 'pheno': return optSupervisedMatcherPheno()

def optSupervisedMatcherPheno():
    fset, metric = 'all-distances', 'all'
    with uspolLabModels():
        optmodel = optMatchModel(logistic(), dataset(0.75, corpus='pheno'), fset, metric)
    return SupervisedMatcherV1(optmodel, fset, metric, id='optLogisticRegV1_pheno')

def runSupervisedCoverTest(iter=0, model='lda', T=50):
    from gtar_context import gtarContext
    optmatcher = optSupervisedMatcher(iter=iter)
    supCover = TopicmatchModelCoverage(optmatcher)
    ldaModels = loadBuild1Modelset(model, T, asContext=True)
    with gtarContext():
        with ldaModels:
            for m in ldaModels.itervalues():
                print m.id
                print '%g' % supCover(uspolRefmodel, m)

def measurePlanePlot(dataset):
    from matplotlib import pyplot as plt, numpy as np
    # calculate distances and labels
    dists = []; labs = []
    for t1, t2, lab in dataset:
        tv1, tv2 = t1.vector, t2.vector
        tdv1, tdv2 = corpusTopicWeights(t1), corpusTopicWeights(t2)
        distPoint = [ cosine(tv1, tv2), cosine(tdv1, tdv2) ]
        dists.append(distPoint)
        labs.append(lab)
    dists = np.array(dists)
    print labs
    # plot
    fig, ax = plt.subplots()
    colors = [ 'r' if l == 1 else 'g' for l in labs ]
    ax.scatter(dists[:, 0], dists[:, 1], color=colors, s=4)
    plt.savefig('coscos' + '.pdf')

if __name__ == '__main__':
    #optSupervisedMatcher(iter=1)
    #runSupervisedCoverTest(1, 'nmf', 50)
    #with modelsContext(): measurePlanePlot(uspolBinLab())
    with uspolLabModels(): measurePlanePlot(dataset(1.0))



