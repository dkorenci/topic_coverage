from pyutils.stat_utils.plots import basicValueDist
from pyutils.stat_utils.utils import Stats

from pytopia.measure.topic_distance import *
from random import sample, seed

from gtar_context.semantic_topics.construct_model import MODEL_ID as GTAR_REFMODEL
from phenotype_context.phenotype_topics.construct_model import MODEL_DOCS_ID as PHENO_REFMODEL
from topic_coverage.resources.pytopia_context import topicCoverageContext
from pytopia.context.ContextResolver import resolve
from topic_coverage.settings import topic_models_folder

def sampleTopicPairs(topics, sampleSize=None, rseed=9342123, split=False):
    '''
    Generate (a sample of) all topic pairs from models
    Return a list of [((modelIndex1, topicId1), (modelIndex2, topicId2))]
    :param models: list of pytopia models
    :param sampleSize: if None, take all pairs
    :param split: if True sample pairs with one member from topics[0] and other from topics[1]
    :return:
    '''
    if not split:
        topicPairs = [(t1, t2) for ti, t1 in enumerate(topics) for tj, t2 in enumerate(topics) if (tj > ti)]
    else:
        topicPairs = [(t1, t2) for t1 in topics[0] for t2 in topics[1]]
    if sampleSize is not None and sampleSize < len(topicPairs):
        seed(rseed)
        topicPairs = sample(topicPairs, sampleSize)
    return topicPairs

def sampleTopicPairsFromModels(models, sampleSize=None, rseed=9342123, split=False):
    if not split:
        topics = [ t for m in models for t in m ]
    else:
        topics = [None]*2
        for i in [0, 1]: topics[i] = [ t for m in models[i] for t in m ]
    return sampleTopicPairs(topics, sampleSize, rseed, split)

def distancesFromTopicPairs(tpairs, dist):
    import gc
    res = []
    for i, p in enumerate(tpairs):
        t1, t2 = p
        res.append(dist(t1.vector, t2.vector))
        if i % 100 == 0: gc.collect()
    return res

prodbuildFolder = topic_models_folder
def topicDistanceDistribution(modelsFolder, corpus = 'uspol', distMeasure=cosine,
                              families = 'all', numT = [50, 100, 200],
                              numModels = 5, sampleSize=20000, rseed=9342123,
                              includeRef=False, refVsRest=False):
    from topic_coverage.modelbuild.modelset_loading import modelset1Families
    modellabel = 'models_%s_T_%s' % (families, ','.join(str(t) for t in numT))
    mlab = distMeasure.id if hasattr(distMeasure, 'id') else distMeasure.__name__
    label='%s_%s_%s_ref[%s]_refXrest[%s]'%(mlab, corpus, modellabel, includeRef, refVsRest)
    modelsets, modelCtx, _ = modelset1Families(corpus, numModels, modelsFolder, families, numT)
    allmodels = [m for mset in modelsets for m in mset]
    if includeRef:
        if corpus == 'uspol': refmodel = resolve(GTAR_REFMODEL)
        elif corpus == 'pheno': refmodel = resolve(PHENO_REFMODEL)
        refmodels = [refmodel]*int(includeRef)
    if includeRef:
        if refVsRest:
            tpairs = sampleTopicPairsFromModels([refmodels, allmodels], sampleSize, rseed,
                                                split=True)
        else:
            allmodels.extend(refmodels)
            tpairs = sampleTopicPairsFromModels(allmodels, sampleSize, rseed)
    else:
        tpairs = sampleTopicPairsFromModels(allmodels, sampleSize, rseed)
    distances = distancesFromTopicPairs(tpairs, distMeasure)
    print label
    print Stats(distances)
    basicValueDist(distances, save=True, boxplt=False, boxplotVals=True, title=label, bins=200,
                   xlabel='udaljenost', ylabel='relativni udio parova')

def plotAll(measures=[cosine, hellinger, l1norm], corpora=['uspol', 'pheno']):
    for corpus in corpora:
        for measure in measures:
            topicDistanceDistribution(prodbuildFolder, corpus=corpus, distMeasure=measure)

def plotModelFamilies(measures=[cosine, hellinger, l1norm]):
    for corpus in ['uspol', 'pheno']:
        for measure in measures:
            for mf in ['lda', 'alda', 'nmf', 'pyp']:
                topicDistanceDistribution(prodbuildFolder, corpus=corpus, distMeasure=measure,
                                          families=[mf])

def plotOneVsRest(model='nmf', measure=jensenShannon):
    ''' Distribution of distances for topics combined from model and one of other model types '''
    allmodels = ['lda', 'alda', 'nmf', 'pyp']
    models = [m for m in allmodels if m != model]
    for corpus in ['uspol', 'pheno']:
        for mf in models:
            topicDistanceDistribution(prodbuildFolder, corpus=corpus, distMeasure=measure,
                                          families=[mf, model])

from topic_coverage.topicmatch.feature_extraction import pearsonCorrTop, spearmanCorrTop, kendalltauCorrTop

if __name__ == '__main__':
    with topicCoverageContext():
        #topicDistanceDistribution(prodbuildFolder, corpus='uspol', distMeasure=hellinger)
        #topicDistanceDistribution(prodbuildFolder, corpus='pheno', distMeasure=hellinger)
        # for dm in [cosine, hellinger, l1norm, l2norm]:
        #     for mf in ['nmf', 'lda', 'alda', 'pyp']:
        #         topicDistanceDistribution(prodbuildFolder, corpus='pheno', numModels=5,
        #                                   families=[mf], distMeasure=dm, sampleSize=5000,
        #                                   includeRef=10)
        for dm in [cosine, hellinger, l1norm, l2norm]:
            for mf in ['nmf', 'lda', 'alda', 'pyp']:
                topicDistanceDistribution(prodbuildFolder, corpus='uspol', numModels=10,
                                          families=[mf], distMeasure=dm, sampleSize=20000,
                                          includeRef=1, refVsRest=True)
                        #plotAll([pearsonCorr, spearmanCorr, kendalltauCorr, pearsonCorrTop, spearmanCorrTop, kendalltauCorrTop])
        #plotAll([l1, jensenShannon], corpora=['uspol'])
        #plotModelFamilies([l1, jensenShannon])
        #plotOneVsRest('nmf', l1)
        #topicDistanceDistribution(prodbuildFolder, corpus='uspol', distMeasure=cosine)
        #plotAll()
        #topicDistanceDistribution(prodbuildFolder, corpus='uspol', distMeasure=cosine)