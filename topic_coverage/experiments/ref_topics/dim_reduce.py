from pytopia.context.ContextResolver import resolve
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

from topic_coverage.experiments.ref_topics.measuring_models import \
        uspolMeasureModel1, uspolMeasureModel2, phenoMeasureModel1, \
        measuringModelsContext
from topic_coverage.experiments.ref_topics.measuring_tools import \
        topicSizes, topicCoverage, topicsEqualCosine04
from topic_coverage.resources import pytopia_context
from stat_utils.utils import Stats

def plotTopicModel(tmodel, topics=None, color=None, rndseed=6682):
    tmodel = resolve(tmodel)
    if topics is None: topics = tmodel.topicIds()
    elif isinstance(topics, int): topics = range(topics, tmodel.numTopics())
    tmat = np.empty((len(topics), len(tmodel.topicVector(topics[0]))))
    cvec = [None]*len(topics) if color else 'b'
    # create topic matrix and colors array that matches topic indices
    for i, ti in enumerate(topics):
        tmat[i] = tmodel.topicVector(ti)
        cvec[i] = color[ti]
    # dimreduce
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000,
                method='exact', learning_rate=500, metric='cosine', random_state=rndseed)
    dimr = tsne.fit_transform(tmat)
    # create outlier indicator array
    s = Stats(cvec); iqr = s.q75-s.q25; k = 1.5
    isout = lambda c: c < s.q25-iqr*k or c > s.q75+iqr*k
    cvec = np.array(cvec); outs = np.array([isout(c) for c in cvec])
    fig, axes = plt.subplots()
    # plot non-outliers
    sc = axes.scatter(dimr[~outs][:,0], dimr[~outs][:,1],
                      c=cvec[~outs], cmap=plt.cm.get_cmap('RdYlBu'))
    axes.scatter(dimr[outs][:,0], dimr[outs][:,1],
                 c='black', marker='x')
    plt.colorbar(sc)
    plt.show()

from topic_coverage.modelbuild.modelbuild_iter1 import addModelsToGlobalContext
from topic_coverage.experiments.clustered_models_v0 import baseModelSet, modelFolders
def modelSet(corpus, model, numTopics, numModels):
    return baseModelSet('', modelFolders[corpus, model, numTopics],
                         numModels, returnIds=True)

def plotDimreducedRefmodel(refmodel, color='size', covModels=None):
    measuringModelsContext() # todo use as py context (with keyword)
    refmodel = resolve(refmodel)
    topics = [t.topicId for t in refmodel.fixedTopics()]
    if color == 'size':
        c = topicSizes(refmodel, topics)
    elif color == 'cover':
        addModelsToGlobalContext()
        c = topicCoverage(refmodel.fixedTopics(), covModels, topicsEqualCosine04)
    plotTopicModel(refmodel, topics, c)

if __name__ == '__main__':
    #plotDimreducedRefmodel(uspolMeasureModel2)
    plotDimreducedRefmodel(uspolMeasureModel2, color='cover',
                           covModels=modelSet('us_politics', 'nmf', 100, 15))
    #plotTopicModel(uspolMeasureModel2, topics=20, color=topicSizes(uspolMeasureModel2, 20))
    #plotTopicModel(phenoMeasureModel1, topics=20, color=topicSizes(phenoMeasureModel1, 20))

