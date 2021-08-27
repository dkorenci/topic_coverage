from pytopia.context.ContextResolver import resolve
from topic_coverage.resources.pytopia_context import topicCoverageContext

from phenotype_context.phenotype_topics.construct_model import MODEL_ID as PHENO_MODEL
from gtar_context.semantic_topics.construct_model import MODEL_ID as GTAR_MODEL
from topic_coverage.experiments.ref_topics.measuring_tools import topicSizes

from topic_coverage.experiments.ref_topics.measuring_models import *
import hypertools as hyp

def plotModelTopics(model):
    ''' Dim. reduce and plot all topics of a topic model. '''
    model = resolve(model)
    tmx = model.topicMatrix()
    hyp.plot(tmx, '.', ndims=3, reduce='SpectralEmbedding',
             #cluster='AgglomerativeClustering',
             n_clusters=10)

def plotTopicsizeDistribution(model, topics=20, thresh=0.1):
    from stat_utils.utils import Stats
    from stat_utils.plots import basicValueDist
    tsize = topicSizes(model, topics, thresh)
    sizes = tsize.values()
    print Stats(sizes)
    basicValueDist(sizes)
    #model = resolve(model); corpus = resolve(model.corpus)

if __name__ == '__main__':
    with topicCoverageContext():
        with measuringModelsContext():
            #plotModelTopics(GTAR_MODEL)
            #plotModelTopics(PHENO_MODEL)
            plotTopicsizeDistribution(uspolMeasureModel)
            #plotTopicsizeDistribution(phenoMeasureModel3, thresh=0.05)