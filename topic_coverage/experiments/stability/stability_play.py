from pytopia.evaluation.stability.modelset import ModelsetStability
from pytopia.evaluation.stability.modelmatch_bipartite import ModelmatchBipartite, TopicmatchVectorsim
from pytopia.measure.topic_similarity import cosine as cosineSim

from topic_coverage.resources.pytopia_context import topicCoverageContext
from topic_coverage.modelbuild.modelset_loading import modelset1Families
from topic_coverage.resources.modelsets import *

from topic_coverage.experiments.stability.stability_factory import ctcStability

def stabilityExperiment0(corpus='uspol', numModels=5,
                         modelsFolder=prodModelsBuild, families=['pyp'], numT=[300]):
    # load modelset
    # init bipartite modelmatch using TopicmatchWordsim
    # init modelset-match using modelmatch
    # run modelset stability score
    # test on diff model classes, correlate with cov
    # caching & id-ability, ...
    msets, mctx, _ = modelset1Families(corpus, numModels, modelsFolder, families, numT)
    #s = ModelsetStability(ModelmatchBipartite(TopicmatchVectorsim(cosineSim)))
    s = ctcStability()
    print s.id
    for mset in msets:
        print mset[0].id
        print s(mset)
    pass

if __name__ == '__main__':
    with topicCoverageContext():
        stabilityExperiment0()