from topic_coverage.resources import pytopia_context

from topic_coverage.experiments.clustered_models_v0 import \
    coverageScoringExperiment, coverMetrics
from topic_coverage.modelbuild.modelbuild_iter2 import *

from pytopia.context.ContextResolver import resolve
from pytopia.measure.avg_nearest_dist import AverageNearestDistance, TopicCoverDist
from pytopia.measure.topic_distance import cosine as cosineDist

coverMetrics = [
    AverageNearestDistance(cosineDist, pairwise=False),
    TopicCoverDist(cosineDist, 0.4)
]

def coverageTestHcaPypUsPol(params):
    target = resolve('gtar_themes_model')
    models = buildLoadTestModels(params, 'pyp')
    for m in models: print m.id
    coverageScoringExperiment(target, [models], coverMetrics)

if __name__ == '__main__':
    #coverageTestHcaPypUsPol(pypQuickTrain)
    #coverageTestHcaPypUsPol(pypParams)
    coverageTestHcaPypUsPol(pypV2Params2)
