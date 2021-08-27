from topic_coverage.modelbuild.modelbuild_docker_v1 import *
from topic_coverage.resources.pytopia_context import topicCoverageContext

def testBuild(cycles=250, burnin=50, folder='/datafast/topic_coverage/modelbuild/test_alda/'):
    params = createParamset('uspol', 'lda-asym', 1, 200, 42110)
    if cycles and burnin:
        for p in params:
            p['C'] = p['Cme'] = cycles
            p['burnin'] = p['Bme'] = burnin
    print params
    buildModels(params, folder)

def prodParamsBuild(T=50, numModels=1, folder='/datafast/topic_coverage/modelbuild/test_alda2/', rseed=84153):
    params = createParamset('uspol', 'lda-asym', numModels, topics=T, rseed=rseed)
    for p in params:
        print p
    buildModels(params, folder)

if __name__ == '__main__':
    with topicCoverageContext():
        #testBuild(300, 50)
        #prodParamsBuild(T=100, numModels=5, rseed=5991)
        prodParamsBuild(T=100, numModels=5, rseed=128811)