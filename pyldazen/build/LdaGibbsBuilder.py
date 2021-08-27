#import pyximport;
#py_importer, pyx_importer = pyximport.install()
from pyldazen.build.LdaGibbsInferer import *
#pyximport.uninstall(py_importer, pyx_importer)

from pyldazen.build.LdaGibbsBuildData import LdaGibbsBuildData

from croelect.resources.resource_builds import resourceBuilder

from logging_utils.setup import createLogger

logger = createLogger(__name__)

class LdaGibbsBuilder():
    '''
    Factory interface for building LDA models with gibbs sampling.
    '''
    def __call__(self, data):
        '''
        Invoke cython build and return results.
        :param data: LdaGibbsBuildData instance
        :return: LdaModel instance
        '''
        inferer = LdaGibbsBuilder.createInferer(data)
        inferer.infer()
        # fig, axes = plt.subplots(3, 1)
        # topicProps, topicPropsNorm = [1,2,3], [4,5,6]
        # axes[0].hist([log(x, 10) for x in topicProps], 1000)
        # axes[1].hist([log(x, 10) for x in topicPropsNorm], 1000)
        # axes[2].hist(sampledTopics, 100)
        # plt.show()

    @staticmethod
    def createInferer(data):
        '''
        Create and return LdaGibbsInferer from data.
        '''
        return LdaGibbsInferer(data.T, data.alpha, data.beta, data.documents,
                                  data.docFormat, data.Tf, data.fixedTopics)

def testBuilder():
    '''
    Check if the builder compiles and runs without errors.
    '''
    logger.info('starting test lda run')
    rb = resourceBuilder()
    corpus = rb.loadBowCorpus('iter0_cronews_agendadedup2')
    corpus = corpus[:200]
    data = LdaGibbsBuildData(100, None, corpus, 1.0, 0.01)
    LdaGibbsBuilder()(data)