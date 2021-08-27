'''
Tests and builds for models measuring referent topic sets
from us_politics and phenotype datasets.
'''

from topic_coverage.resources.pytopia_context import topicCoverageContext

from gtar_context.semantic_topics.construct_model import MODEL_ID as GTAR_REFMODEL
from phenotype_context.phenotype_corpus.construct_corpus import CORPUS_ID as PHENO_CORPUS
from phenotype_context.dictionary.create_4outof5_dictionary import DICT_ID as PHENO_DICT
from phenotype_context.phenotype_topics.construct_model import MODEL_ID as PHENO_MODEL

from pytopia.context.ContextResolver import resolve
from pyldazen.model.LdaGibbsTopicModel import LdaGibbsTopicModelBuilder
from pytopia.resource.FolderResourceCache import FolderResourceCache

from topic_coverage.settings import topicsize_measure_models

import numpy as np

def cachedLdaGibbsBuilder():
    ''' Create caced builder for models for measuring referent topics. '''
    return FolderResourceCache(LdaGibbsTopicModelBuilder, topicsize_measure_models,
                               'reftopic_measure_cache')

def buildReftopicMeasureModel(corpus, txt2tok, dict, refmodel, iter=1000):
    label = 'refmeasure[%s]' % corpus
    refmodel = resolve(refmodel)
    builder = cachedLdaGibbsBuilder()
    model = builder(corpus=corpus, dictionary=dict, text2tokens=txt2tok,
                    numTopics=20, gibbsIter=iter, idLabel=label,
                    fixedTopics=refmodel.topicMatrix(np.float64))
    return model

def buildUspoliticsRefmodel(iters=1000):
    # buildReftopicMeasureModel('us_politics', 'RsssuckerTxt2Tokens', 'us_politics_dict',
    #                           GTAR_REFMODEL, iters)
    from topic_coverage.modelbuild.modelbuild_docker_v1 import uspolBase
    buildReftopicMeasureModel(uspolBase['corpus'], uspolBase['text2tokens'], uspolBase['dictionary'],
                              GTAR_REFMODEL, iters)


def buildPhenotypeRefmodel(iters=1000):
    buildReftopicMeasureModel(PHENO_CORPUS, 'whitespace_tokenizer',
                              PHENO_DICT, PHENO_MODEL, iters)

def createModelsContext(modelFolders):
    ''' Add built models to global pytopia contex. '''
    from pytopia.context.Context import Context
    from pytopia.resource.loadSave import loadResource
    from pyutils.file_utils.location import FolderLocation as loc
    ctx = Context('testing_models_context')
    mfolders = []
    for mf in modelFolders:
        mfolders.extend(loc(mf).subfolders())
    for mf in mfolders:
        m = loadResource(mf)
        ctx.add(m)
    return ctx

def measuringModelsContext():
    return createModelsContext([topicsize_measure_models])

# requires corpus_topic_index_builder
def printTopTopicDocuments(topic, topDocs=20, corpus=None):
    '''
    :param topic: (modelId, topicId)
    :param topDocs:
    :return:
    '''
    mid, tid = topic
    ctiBuilder = resolve('corpus_topic_index_builder')
    if corpus is None: corpus = resolve(mid).corpus
    cti = ctiBuilder(corpus=corpus, model=mid)
    wtexts = cti.topicTexts(tid, top=topDocs)
    txtIds = [ id_ for id_, _ in wtexts ]
    id2weight = {id:w for id, w in wtexts}
    corpus = resolve(corpus)
    idTexts = corpus.getTexts(txtIds)
    print 'topic, model: %s, %s' % (tid, mid)
    for i, txto in enumerate(idTexts):
        print 'ID: %s, weight: %g' % (txto.id, id2weight[txto.id])
        print '  title: %s' % txto.title

def printTopDocs(m, t, printModel=False):
    measuringModelsContext() # todo use as py context (with keyword)
    if printModel:
        m = resolve(m)
        print m.corpus
        print m
    else:
        printTopTopicDocuments((m, t))

uspolMeasureModel = 'LdaGibbsTopicModel_T[20]_alpha[0.326797]_beta[0.01]_corpus[us_politics_textperline]_dictionary[us_politics_dict]_gibbsIter[1000]_idLabel[refmeasure[us_politics_textperline]]_rseed[88911]_text2tokens[whitespace_tokenizer]'
phenoMeasureModel = 'LdaGibbsTopicModel_T[20]_alpha[0.378788]_beta[0.01]_corpus[pheno_corpus1]_dictionary[pheno_dict1]_gibbsIter[1000]_idLabel[refmeasure[pheno_corpus1]]_rseed[88911]_text2tokens[whitespace_tokenizer]'

if __name__ == '__main__':
    with topicCoverageContext():
        #printTopDocs(uspolMeasureModel2, 104)
        #buildUspoliticsRefmodel(1000)
        buildPhenotypeRefmodel(1000)