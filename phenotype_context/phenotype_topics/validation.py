from phenotype_context.phenotype_topics.construct_model import modelBuildContext, loadModel, \
    DICT_ID, CORPUS_ID, MODEL_DOCS_ID

from pytopia.resource.loadSave import loadResource
from pyldazen.model.LdaGibbsTopicModel import LdaGibbsTopicModelBuilder

import numpy as np

def ldaDoctopicsInfer(iter=50, T=30):
    builder = LdaGibbsTopicModelBuilder
    model = loadModel(MODEL_DOCS_ID)
    phenoTopics = model.topicMatrix().astype(np.double)
    with modelBuildContext():
        model = builder(corpus=CORPUS_ID, dictionary=DICT_ID, text2tokens='whitespace_tokenizer',
                numTopics=T, fixedTopics=phenoTopics, gibbsIter=iter, idLabel='phenoLdaDoctopic')
        model.save('lda_doctopic_%d'%iter)
        print model

def compareDoctopicsWithLda(iter, T=30):
    from pytopia.measure.topic_distance import cosine
    from pyutils.stat_utils.utils import Stats
    origModel = loadModel(MODEL_DOCS_ID)
    ldaModel = loadResource('lda_doctopic_%d'%iter)
    def topicId2Docweights(model):
        dt = model.corpusTopicVectors()
        res = {}
        for i, tid in enumerate(model.topicIds()):
            res[i] = dt[:, i]
        return res
    origDvec = topicId2Docweights(origModel)
    ldaDvec = topicId2Docweights(ldaModel)
    matches = 0
    dists = []; dists2nd=[]
    for ti, dvec in origDvec.iteritems():
        mini = None; mind = 2.0; mind2 = 2.0
        for tj, dvec2 in ldaDvec.iteritems():
            if cosine(dvec, dvec2) < mind:
                mind2 = mind
                mind = cosine(dvec, dvec2)
                mini = tj
        dists.append(mind)
        dists2nd.append(mind2)
        #print '%g'%mind, '%g'%mind2
        if ti == mini-T: matches += 1
        else:
            print ti, mini - T, mini, '%g' % mind
            print origModel.topic2string(ti)
            print ldaModel.topic2string(mini)
            print
    origT = len(origDvec)
    print 'num.matches: %d out of %d, %g' % (matches, origT, float(matches)/origT)
    print 'min. distances: ', Stats(dists)
    print '2nd. distances: ', Stats(dists2nd)

if __name__ == '__main__':
    #ldaDoctopicsInfer(200)
    with modelBuildContext(): compareDoctopicsWithLda(200)
