'''
Calculate coverages of topic sets characterized by 'size' - frequent and rare topic etc.
'''

from pytopia.context.ContextResolver import resolve
from topic_coverage.resources.pytopia_context import topicCoverageContext
from topic_coverage.experiments.correlation.experiment_runner import refmod

from topic_coverage.experiments.concept_types.subset_topic_model import SubsetTopicModel
from topic_coverage.experiments.ref_topics.measuring_models import \
        uspolMeasureModel, phenoMeasureModel, measuringModelsContext
from topic_coverage.experiments.ref_topics.measuring_tools import topicSizes

import sys

def subsetModelRuntest(corpus='uspol'):
    model = resolve(refmod(corpus))
    tids = model.topicIds()
    eventopics = [ti for i, ti in enumerate(tids) if i % 2 == 0]
    stm = SubsetTopicModel('subsetmodel', model, eventopics)
    print stm.id
    print stm
    print model
    #print stm.topicMatrix()
    #print stm.corpusTopicVectors()

uspolSegQuartIter1000 = {'q1':(0, 67.1), 'q2': (67.1, 112.1), 'q3': (112.1, 167.1), 'q4': (167.1, sys.maxint)}
phenoSegQuartIter1000 = {'q1':(0, 9.1), 'q2': (9.1, 20.1), 'q3': (20.1, 42.1), 'q4': (42.1, sys.maxint)}

def getSubRefmodelBySize(corpus, segment, segmentDefs, topicIdOffset=20, topicInDocThresh=0.1):
    if corpus == 'uspol': measureModel = uspolMeasureModel
    elif corpus == 'pheno': measureModel = phenoMeasureModel
    range = segmentDefs[segment]
    label = 'topicsizeSubset_%s_%d-%d' % (segment, range[0], range[1])
    tsize = topicSizes(measureModel, topicIdOffset, topicInDocThresh)
    topicIds = [ tid - topicIdOffset for tid, sz in tsize.iteritems() if range[0] <= sz < range[1] ]
    model = SubsetTopicModel(label, refmod(corpus), topicIds, origTopicIds=True)
    return model

def measureCoverage(corpus, segment, segmentDefs, topicIdOffset=20, topicInDocThresh=0.1):
    from topic_coverage.experiments.coverage.experiment_runner import evaluateCoverage
    from pytopia.context.Context import Context
    refmod = getSubRefmodelBySize(corpus, segment, segmentDefs, topicIdOffset, topicInDocThresh)
    #print refmod
    with Context('refmod_ctx', refmod):
        evaluateCoverage(eval='metrics', corpus=corpus, refmodel=refmod, numModels=10, bootstrap=20000)

def topicsetSizes(segmentDefs, thresh, topicIdOffset=20):
    for corpus in ['uspol', 'pheno']:
        if corpus in segmentDefs:
            for sz in ['q1', 'q2', 'q3', 'q4']:
                refmod = getSubRefmodelBySize(corpus, sz, segmentDefs[corpus], topicIdOffset, thresh[corpus])
                print corpus, sz, '[%d-%d]'%segmentDefs[corpus][sz], refmod.numTopics()

def measureCovsBySize(segmentDefs, thresh, topicIdOffset=20):
    for corpus in ['uspol', 'pheno']:
        if corpus in segmentDefs:
            print '*********** %s ************' % corpus.upper()
            for sz in ['q1', 'q2', 'q3', 'q4']:
                print '--------', sz, '[%d-%d]'%segmentDefs[corpus][sz]
                measureCoverage(corpus, sz, segmentDefs=segmentDefs[corpus],
                                topicIdOffset=topicIdOffset, topicInDocThresh=thresh[corpus])
            print

if __name__ == '__main__':
    with topicCoverageContext():
        with measuringModelsContext():
            #topicsetSizes(segmentDefs={'uspol':uspolSegQuartIter1000, 'pheno':phenoSegQuartIter1000},
            #               thresh={'uspol':0.1, 'pheno':0.05})
            # topicsetSizes(segmentDefs={'pheno':phenoSegQuart},
            #               thresh={'uspol':0.1, 'pheno':0.05})
            #
            measureCovsBySize(segmentDefs={'uspol':uspolSegQuartIter1000}, thresh={'uspol': 0.1, 'pheno': 0.05})
            #measureCovsBySize(segmentDefs={'pheno':phenoSegQuartIter1000}, thresh={'uspol':0.1, 'pheno':0.05})