'''
Evaluation of coverage measures: supervised matcher-based, CTC, ...
'''

from topic_coverage.modelbuild.modelset_loading import modelset1Families, modelsetLoad
from topic_coverage.resources.modelsets import *
from topic_coverage.resources.pytopia_context import topicCoverageContext

def printModel(corpus='uspol', numModels=10, modelsFolder=prodModelsBuild,
               families='all', numT=[50, 100, 200]):
    msets, mctx, _ = modelset1Families(corpus, numModels, modelsFolder, families, numT)
    with mctx:
        # [ m for mset in msets for m in mset ]
        model = msets[0][0]
        print model.id
        for i, tid in enumerate(model.topicIds()):
            print 'TOPIC %d ; WORDS: %s' % (i, model.topic2string(tid, 10))
            titles = model.topTopicDocs(tid, 10)
            texts = model.topTopicDocs(tid, 10, titles=False)
            for i, t in enumerate(titles):
                print texts[i].id, t
            print

if __name__ == '__main__':
    # TODO !!! stari ref.model: from phenotype_context.phenotype_topics.construct_model import MODEL_ID
    with topicCoverageContext():
        printModel(corpus='uspol', families=['nmf'], numT=[200])