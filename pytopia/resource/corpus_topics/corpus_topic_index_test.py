from pytopia.testing.setup import *
from pytopia.context.ContextResolver import resolve
from pytopia.resource.corpus_topics.CorpusTopicIndex import \
        CorpusTopicIndexBuilder as Builder
from pytopia.resource.loadSave import *

from pytopia.utils.logging_utils.setup import createLogger
from os import path

def ctiBuildSaveLoadCompare(builder, opts, dir):
    '''
    Create CorpusTopicIndex, save, load, and compare original and loaded index.
    '''
    cti = builder(**opts); saveDir = path.join(dir, cti.id)
    saveResource(cti, saveDir)
    ctiLoad = loadResource(saveDir)
    print cti.id
    assert len(cti) == len(ctiLoad)
    assert cti.id == ctiLoad.id
    corpus = resolve(opts['corpus'])
    for txto in corpus:
        assert cti.textTopics(txto.id) == ctiLoad.textTopics(txto.id)

def topicTextQuery(builder, opts):
    '''
    Create CorpusTopicIndex, run topicTexts queries
    for model topics and validate results.
    '''
    cti = builder(**opts)
    model = resolve(opts['model'])
    def compareIdValues(a1, a2):
        m1 = { id:val for id, val in a1 }
        m2 = { id:val for id, val in a2 }
        assert m1 == m2
    import numpy as np
    for ti in model.topicIds():
        tt = cti.topicTexts(ti, sorted=None)
        tta = cti.topicTexts(ti, sorted='asc')
        ttd = cti.topicTexts(ti, sorted='desc')
        assert len(tt) == len(tta)
        assert len(tt) == len(ttd)
        compareIdValues(tt, tta)
        compareIdValues(tt, ttd)
        # reversed value column on desc must be value column on asc
        assert np.array_equal(tta[:, 1], ttd[::-1][:, 1])
        top = 20 if 20 < len(tt) else len(tt)/2
        ttatop = cti.topicTexts(ti, sorted='asc', top=top)
        assert np.array_equal(ttatop[:, 1], tta[:, 1][:top])
        assert np.array_equal(ttatop[:, 1], np.sort(tt[:, 1])[:top])
        ttdtop = cti.topicTexts(ti, sorted='desc', top=top)
        assert np.array_equal(ttdtop[:, 1], -np.sort(-tt[:, 1])[:top])
        assert np.array_equal(ttdtop[:, 1], ttd[:, 1][:top])

def testCorpusTopicIndexSmall1(tmpdir):
    '''Test build on small corpus, using dict and txt2tokens from the model. '''
    createLogger(testCorpusTopicIndexSmall1.__name__).info(' ')
    buildOpts = {'corpus': 'us_politics_dedup_[100]_seed[1]',
         'model': 'model1', 'dictionary': None, 'txt2tokens': None}
    builder = Builder()
    ctiBuildSaveLoadCompare(builder, buildOpts, str(tmpdir))

def testCorpusTopicIndexSmall2(tmpdir):
    '''Test build on small corpus, explicitly setting dict and txt2tokens. '''
    createLogger(testCorpusTopicIndexSmall2.__name__).info(' ')
    buildOpts = {'corpus': 'us_politics_dedup_[100]_seed[1]', 'model': 'model1',
                 'dictionary': 'us_politics_dict', 'txt2tokens': 'english_alphanum_tokenizer'}
    builder = Builder()
    ctiBuildSaveLoadCompare(builder, buildOpts, str(tmpdir))

def testCorpusTopicIndexBig(tmpdir):
    '''Test build on big corpus, using dict and txt2tokens from the model. '''
    createLogger(testCorpusTopicIndexBig.__name__).info(' ')
    buildOpts = {'corpus': 'us_politics_dedup_[2500]_seed[3]',
                 'model': 'model1', 'dictionary': None, 'txt2tokens': None}
    builder = Builder()
    ctiBuildSaveLoadCompare(builder, buildOpts, str(tmpdir))

def testTopicTextQuerySmall():
    '''Test topicTexts query on small corpus. '''
    createLogger(testTopicTextQuerySmall.__name__).info(' ')
    buildOpts = {'corpus': 'us_politics_dedup_[100]_seed[1]',
                 'model': 'model1', 'dictionary': None, 'txt2tokens': None}
    builder = Builder()
    topicTextQuery(builder, buildOpts)

def testTopicTextQueryBig():
    '''Test topicTexts query on big corpus. '''
    createLogger(testTopicTextQueryBig.__name__).info(' ')
    buildOpts = [
                 # {'corpus': 'us_politics_dedup_[:2500]_seed[3]',
                 # 'model': 'model1', 'dictionary': None, 'txt2tokens': None},
                 {'corpus': 'us_politics_dedup_[2500]_seed[3]',
                  'model': 'nmf_model1', 'dictionary': None, 'txt2tokens': None},
                 ]
    builder = Builder()
    for opts in buildOpts:
        topicTextQuery(builder, opts)

def testCrossCorpus(tmpdir):
    '''Test inference of corpus-topic vectors on corpora not used to build the models'''
    ctiBuildOpts = [ {'corpus':  'us_politics_dedup_[100]_seed[1]', 'dictionary': None,
                        'txt2tokens': None}, ]
    models = ['model1', 'nmf_model1']
    builder = Builder()
    for m in models:
        for opts in ctiBuildOpts:
            opts['model'] = m
            ctiBuildSaveLoadCompare(builder, opts, str(tmpdir))
            topicTextQuery(builder, opts)

if __name__ == '__main__':
    #testTopicTextQueryBig()
    testCrossCorpus('/datafast/topic_coverage/tmp/')

