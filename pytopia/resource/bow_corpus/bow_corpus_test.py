from pytopia.testing.setup import *

from pytopia.resource.bow_corpus.bow_corpus import BowCorpusBuilder
from pytopia.resource.loadSave import loadResource
from pytopia.context.ContextResolver import ContextResolver

def compareBowWithCorpus(builder, corpus, txt2tok, dict, folder):
    '''
    Build BowCorpus, save, load, and compare, text by text
    with text from corpus converted to bow list.
    :param corpus: iteration order has to be fixed over multiple iterations
    '''
    # create bow corpus, save, load
    bow = builder(corpus, txt2tok, dict)
    bowFolder = str(folder.join(bow.id))
    bow.save(bowFolder)
    bow = loadResource(bowFolder)
    assert bow
    # compare loaded with texts from corpus, text by text
    corpus, txt2tok, dict = ContextResolver().resolve(corpus, txt2tok, dict)
    assert len(bow) == len(corpus)
    texts = { txto.id:txto for txto in corpus }
    cind = bow.corpus_index
    for i, bowTxt in enumerate(bow):
        txto = texts[cind[i]]
        # convert bow list to python list, for comparison
        bowTxt = [(wordId, wordCnt) for wordId, wordCnt in bowTxt ]
        origBowTxt = dict.tokens2bow(txt2tok(txto.text))
        assert sorted(bowTxt) == sorted(origBowTxt)

from pytopia.testing.corpora import *
def runtestBowCorpus(tmpdir, corpus=None):
    builder = BowCorpusBuilder()
    compareBowWithCorpus(builder, corpus,  'english_word_tokenizer',
                         'us_politics_dict', tmpdir)
    #compareBowWithCorpus()

def testBowCorpusSmall(tmpdir):
    runtestBowCorpus(tmpdir, corpus_uspol_small())

def testBowCorpusMedium(tmpdir):
    runtestBowCorpus(tmpdir, corpus_uspol_medium())

