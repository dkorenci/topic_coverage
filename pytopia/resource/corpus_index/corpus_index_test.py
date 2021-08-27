from pytopia.testing.setup import *
from pytopia.testing.corpora import *

from pytopia.resource.corpus_index.CorpusIndex import CorpusIndex, CorpusIndexBuilder
from pytopia.context.ContextResolver import resolve
from pytopia.resource.loadSave import *

from os import path

def runtestCorpusIndex(tmpdir, corpus):
    '''
    Create corpus index, save, load, and test that it corresponds to the corpus.
    '''
    cind = CorpusIndexBuilder()(corpus)
    assert cind
    oldId = cind.id
    folder = path.join(tmpdir, cind.id)
    saveResource(cind, folder)
    cind = loadResource(folder)
    assert cind
    assert cind.id == oldId
    corpus = resolve(corpus)
    assert len(corpus) == len(cind)
    for txto in corpus:
        assert txto.id in cind
        assert cind[cind.id2index(txto.id)] == txto.id
    idset = set(txto.id for txto in corpus)
    idset1 = set(id for id in cind)
    assert idset == idset1
    idset2 = set(cind[i] for i in range(len(cind)))
    assert idset == idset2

def testCorpusIndexSmall(tmpdir):
    runtestCorpusIndex(str(tmpdir), corpus_uspol_small())

def testCorpusIndexMedium(tmpdir):
    runtestCorpusIndex(str(tmpdir), corpus_uspol_medium())
