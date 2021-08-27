import pytopia.testing.setup

from pytopia.testing.utils import flattenParamList
from pytopia.testing.corpora import *
from pytopia.resource.builder_cache.ResourceBuilderCache import ResourceBuilderCache
from pytopia.adapt.scikit_learn.nmf.adapter import SklearnNmfBuilder

import os
from os import path

nmfbuildopts = [
    {
        'corpus': corpus_uspol_small(), 'dictionary': 'us_politics_dict',
        'text2tokens': 'english_word_tokenizer', 'T': [5, 7, 10, 13, 15]
    },
]

def runBuilding(folder, builder, opts, dummyHash):
    cache = ResourceBuilderCache(builder, folder, memCache=True, dummyHash=dummyHash)
    res = {}
    for i, o in enumerate(opts):
        res[i] = cache(**o)
    # rebuild with same cache, testing mem cache
    for i, o in enumerate(opts):
        cres = cache(**o)
        # todo assert equality of objects, when __equals__ is implemented for built classes
        assert cres.id == res[i].id
    # rebuild with same new cache, forcing loading from disk
    cache = ResourceBuilderCache(builder, folder, dummyHash=dummyHash)
    for i, o in enumerate(opts):
        cres = cache(**o)
        #todo assert equality of objects, when __equals__ is implemented for built classes
        assert cres.id == res[i].id

def testBuilderCache(tmpdir):
    testDir = str(tmpdir)
    testDir = '/datafast/topic_coverage/test_hashed_cache/'
    f1, f2 = path.join(testDir, 'dummyHash'), path.join(testDir, 'normalHash')
    os.mkdir(f1); os.mkdir(f2)
    runBuilding(f1, SklearnNmfBuilder(), flattenParamList(nmfbuildopts), True)
    runBuilding(f2, SklearnNmfBuilder(), flattenParamList(nmfbuildopts), False)

if __name__ == '__main__':
    pass