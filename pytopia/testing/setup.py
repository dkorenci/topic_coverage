import pytest
from os import path, listdir

from pytopia.corpus.text.TextPerLineCorpus import TextPerLineCorpus
from pytopia.context.Context import Context
from pytopia.context.GlobalContext import GlobalContext
from pytopia.resource.loadSave import loadResource
from pytopia.utils.load import listSubfolders
from pytopia.resource.FolderResourceCache import FolderResourceCache

from os import path, mkdir
import shutil

testingDir = path.dirname(__file__)
corporaDir = path.join(testingDir, 'corpora')
dictDir = path.join(testingDir, 'dictionaries')

tmpDir = path.join(testingDir, 'tmp')
if path.exists(tmpDir):
    shutil.rmtree(tmpDir); mkdir(tmpDir)

def createContextWithCorpora(folder):
    '''
    Create context with one TextPerLineCorpus created for each .txt file in the folder.
    '''
    ctx = Context(None)
    for f in listdir(folder):
        if f.endswith('.txt'):
            p = path.join(folder, f) # full pat
            id = f[:-4] # remove extension from fname
            ctx.add(TextPerLineCorpus(p, id=id))
    return ctx

def createToyCorporaContext():
    '''
    Create context containing all toy corpora, as well as their dictionaries.
    '''
    from pytopia.testing.corpora import toyCorpusUsPolitics
    corpora = [ toyCorpusUsPolitics() ]
    ctx = Context('toy_corpora_context')
    for c in corpora:
        ctx.add(c)
        ctx.add(c.dictionary)
    return ctx

def dictionaryContext(folder):
    '''
    Create Context with dictionaries from folders,
    load subfolders as pytopia resources.
    '''
    ctx = Context(None)
    for subf in listSubfolders(folder):
        ctx.add(loadResource(subf))
    return ctx

def tokenizerContext():
    '''Create context with standard tokenizers.'''
    from pytopia.nlp.text2tokens.regexp import alphanumTokenizer, wordTokenizer
    from pytopia.nlp.text2tokens.gtar.text2tokens import RsssuckerTxt2Tokens
    ctx = Context(None)
    ctx.add(alphanumTokenizer())
    ctx.add(wordTokenizer())
    ctx.add(RsssuckerTxt2Tokens())
    return ctx

def modelContext():
    '''Create context with test topic models. '''
    from pytopia.testing.topic_models.load import loadTestModels
    ctx = Context(None)
    for m in loadTestModels(): ctx.add(m)
    return ctx

def builderContext():
    from pytopia.resource.builders_context import basicBuildersContext, cachedResourceBuilder
    from pytopia.utils.file_utils.location import FolderLocation as loc
    ctx = basicBuildersContext(tmpDir)
    from pytopia.adapt.artm.batch_vectorizer import BatchVectorizerBuilder
    ctx.add(cachedResourceBuilder(BatchVectorizerBuilder,
                                  loc(tmpDir)('artm_batch_vectorizer'),
                                  id='artm_batch_vectorizer_builder'))
    return ctx

@pytest.fixture(scope="session")
def testingContextPhase0():
    '''
    Context with resources that do not require any resources
      already in the global context to be created.
    All the resources created in subsequent phases can
     fetch by id the resources created in previous phases.
    :return:
    '''
    ctx = Context('context_pytopia_testing')
    ctx.merge(createContextWithCorpora(corporaDir))
    ctx.merge(dictionaryContext(dictDir))
    ctx.merge(tokenizerContext())
    ctx.merge(builderContext())
    ctx.merge(modelContext())
    return ctx

def testingContextPhase1():
    ctx = Context('context_pytopia_testing_phase1')
    ctx.merge(createToyCorporaContext())
    return ctx

def testingContext():
    # todo implement phases with context stacking
    ctx = Context('pytopia_testing_context')
    ctx0 = testingContextPhase0()
    ctx.merge(ctx0)
    with ctx0:
        ctx1 = testingContextPhase1()
        ctx.merge(ctx1)
    # with ctx: print ctx
    return ctx

def printGlobalContext():
    print GlobalContext.get()

GlobalContext.set(testingContext())

#printGlobalContext()

if __name__ == '__main__':
    printGlobalContext()
