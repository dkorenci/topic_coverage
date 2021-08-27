from pytopia.adapt.artm import setup_logging

from pytopia.tools.IdComposer import IdComposer
from pytopia.resource.loadSave import pickleObject
from pytopia.context.ContextResolver import resolve, resolveIds
from pytopia.resource.ResourceBuilder import SelfbuildResourceBuilder
from pytopia.utils.file_utils.location import FolderLocation

import artm
from artm import BatchVectorizer

import os, shutil, tempfile
from os import path

class BatchVectorizerReswrap(IdComposer):
    '''
    Wraps artm.BatchVectorizer as a pytopia resource.
    '''

    def __init__(self, corpus, dictionary, text2tokens, batchSize=None):
        self.corpus, self.dictionary, self.text2tokens = resolveIds(corpus, dictionary, text2tokens)
        self.batchSize = batchSize
        self._batchVect = None
        IdComposer.__init__(self)

    def __del__(self): self.__clearTmpFolder()

    def resource(self): return self._batchVect, self._dictionary

    def build(self):
        self.__initTmpFolder()
        d, m = createBatchvectInput(self.corpus, self.dictionary, self.text2tokens)
        sz = self.batchSize if self.batchSize else m.shape[1]
        self._batchVect = BatchVectorizer(n_wd=m, vocabulary=d, batch_size=sz,
                                          data_format='bow_n_wd', target_folder=self._tmpFolder)
        self._dictionary = self._batchVect.dictionary

    def __batchesDir(self, folder):
        return path.join(folder, 'batches')

    def save(self, folder):
        # picke self without artm data
        bvect, d = self._batchVect, self._dictionary
        self._batchVect = None
        pickleObject(self, folder)
        self._batchVect, self._dictionary = bvect, d
        # save batches
        bfolder = self.__batchesDir(folder)
        if not path.exists(bfolder): os.mkdir(bfolder)
        for f in FolderLocation(self._tmpFolder).files():
            dest = path.join(bfolder, path.basename(f))
            shutil.copyfile(f, dest)

    def __initTmpFolder(self):
        self._tmpFolder = tempfile.mkdtemp()
        self._tmpFolder = os.path.join(self._tmpFolder, 'batchVectFolder')

    def __clearTmpFolder(self):
        if hasattr(self, '_tmpFolder') and self._tmpFolder:
            shutil.rmtree(self._tmpFolder, ignore_errors=True)

    def load(self, folder):
        bfolder = self.__batchesDir(folder)
        #batches = [path.basename(f) for f in FolderLocation(bfolder).files()]
        self._batchVect = BatchVectorizer(data_format='batches', data_path=bfolder)
        self._dictionary = artm.Dictionary()
        self._dictionary.gather(data_path=bfolder)

BatchVectorizerBuilder = SelfbuildResourceBuilder(BatchVectorizerReswrap)

# requires bow_corpus_builder
def createBatchvectInput(corpus, dict, txt2tok):
    '''
    Convert pytopia basic resources to artm.BatchVectorizer's "bow_n_wd" format,
    ie a bag-of-words ndarray of doc-word counts (transposed)
    and a dict - mapping of token -> index.
    :return: dict, matrix
    '''
    bow = resolve('bow_corpus_builder')(corpus, txt2tok, dict)
    mtx = bow.corpusMatrix(sparse=False)
    d = resolve(dict)
    mapdict = {d[tok]: tok for tok in d}
    return mapdict, mtx.T
