from pytopia.corpus.Corpus import Corpus
from pytopia.corpus.Text import Text

from os import path

from pyutils.file_utils.location import FolderLocation as loc
from phenotype_context.settings import corporaRoot, corpusLabels

class PhenotypeCorpus(Corpus):

    def __init__(self, id, rootFolder, subfolders):
        self.id = id
        self._root = loc(rootFolder)
        self._subfolders = [sf for sf in set(subfolders)]

    def __initFolderIteration(self):
        f = loc(self._root(self._currFolder))
        self.__currFiles = f.files()
        #print '\n'.join(self.__currFiles)

    def __iter__(self):
        for self._currFolder in self._subfolders:
            self.__initFolderIteration()
            for f in self.__currFiles:
                fname = path.splitext(path.basename(f))[0]
                id = '%s_%s' % (self._currFolder, fname)
                #print fname
                assert '__' in fname
                sp = fname.split('__')[0]
                #print sp
                yield Text(id, loadText(f), filename=fname, species=sp)

def loadText(file, encoding='utf-8'):
    '''
    Load text file to string.
    :param file: path to file
    :return:
    '''
    import codecs
    with codecs.open(file, 'r', encoding) as f:
        res = ''.join(f.readlines())
    return res

def getCorpus():
    return PhenotypeCorpus('phenotype_corpus', corporaRoot, corpusLabels)

def testCorpus():
    corpus = getCorpus()
    print corpus.id, len(corpus)
    for i, txto in enumerate(corpus):
        if i == 100: break
        print txto.id
        print txto.text
        print

if __name__ == '__main__':
    testCorpus()