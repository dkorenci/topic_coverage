import wikivectors
from wikivectors import WordToVectorDiskMap

javaVMInitialized = False
def initVM():
    '''initialize java virtual machine'''
    global javaVMInitialized
    if not javaVMInitialized:
        try:
            wikivectors.initVM(vmargs=['-Djava.awt.headless=true',
                                    '-Dsun.arch.data.model=64', '-Dsun.cpu.endian=little'])
        except:
            wikivectors.initVM()
        javaVMInitialized = True

initVM()

def testVectors():
    vectors = WordToVectorDiskMap("/datafast/wiki_esa/wiki-esa-terms.txt",
                "/datafast/wiki_esa/wiki-esa-vectors.txt", "esa", True, True,
                "/datafast/wiki_esa/wikivectors_cache/")
    for w in ['trump', 'president']:
        vec = vectors.getWordVector(w)
        print vec

if __name__ == '__main__':
    testVectors()