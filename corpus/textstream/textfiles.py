from os import listdir
from os.path import isfile, join

from corpus.text import Text

def filesInFolder(folder):
    '''
    return a list of non-folder files in a folder 
    '''
    return [ file for file in listdir(folder) if isfile(join(folder,file)) ]    

def writeTextsToFolder(texts, folder):
    '''
    write (iterable of) texts in id-named files
    '''
    for txt in texts :
        txt.toFile(join(folder, str(txt.id)+".txt"))

class FileStream():
    '''
    Create stream of text Objects from a folder with files containing texts
    '''
    def __init__(self, folder):
        self.folder = folder
                    
    def __iter__(self):
        '''
        iterate over files in the folder, return Text objects
        '''
        for file in filesInFolder(self.folder) :
            yield Text.fromFile(join(self.folder,file))
            
    def __len__(self):
        if not hasattr(self, 'length'):
            # cache the corpus length
            self.length = sum(1 for _ in self)
        return self.length           
  