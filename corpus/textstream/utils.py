import numpy as np

class Text2TokensStream():
    '''
    Utility class to create a stream of tokenized texts from text stream and a tokenizer 
    '''
    
    def __init__(self, stream, tok):
        self.text_stream = stream
        self.text2tokens = tok
        
    def __iter__(self):
        for txt in self.text_stream :
            t = self.text2tokens(txt.text)
            if len(t) > 0 : yield t
            
    def __len__(self):
        if not hasattr(self, 'length'):
            # cache the corpus length
            if hasattr(self.text_stream,'__len__'): length = len(self.text_stream)
            else: length = sum(1 for _ in self)
            self.length = length
        return self.length
    
class BowStream():
    '''
    Create a stream of bows (documents). This is gensim corpus, difference from
    TextStream is that it accepts its own dictionary.    
    '''            
    def __init__(self, token_stream, dictionary):
        '''
        Construct from iterable over list of tokens and gensim dictionary.
        '''
        self.dictionary = dictionary
        self.token_stream = token_stream

    def __iter__(self):
        for token_list in self.token_stream :
            yield self.dictionary.doc2bow(token_list, allow_update=False)
        
    def __len__(self):
        return len(self.token_stream)    

def bowstreamToArray(bowstream):
    initSize = 100
    array = np.empty(initSize, dtype=np.object); pos = 0
    for bow in bowstream:
        if pos+1 > len(array):
            array = np.resize(array, len(array)*2)
        array[pos] = bowToArray(bow)
        pos += 1
    return np.resize(array, pos)

def bowToArray(bow):
    #array = np.empty(len(bow), dtype = np.object)
    array = np.empty(len(bow), dtype = 'u4,u4')
    for i in range(len(bow)): array[i] = bow[i]
    return array

def bowCorpusesEqual(bows1, bows2):
    if len(bows1) != len(bows2) : return False
    else: print 'lengths equal'
    for i in range(len(bows1)):
        bow1 = bows1[i]; bow2 = bows2[i]
        if len(bow1) != len(bow2) :
            print 'lenghth %d not equal: %d %d' % (i, len(bow1), len(bow2))
            return False
        for j in range(len(bow1)):
            if bow1[j][0] != bow2[j][0] or bow1[j][1] != bow2[j][1] :
                print 'bow1: ' + str(bow1[j]) + ' bow2: ' + str(bow2[j])
                return False



    return True
