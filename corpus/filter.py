from corpus import Corpus
from text import Text

class FilteredCorpus(Corpus):
    ' corpus created by filtering texts in another corpus by a specific criteria '
    #2do implement filtered id-fetching methods

    def __init__(self, corpus, filter, id = None):
        self.corpus = corpus
        self.filter = filter
        self.id = id

    def corpusId(self):
        if self.id is not None: return self.id
        else: return self.filter.getId()+'_'+self.corpus.corpusId()

    def __iter__(self):
        for txtobj in self.corpus:
            if not self.filter(txtobj): yield txtobj

    def getTexts(self, id_list):
        'get texts that pass the filter '
        texts = self.corpus.getTexts(id_list)
        return [ txtid for txtid in texts if not self.filter(txtid[1]) ]

    def getText(self, id):
        'get text if it passes filter, None otherwise'
        txto = self.corpus.getText(id)
        if self.filter(txto) : return None
        else: return txto

    def getFeeds(self, txto):
        if self.filter(txto) : return
        else: self.corpus.getFeeds(txto)

    def getOutlets(self, txto):
        if self.filter(txto) : return
        else: self.corpus.getOutlets(txto)

class MultiFilteredCorpus(Corpus):
    ' corpus created by filtering texts in another corpus by a specific criteria '
    #2do implement filtered id-fetching methods

    def __init__(self, corpus, filter, id = None):
        self.corpus = corpus; self.id = id
        if isinstance(filter, list): self.filters = filter
        else: self.filters = [ filter ]

    def filtered(self, textobj):
        for f in self.filters:
            if f(textobj): return True
        return False

    def corpusId(self):
        if self.id is not None: return self.id
        else: return self.filter.getId()+'_'+self.corpus.corpusId()

    def __iter__(self):
        for txtobj in self.corpus:
            if not self.filtered(txtobj): yield txtobj

    def getTexts(self, id_list):
        'get texts that pass the filter '
        texts = self.corpus.getTexts(id_list)
        return [ txtid for txtid in texts if not self.filtered(txtid[1]) ]

    def getText(self, id):
        'get text if it passes filter, None otherwise'
        txto = self.corpus.getText(id)
        if txto is None : return None
        if self.filtered(txto) : return None
        else: return txto

class TxtId():
    'auxiliary class for duplicate text filter, stores only text and id'
    def __init__(self, id, txt): self.text = txt; self.id = id

class DuplicateTextFilter():
    '''
    filters duplicate texts using hash functions.
    this filter compares with texts already filtered so order of filtering matters.
    if it is initialized with corpus only id's are stored and texts are
    fetched from corpus when neccessary, to conserve space
    '''
    def __init__(self, corpus = None):
        self.corpus = corpus
        self.__init_data()

    def __init_data(self):
        self.hash2texts = {}
        self.hashClashes = 0
        self.fetches = 0
        self.duplicates = 0

    def __getstate__(self):
        return self.corpus

    def __setstate__(self, state):
        self.corpus = state
        self.__init_data()

    def getId(self): return 'filter_duptext'

    def textDuplicate(self, txto, addNew = True):
        'check if Text object with the same text but different id exists'
        h = hash(txto.text)
        if h in self.hash2texts :
            self.hashClashes += 1
            if txto.id in [ txtid.id for txtid in self.hash2texts[h] ] :
                return False # same text already stored
            else:
                for txtid in self.hash2texts[h]:
                    # fetch from database if corpus exists
                    if txtid.text is None and self.corpus is not None:
                        result = self.corpus.getText(txtid.id)
                        self.fetches += 1
                        if result is None: txtid.text = None
                        else: txtid.text = result.text
                    if txtid.text == txto.text: return True #duplicate text
                if addNew:
                    if self.corpus is None: txtid = TxtId(txto.id, txto.text)
                    else: txtid = TxtId(txto.id, None)
                    self.hash2texts[h].append(txtid)
                return False
        elif addNew:
            if self.corpus is None: txtid = TxtId(txto.id, txto.text)
            else: txtid = TxtId(txto.id, None)
            self.hash2texts[h] = [ txtid ]

    def __call__(self, txtobj):
        if self.textDuplicate(txtobj):
            self.duplicates += 1
            #print 'duplicate: ' + txtobj.title
            return True
        else: return False

