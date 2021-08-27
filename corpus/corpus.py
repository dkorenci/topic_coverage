"abstract corpus functionality"

class Corpus:
    "corpus of texts"

    def corpusId(self):
        raise NotImplementedError()

    def getIds(self):
        "get ids of all the texts in the corpus"
        return [ txto.id for txto in self ]
        #raise NotImplementedError()

    def getTexts(self, id_list):
        "get texts for selected ids as iterable of (id, Text) pairs"
        raise NotImplementedError()

    def getText(self, id):
        result = [ txt for id, txt in self.getTexts([id]) ]
        if len(result) == 0 : return None
        else : return result[0]

    def getSample(self, size = 100, seed = 12345):
        raise NotImplementedError()

    def __iter__(self):
        "iterate over texts in the corpus"
        raise NotImplementedError()

    def __len__(self):
        if not hasattr(self, 'length'):
            # cache the corpus length
            self.length = sum(1 for _ in self)
        return self.length


