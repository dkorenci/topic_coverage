from pytopia.corpus.Text import Text
from pyutils.file_utils.location import FolderLocation as loc
from phenotype_context.settings import corporaRoot
from phenotype_context.phenotype_corpus.raw_corpus import loadText

import re

corporaRoot = loc(corporaRoot)

class PhenoText(Text):

    @property
    def title(self):
        if hasattr(self, '_title'): return self._title
        fallback = False
        folder, fname = self.__parseId()
        if not folder or not fname: fallback=True
        if not fallback:
            text = None
            try: text = loadText(corporaRoot(folder, fname))
            except: fallback = True
            if not fallback:
                text = snl(text); title = None
                #print text
                bname = self.__parseBacteria()
                if bname is None: fallback = True
                if not fallback:
                    genus, species = bname
                    #print 'genus[%s], species[%s]' % (genus, species)
                    if folder == 'hamap': title = hamapTitle(text, genus, species)
                    elif folder == 'microbewiki': title = mwikiTitle(text, genus, species)
                    elif folder == 'PMCArticles': title = pmcaTitle(text, genus, species)
                    elif folder == 'pubmedSearch': title = pubmedTitle(text, genus, species)
                    elif folder == 'wikipedia': title = wikipediaTitle(text, genus, species)
                if title is None: fallback = True
        if fallback: self._title = self.text[:150]
        else: self._title = title.strip()
        return self._title

    @property
    def rawtext(self):
        if hasattr(self, '_rawtext'): return self._rawtext
        fallback = False
        folder, fname = self.__parseId()
        if not folder or not fname: fallback=True
        if not fallback:
            text = None
            try:
                text = loadText(corporaRoot(folder, fname))
                text = text.strip()
            except: fallback = True
            if not text: fallback = True
        if fallback: text = self.text
        self._rawtext = text
        return self._rawtext

    def __parseId(self):
        id = self.id
        di = id.find('_')
        if di == -1: return None, None
        folder, fname = id[:di], id[di+1:]+'.txt'
        return folder, fname

    def __parseBacteria(self):
        _, fname = self.__parseId()
        if fname is None: return None
        ei = fname.find('__')
        if ei == -1: return None
        name = fname[:ei]
        if '_' in name: si = name.find('_')
        elif ' ' in name: si = name.find(' ')
        else: return None
        genus, species = name[:si].strip(), name[si + 1:].strip()
        species = species.replace('_', ' ')
        if genus and species: return genus, species
        else: return None

def snl(txt):
    '''Strip new line'''
    if '\r\n' in txt: nline = '\r\n'
    else: nline = '\n'
    if txt.find(nline) == -1: return txt
    return txt.replace(nline, ' ')
    #return u' '.join(txt.split(nline))

def variants(genus, species):
    return [(genus + ' ' + species).lower(), (genus[0] + '. ' + species).lower()]

def isOccurence(text, genus, species):
    '''
    Return index of first of occurence in text of 'genus species is' or 'g. species is'.
    '''
    text = text.lower()
    vars = variants(genus, species)
    #print 'variants', vars
    for v in vars:
        i = text.find(v+' is')
        #print 'isOcc', i, v
        if i != -1: return i
    return None

def allOccurences(text, genus, species):
    text = text.lower()
    vars = variants(genus, species)
    occs = []
    for v in vars:
        occs.extend([m.start() for m in re.finditer(re.escape(v), text)])
    return sorted(set(occs))

def wikipediaTitle(text, genus, species, K=300):
    '''Longer text, possible header, return fragmet of text from
    'is occurence' or 4th occurence '''
    io = isOccurence(text, genus, species)
    if io is None:
        occs = allOccurences(text, genus, species)
        if len(occs) < 2: return None
        if len(occs) < 4: oi = occs[-1]
        else: oi = occs[3]
        return text[oi:oi+K]
    else:
        return text[io:io+K]

def pubmedTitle(text, genus, species, K=300):
    '''Lots of abstracts in one document, get fragment of text arround first occurence.'''
    occs = allOccurences(text, genus, species)
    if len(occs) == 0: return None
    else:
        fo = occs[0]
        return text[fo-100:fo+K]

def mwikiTitle(text, genus, species, K=300):
    '''Longer text, possible header, return fragmet of text after
    'is occurence' or second occurence. '''
    io = isOccurence(text, genus, species)
    if io is None:
        occs = allOccurences(text, genus, species)
        if len(occs) < 2: return None
        else: return text[occs[1]:occs[1]+K]
    else:
        #print '[%s]'%text[io:io+K]
        return text[io:io+K]

def hamapTitle(text, genus, species, K=300):
    ''' Text contains short bacteria description. '''
    occs = allOccurences(text, genus, species)
    if len(occs) > 0: return text[occs[0]:occs[0]+K]
    else: return text[:K]

def pmcaTitle(text, genus, species, K=300, abstrKword='abstract'):
    ''' Text contains article about the bacteria, return start of the abstract. '''
    occs = allOccurences(text, genus, species)
    if len(occs) > 0:
        return text[occs[0]:occs[0]+K]
    else:
        # return start of the abstract
        ai = text.lower().find(abstrKword.lower())
        if ai == -1: return text[:K]
        else:
            aend = ai+len(abstrKword)
            return text[aend:aend+K]
