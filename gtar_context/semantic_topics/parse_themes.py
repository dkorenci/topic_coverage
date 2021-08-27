'''
Compose artificial topic model with topics corresponding
to semantic topics (themes) as defined in definition files.
'''

from pytopia.topic_model.ArtifTopicModel import ArtifTopicModel
from gtar_context.corpus_context import gtarCorpusContext
from gtar_context.text2tokens_context import gtarText2TokensContext
from gtar_context.dict_context import gtarDictionaryContext
from gtar_context.corpus_context import gtarCorpusContext

from pyutils.file_utils.location import FolderLocation as loc
from os import path
thisfolder = loc(path.dirname(__file__))

class ParsedSemTopics():
    ''' Semantic topic data from definition file '''
    def __init__(self, words, docs, label, topics, dominantData):
        '''
        :param words: words defining the topics
        :param docs: ids of documents defining the topic
        :param dominantData: 'docs' or 'words', singify if documents should
            be use primarily (the words are too few or of low quality) or vice versa
        :param label: label of the semantic topic
        :param topics: model topics corresponding to the semantic topic
        '''
        self.words, self.docs, self.dominantData, self.topics, self.label = \
            words, docs, dominantData, topics, label

    def __str__(self):
        return ' label: %s\n topics: %s\n words: %s\n docs: %s\n dominant: %s' % \
               (self.label, ','.join(self.topics), ','.join(self.words), ','.join(self.docs), str(self.dominantData))

def definitionFiles():
    return [
        thisfolder('theme_annotations_1.txt'),
        thisfolder('theme_annotations_2.txt')
    ]

def parseSemTopics(lines, verbose=False):
    '''
    Create ParsedSemTopics from a list of text lines.
    '''
    if not lines: return None
    txt2tok = gtarText2TokensContext()['RsssuckerTxt2Tokens']
    themeHeader = 'THEME:'
    topicsHeader = 'TOPICS:'
    wordsHeader = 'WORDS:'; wordsBlock = False; words = []
    docHeader = 'DOCS:'; docBlock = False; docs = []
    wordsDominant = 'words'; docDominant = 'docs'; dominantData = None
    # the assumed order of data is: theme, topics, words, docs, doc/word dominant indicator
    if verbose: print 'STARING THEME PARSE'
    for i, l in enumerate(lines):
        l = l.strip()
        if verbose: print l
        if not l: continue
        if l.startswith(wordsDominant) or l.startswith(docDominant):
            if l.endswith('?'): l = l[:-1]
            dominantData = l
            assert not wordsBlock and docBlock
            assert i == len(lines)-1
            docBlock = False
        if l.startswith(themeHeader): label = l[len(themeHeader):].strip()
        if l.startswith(topicsHeader):
            topics = l[len(topicsHeader):].split()
            topics = [t.strip() for t in topics]
        if l.startswith(docHeader) or docBlock:
            if l.startswith(docHeader):
                assert wordsBlock
                d = l[len(docHeader):].split()
                docBlock = True; wordsBlock = False
            else: d = l.split()
            docs.extend(d)
        if l.startswith(wordsHeader) or wordsBlock:
            if l.startswith(wordsHeader):
                w = l[len(wordsHeader):].strip()
                # words on the header line are written by humans, must be lemmatized and stemmed
                w = txt2tok(w)
                wordsBlock = True
            else: w = l.split()
            words.extend(w)
    parsed = ParsedSemTopics(words=words, docs=docs, dominantData=dominantData,
                             label=label, topics=topics)
    if verbose: print parsed
    return parsed

def validate():
    dict = gtarDictionaryContext()['us_politics_dict']
    print len(dict)
    print ','.join(w for w in dict)
    for t in parseAllThemes():
        for w in t.words:
            if not w in dict:
                print w
                break

def countWords(words, corpusId='us_politics', txt2tokId='RsssuckerTxt2Tokens'):
    '''
    Count occurence of words in a corpus.
    '''
    corpus = gtarCorpusContext()[corpusId]
    txt2tok = gtarText2TokensContext()[txt2tokId]
    wrdcnt = { w:0 for w in words }
    for txto in corpus:
        tokens = set(txt2tok(txto.text))
        for w in words:
            if w in tokens: wrdcnt[w] += 1
    for w, cnt in wrdcnt.iteritems():
        print w, cnt

def parseDefinitionFile(file, verbose=False):
    import codecs
    themeDelimiter = '**********'
    allLines = codecs.open(file, 'r', 'utf-8').readlines()
    topics, lines = [], []
    for i, line in enumerate(allLines):
        if line.strip() == themeDelimiter or i == len(allLines)-1:
            parsed = parseSemTopics(lines, verbose)
            if parsed: topics.append(parsed)
            lines = []
            #if len(topics) == 5: break
        else:
            lines.append(line)
    if verbose: print len(topics)
    return topics

def parseAllThemes():
    topics = []
    for dfile in definitionFiles():
        topics.extend(parseDefinitionFile(dfile))
    return topics

if __name__ == '__main__':
    #parseAllThemes()
    #parseDefinitionFile(definitionFiles()[0])
    #print 'abc'[:-1]
    #validate()
    countWords(['bush', 'obama', 'republican', 'senat', 'health', 'converv', 'hillari'])