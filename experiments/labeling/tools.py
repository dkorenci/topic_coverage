import random
import pickle
import codecs
import re
from textwrap import wrap

from corpus.factory import CorpusFactory
from pymedialab_settings.settings import labeling_folder
from preprocessing.text2tokens import RsssuckerTxt2Tokens


def getCorpusIdSample(seed, sampleSize, corpus_id = 'us_politics'):
    corpus = CorpusFactory.getCorpus(corpus_id)
    all_ids = [ id for id in corpus.getIds() ]
    random.seed(seed)
    random.shuffle(all_ids)
    id_list = all_ids[:sampleSize]
    assert len(id_list) == len(set(id_list))
    return id_list

def saveCorpusIdSample(id_list, sample_id):
    file = labeling_folder+sample_id
    pickle.dump(id_list, open(file, 'wb'))

def loadCorpusIdSample(sample_id):
    file = labeling_folder+sample_id
    return pickle.load(open(file, 'rb'))

TEXT_START = '************ DOCUMENT START ************'
TEXT_END = '************ DOCUMENT END ************'
def printIndexedTextsForLabeling(indTexts, file, labels, range, tfidfIndex=None):
    f = codecs.open(file, "w", "utf-8")
    f.write('[ TEXTS: %d .. %d ]\n\n' % (range[0], range[1]-1))
    for ind, txto in indTexts :
        f.write(TEXT_START + '\n')
        f.write(formatTextForLabeling(txto, labels, ind, tfidfIndex))
        f.write('\n\n')

def printTextsForLabeling(texts, file, labels):
    f = codecs.open(file, "w", "utf-8")
    for txto in texts :
        f.write(TEXT_START + '\n')
        f.write(formatTextForLabeling(txto, labels))
        f.write('\n\n')

ID_LABEL = 'ID: '; TITLE_LABEL = 'TITLE: '
TEXT_LABEL = 'TEXT: ' ; INDEX_LABEL = 'INDEX: '
TFIDF_LABEL = 'TFIDF TOKENS:'
def formatTextForLabeling(txto, label_list, ind = None, tfidfIndex=None):
    s = ''
    if ind is not None: s += INDEX_LABEL+str(ind)+'\n'
    s += ID_LABEL+str(txto.id)+'\n'
    s += formatLabels(label_list) + '\n'
    s += TITLE_LABEL+ formatTitle(txto.title)+'\n'
    if tfidfIndex is not None: s += TFIDF_LABEL+'\n'+formatTfidf(txto.text, tfidfIndex)+'\n'
    s += TEXT_LABEL + '\n' + formatText(txto.text) + '\n'
    return s

def formatTitle(title): return title.replace('\n', ' ').strip()
def formatText(text): return wrap_text(text, 100)
def wrap_text(text, charsPerLine):
    return '\n'.join(wrap(text.replace('\n', ' '), charsPerLine))

tokenizer = RsssuckerTxt2Tokens()
def formatTfidf(text, tfidfIndex):
    dict = tfidfIndex.tfidf.id2word
    tfidfBow = tfidfIndex.tfidf[dict.doc2bow(tokenizer(text))]
    byweight = sorted(tfidfBow, key=lambda x : x[1], reverse=True)
    s = '';
    for wordId, weight in byweight:
        s += '%s ' % dict[wordId]
    return formatText(s)

LABEL_LEFT = '[['; LABEL_RIGHT = ']]' ; LABEL_DELIM = ':'
def formatLabels(label_list):
    s = ''
    for label in label_list:
        s += '%s %s %s 0 %s' % (LABEL_LEFT, label, LABEL_DELIM, LABEL_RIGHT) + '\n'
    return s

def parseLabeledTexts(filename):
    '''
    parse texts generated with printIndexedTextsForLabeling()
    :return list of triples (textIndex, textId, map{label->value})
    '''
    f = codecs.open(filename, "r", "utf-8")
    texts = [];
    docCnt = 0;
    for li, line in enumerate(f.readlines()):
        line = line.strip()
        if line == TEXT_START :
            if docCnt > 0: addTextData(texts, li, index, id, labels)
            docCnt += 1
            index, id, labels = None, None, {}
            continue
        if line.startswith(ID_LABEL):
            try: id = int(line.split(':')[1].strip())
            except Exception as e: print line; raise e
            continue
        if line.startswith(INDEX_LABEL):
            try: index = int(line.split(':')[1].strip())
            except Exception as e: print line; raise e
            continue
        match = re.match(r'\[\[(.*):(.*)\]\]', line)
        if match is not None :
            lab = match.groups()[0].strip()
            labVal = match.groups()[1].strip()
            if re.match('[0-9]+', labVal) is not None:
                val = int(labVal) # value is integer, parse
            else: val = labVal # else set to string value
            labels[lab] = val
    addTextData(texts, -1, index, id, labels)
    return texts

def getLabeledTextObjectsFromParse(parse, corpus = CorpusFactory.getCorpus('us_politics')):
    'from a parse create a list of (texto, labeling) pairs, fetching text objects form corpus'
    id2lab = { id:lab for _, id, lab in parse }
    ids = [ id for _, id, _ in parse ]
    return [ (texto, id2lab[id]) for id, texto in corpus.getTexts(ids) ]

def addTextData(textList, lineIndex, textIndex, textId, labels):
    if textId is None or labels == {}: #or textIndex is None
        #if textIndex is None : textIndex = -99
        if textId is None : textId = -99
        raise Exception('uncomplete data: line %d ; i %d ; id %d \n labels: %s'
                        % (lineIndex, textIndex, textId, labels ))
    else: textList.append((textIndex, textId, labels))


