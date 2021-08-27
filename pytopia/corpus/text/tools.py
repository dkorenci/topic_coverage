from pytopia.context.ContextResolver import ContextResolver, resolve

import codecs, random
from os import path

def createTextPerLine(corpus, folder, maxTexts=None, seed=None):
    '''
    Create text per line corpus from some pytopia corpus and save to file,
     composing the file name from the params.
    :param folder: folder to store text new corpus to
    :param maxTexts, seed: see corpus2textPerLine
    :return:
    '''
    corpus = resolve(corpus)
    fname = '%s%s%s.txt' % (corpus.id, '' if not maxTexts else '_[:%s]'%str(maxTexts),
                                      '' if not seed else '_seed[%s]'%str(seed))
    txtFile = path.join(folder, fname)
    corpus2textPerLine(corpus, txtFile, maxTexts=maxTexts, seed=seed)

def corpus2textPerLine(corpus, file=None, text2tokens=None, enc='utf-8', maxTexts=None, seed=None):
    '''
    Read texts from pytopia coprus and write them to text-per-line corpus.
    :param corpus: pytopia corpus or id
    :param file: file where to store text corpus
    :param maxTexts: take only this many texts from the start of the corpus
    :param seed: if not None, shuffle corpus texts before
            taking texts from the start and creating new corpus
    :return:
    '''
    corpus = resolve(corpus)
    if text2tokens is not None: text2tokens = resolve(text2tokens)
    # suffle and take first maxTexts texts
    if maxTexts is None and seed is None: texts = corpus
    else:
        texts = [txto for txto in corpus]
        if seed is not None:
            random.seed(seed)
            random.shuffle(texts)
        if maxTexts is not None: texts = texts[:maxTexts]
    # compose file name
    if file is None:
        file = '%s_textPerLine%s%s%s.txt' % \
            (corpus.id, '_tokenized[%s]'%text2tokens.id if text2tokens else '',
             '_shuffleSeed[%d]'% seed if seed is not None else '',
             '_maxTexts[%d]'% maxTexts if maxTexts is not None else '')
        print file
    # write texts as text per line corpus file
    file = codecs.open(file, 'w', enc)
    for txto in texts:
        file.write(u'id=%s, ' % transformValue(txto.id))
        for name, val in txto:
            s = u'%s=%s, ' % (name, transformValue(val))
            file.write(s)
        if text2tokens: text = ' '.join(tok for tok in text2tokens(txto.text))
        else: text = txto.text
        file.write(u'text=%s ' % transformText(text))
        file.write(u'\n')
    file.close()

def transformValue(val):
    return escapeComma(removeNewline(unicode(val)))

def transformText(txt): return removeNewline(txt)

def removeNewline(str, nl='\n'):
    '''Replace newline by blank
    :param nl: sequence of chars representing new line, commonly '\n' or '\r\n'.
    '''
    return str.replace(nl, ' ')

def escapeComma(str):
    '''Replace each ',' by '\,' '''
    return str.replace(',', '\\,')
