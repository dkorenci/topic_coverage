from doc_topic_coh.resources import pytopia_context
from doc_topic_coh.resources.pytopia_context import data_store
from pytopia.context.ContextResolver import resolve

import codecs

def exportUspolCorpus(outfile):
    '''
    Export all texts from uspol corpus, tokenized with native tokenizer,
    one text per line.
    :return:
    '''
    f = codecs.open(outfile, 'w', 'utf8')
    corpus = resolve('us_politics')
    text2tokens = resolve('RsssuckerTxt2Tokens')
    for txto in corpus:
        toks = text2tokens(txto.text)
        textLine = u' '.join(toks).replace(u'\n', u' ')
        f.write(textLine); f.write(u'\n')
    f.close()

if __name__ == '__main__':
    exportUspolCorpus(data_store('corpus_output.txt'))
