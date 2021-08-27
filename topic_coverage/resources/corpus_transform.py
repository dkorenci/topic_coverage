from topic_coverage.resources import pytopia_context
from topic_coverage.resources.pytopia_context import testGlobalContext

from pytopia.context.ContextResolver import resolve
from pytopia.corpus.text.tools import corpus2textPerLine

def uspol2textPerLine():
    ''' Transform original us politics corpus stored
    in a postgres database to text-per-line format '''
    corpus = 'us_politics'; txt2tok = 'RsssuckerTxt2Tokens'
    corpus2textPerLine(corpus, text2tokens=txt2tok)

def uspolTextPerLineTest(txtPerLineFile):
    from pytopia.testing.utils import compareCorpora
    from pytopia.nlp.text2tokens.regexp import whitespaceTokenizer
    from pytopia.corpus.text.TextPerLineCorpus import TextPerLineCorpus
    tpl = TextPerLineCorpus(txtPerLineFile)
    tplTok = whitespaceTokenizer()
    compareCorpora('us_politics', tpl, txt2tok=('RsssuckerTxt2Tokens', tplTok), id2str=True,
                   testProperties=True, prop2str=True)

if __name__ == '__main__':
    #uspol2textPerLine()
    uspolTextPerLineTest('us_politics_textPerLine_tokenized[RsssuckerTxt2Tokens].txt')
    #testGlobalContext()