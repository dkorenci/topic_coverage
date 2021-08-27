'''
Various dataset-related statistics.
'''

from pytopia.context.ContextResolver import resolve
from gtar_context.compose_context import gtarContext
from stat_utils.utils import Stats

def wordLengthDistribution(corpus='us_politics', txt2tok='gtar_alphanum_stopword_tokenizer'):
    ctx = gtarContext()
    with ctx:
        txt2tok = resolve(txt2tok)
        texlens = [len(txt2tok(txto.text)) for txto in resolve(corpus)]
        print Stats(texlens)

if __name__ == '__main__':
    print wordLengthDistribution()