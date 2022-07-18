'''
Provide fixtures as aliases for corpus ids that can change.
'''

import pytest

@pytest.fixture(scope="session")
def corpus_uspol_small():
    return 'us_politics_dedup_[100]_seed[1]'

@pytest.fixture(scope="session")
def corpus_uspol_medium():
    return 'us_politics_dedup_[2500]_seed[3]'

def toyCorpusUsPolitics(verbose=False):
    '''
    Create toy corpus with similar structure as us_politics.
    '''
    # these are sentences from the original corpus
    corpusTxt = '''
    id = 0, text = The White House group's agenda was deep--with racial concerns about criminal justice
    id = 1, text = Mitch McConnell threw down the gauntlet. "What the administration has done to the coal industry is a true outrage,"
    id = 2, text = Bruce Rauner's efforts to weaken labor unions, saying two of the main ideas the Republican has been pitching across the state would be illegal.
    id = 3, text = Because the radical rhetoric of the National Rifle Association's (NRA) leadership tells us that "the only thing that can stop a bad guy with a gun is a good guy with a gun."
    id = 5, text = The president declined to label al Qaeda and the Islamic State of Iraq and Syria (ISIS) "Islamic terrorists," and he's taking some heat from critics.
    '''
    from pytopia.corpus.text.TextCorpus import TextCorpus
    c = TextCorpus(corpusTxt)
    c.id = 'us_politics_toy'
    c.dictionary = createDefaultGensimDict(c, 'RsssuckerTxt2Tokens')
    if verbose: print 'TOY DICT'
    d = c.dictionary
    for t in d:
        i = d.token2index(t)
        if verbose:
            print t
            print d.token2index(t)
            print d.index2token(i)
    c.dictionary.id = 'us_politics_toy_dict'
    return c

def createDefaultGensimDict(corpus, txt2Tokens):
    from pytopia.adapt.gensim.dictionary.dict_build import GensimDictBuilder, \
        GensimDictBuildOptions
    opts = GensimDictBuildOptions(None, None, None)
    return GensimDictBuilder().buildDictionary(corpus, txt2Tokens, opts)
