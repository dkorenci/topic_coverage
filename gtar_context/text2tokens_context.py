from pytopia.context.Context import Context

def gtarText2TokensContext():
    from pytopia.nlp.text2tokens.gtar.text2tokens import RsssuckerTxt2Tokens
    from pytopia.nlp.text2tokens.regexp import alphanumTokenizer, whitespaceTokenizer
    from pytopia.nlp.text2tokens.gtar.text2tokens import alphanumStopwordsTokenizer
    from pytopia.nlp.text2tokens.gtar.stopwords import RsssuckerSwRemover
    ctx = Context('gtar_text2tokens_context')
    ctx.add(RsssuckerTxt2Tokens())
    ctx.add(alphanumTokenizer())
    ctx.add(whitespaceTokenizer())
    alphasw = alphanumStopwordsTokenizer(RsssuckerSwRemover())
    alphasw.id = 'gtar_alphanum_stopword_tokenizer'
    ctx.add(alphasw)
    return ctx

if __name__ == '__main__':
    print gtarText2TokensContext()
