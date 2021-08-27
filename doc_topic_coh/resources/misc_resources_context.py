from pytopia.context.Context import Context

def palmettoContext():
    '''
    Locations of Palmetto Lucene indexes.
    :return:
    '''
    ctx = Context('palmetto_context')
    ctx['wiki_docs'] = '/datafast/palmetto/enwiki/lucene_index' #'/datafast/palmetto_indexes/wikipedia_uspoltok/windowed_documents'
    ctx['wiki_paragraphs'] = '/datafast/palmetto_indexes/wikipedia_uspoltok/boolean_paragraphs'
    ctx['wiki_standard'] = '/datafast/palmetto/wikipedia_bd'
    ctx['uspol_palmetto_index'] = '/datafast/palmetto/us_politics/windowed'
    return ctx

def dictionaryContext():
    from pytopia.context.Context import Context
    from pytopia.resource.loadSave import loadResource
    from doc_topic_coh.settings import dataStore
    ctx = Context('gtar_dict_context')
    dict = loadResource(dataStore('dictionaries',
        'GensimDictAdapter_buildOpts[GensimDictBuildOptions_words2keep[50000]]_corpusId[us_politics]_txt2tokId[alphanum_gtar_stopword_tokenizer]/'))
    dict.id = 'uspol_dict_notnormalized'
    ctx.add(dict)
    return ctx
