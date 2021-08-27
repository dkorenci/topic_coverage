from pytopia.context.Context import Context
from phenotype_context.dictionary.create_4outof5_dictionary import \
    loadDictionary as load4outof5

def phenotypeDictContext():
    ctx = Context('phenotype_dictionary_context')
    ctx.add(load4outof5())
    return ctx