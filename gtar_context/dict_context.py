from pytopia.context.Context import Context
from pytopia.utils.load import listSubfolders
from pytopia.resource.loadSave import loadResource
from pyutils.file_utils.location import FolderLocation as loc

from os import path

thisfolder = loc(path.dirname(__file__))

def gtarDictionaryContext():
    f = thisfolder('dictionary_resources')
    ctx = Context('gtar_dictionary_context')
    for sf in listSubfolders(f):
        ctx.add(loadResource(sf))
    return ctx

def test():
    from pytopia.testing.validity_checks import checkDictionary
    ctx = gtarDictionaryContext()
    # todo colve problem with the dedup dict
    for d in ctx:
        checkDictionary(ctx[d])

if __name__ == '__main__':
    test()