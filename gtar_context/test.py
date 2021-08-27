from pytopia.context.GlobalContext import GlobalContext
from pytopia.context.ContextResolver import resolve
from gtar_context.compose_context import gtarContext, gtarRefModelsContext

def printStats():
    with gtarContext():
        corpus = resolve('us_politics_textperline')
        print len(corpus)
        dict = resolve('us_politics_dict')
        print len(dict)

def printContext():
    GlobalContext.set(gtarContext())
    print unicode(GlobalContext.get())

if __name__ == '__main__':
    printStats()