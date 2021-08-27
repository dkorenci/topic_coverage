from doc_topic_coh.resources import pytopia_context

from pytopia.context.GlobalContext import GlobalContext
from pytopia.context.ContextResolver import resolve

if __name__ == '__main__':
    print len(resolve('us_politics_dict'))
    #print unicode(GlobalContext.get())