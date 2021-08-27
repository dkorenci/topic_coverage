from doc_topic_coh.settings import dataStore
from pytopia.context.Context import Context

from os import path, listdir

def modelsContext():
    ctx = Context('models_context')
    modelFolders = ['models1']
    for f in modelFolders:
        addModelsFromFolder(ctx, dataStore.subfolder(f), f)
    return ctx

def addModelsFromFolder(ctx, folder, label):
    from pytopia.resource.loadSave import loadResource
    for i, f in enumerate(sorted(listdir(folder))):
        m = loadResource(path.join(folder,f))
        m.id = '%s.%d' % (label, i+1)
        ctx.add(m)

if __name__ == '__main__':
    ctx = modelsContext()