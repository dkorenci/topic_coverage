from pytopia.context.Context import Context
from phenotype_context.phenotype_topics.construct_model import loadModel, MODEL_ID, MODEL_DOCS_ID

def phenotypeModelContext():
    ctx = Context('phenotype_model_context')
    ctx.add(loadModel(MODEL_ID))
    ctx.add(loadModel(MODEL_DOCS_ID))
    return ctx