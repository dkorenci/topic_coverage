'''
modelLabel is moved to separate file to avoid import circularity in gensim_mode.models.ldamodel
when ModelOptions is moved out of model_utils, modelLabel can be returned there
'''

def modelLabel(model):
    ' create string label for modified gensim LdaModel or ModelOptions '
#    if not isinstance(model, (LdaModel, LdaModelMod, ModelOptions)): raise TypeError()
#     if not hasattr(model, 'alpha'):
#         model.alpha = ''
#         #todo check why saved models do not have alpha
#     if not hasattr(model, 'eta'): model.eta = ''
    if isinstance(model.alpha, (int, long, float)):
        alphastr = '%.3f' % model.alpha
    elif isinstance(model.alpha, str): alphastr = model.alpha
    else: alphastr = 'array'
    alpha_init = model.alpha_init if hasattr(model, 'alpha_init') else ''
    if isinstance(model.alpha, str) and model.alpha == 'auto' :
        alphastr += str(alpha_init)
    if isinstance(model.eta, (int, long, float)): etastr = '%.3f'%model.eta
    else: etastr = 'vector'
    label = 'ldamodel_T%d_A%s_Eta%s_Off%.3f_Dec%.3f_Chunk%d_Pass%d' % \
            (model.num_topics, alphastr, etastr, model.offset,
             model.decay, model.chunksize, model.passes)
    if hasattr(model, 'label'): label += '_label' + str(model.label)
    if hasattr(model, 'seed'): label += '_seed' + str(model.seed)
    return label