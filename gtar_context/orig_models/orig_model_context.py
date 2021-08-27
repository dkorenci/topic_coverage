'''
Loads five topic models from the original eksperiment.
'''

from os import path

from gtar_context.settings import models_folder
from pytopia.adapt.gensim.lda.GensimLdaModel import GensimLdaModel
from pytopia.context.Context import Context

modelId2Folder = {
    'uspolM0' : 'uspolM0_234_ldamodel_T50_A1.000_Eta0.010_Off1.000_Dec0.500_Chunk1000_Pass10_label_seed345556',
    'uspolM1' : 'uspolM1_234_ldamodel_T50_A1.000_Eta0.010_Off1.000_Dec0.500_Chunk1000_Pass10_label_seed877312',
    'uspolM2' : 'uspolM2_234_ldamodel_T50_A1.000_Eta0.010_Off1.000_Dec0.500_Chunk1000_Pass10_label_seed8903',
    'uspolM10' : 'uspolM10_045_ldamodel_T100_A0.500_Eta0.010_Off1.000_Dec0.500_Chunk1000_Pass10_label_seed345556',
    'uspolM11' : 'uspolM11_045_ldamodel_T100_A0.500_Eta0.010_Off1.000_Dec0.500_Chunk1000_Pass10_label_seed133890'
}

def createWrappedGensimModel(mid, folder):
    m = GensimLdaModel(None, id=mid)
    m.load(folder)
    m.corpus = 'us_politics'
    m.dictionary = 'us_politics_dict'
    m.text2tokens = 'RsssuckerTxt2Tokens'
    return m

def gtarModelsContext():
    ctx = Context('gtar_orig_models_context')
    for mid, mfolder in modelId2Folder.iteritems():
        f = path.join(models_folder, mfolder)
        if path.exists(f):
            ctx.add(createWrappedGensimModel(mid, f))
    return ctx

if __name__ == '__main__':
    print gtarModelsContext()