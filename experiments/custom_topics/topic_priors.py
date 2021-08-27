from models.label import modelLabel
from experiments.rsssucker_lda import *
from experiments.grid_search.options import ModelOptions
from experiments.custom_topics.tools import create_prior_vector
from pymedialab_settings import settings


def train_reshaped():
    database = 'rsssucker_topus1_27022015'
    dictionary = loadRssSuckerDictionary(database)
    bowstream = loadBowStream_numpy(database)
    T = 100; N = len(dictionary)
    topic_priors = np.empty((T, N))
    defined_topics = {
        0: ['marriag','gay','judg','suprem','legal','justic','coupl','appeal','ban','same-sex'],
            #'attorney','constitut','marri','alabama','licens','counti','district','civil','violat'],
        1: ['dog','anim','cat','pet','owner','zoo','human','hors','shelter','rescu'],
            #'puppi','lion','adopt','bull','eleph','societi','whale','foster','nieto','pup']
        2: ['greec','european','greek','euro','europ','debt','bank','eu','parliament','elect'],
            #'syriza','athen','central','bailout','tsipra'
        3: ['missouri','princip','elementari','joshua','foust']
    }
    # test dictionary
    for ti in defined_topics:
        for w in defined_topics[ti]:
            id = dictionary.token2id[w]
            print w, id, dictionary[id]
    prior = 0.01
    beta_ps = [0.01, 0.001, 0.0001]
    p = 0.03
    #prior_pairs = [(0.05,0.001),(1,0.01),(0.1,0.0001),(10,0.01)]
    for beta_p in beta_ps:
        for t in range(T):
            if t in defined_topics.keys():
                Nw = len(defined_topics[t])
                beta_w = ( p * (N - Nw) * beta_p ) / (1.0 - p*Nw)
                print t, Nw, beta_w, beta_p
                topic_priors[t, ] = create_prior_vector(N, dictionary, defined_topics[t], beta_w, beta_p)
            else:
                topic_priors[t, ] = np.repeat(prior, N)
        opts = ModelOptions(num_topics=T, alpha='auto', alpha_init=0.005,
                        eta=topic_priors, offset=1, decay=0.625, chunksize=1000, passes=1)
        model_name = modelLabel(opts)+ ('_priors_p%.2f_betap%.4f' % (p, beta_p))
        print model_name
        model = opts.getModel(dictionary)
        model.update(bowstream)
        tmodel = GensimLdamodel(model)
        tmodel.save(settings.models_folder+model_name)
