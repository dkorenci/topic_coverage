from gensim_mod.models.ldamodel import LdaModel as LdaModelMod
from models.label import modelLabel

def gridSearchUsPolNumTopicsOptions(numPasses):
    # uspolM10_045_ldamodel_T100_A0.500_Eta0.010_Off1.000_Dec0.500_Chunk1000_Pass10_label_seed345556
    # uspolM0_234_ldamodel_T50_A1.000_Eta0.010_Off1.000_Dec0.500_Chunk1000_Pass10_label_seed345556
    # ldamodel_T200_A0.250_Eta0.010_Off10.000_Dec0.500_Chunk1000_Pass2_label_seed345556
    # ldamodel_T100_A0.500_Eta0.010_Off10.000_Dec0.500_Chunk1000_Pass2_label_seed345556
    options = []
    num_topics = range(50,201)
    for T in num_topics:
        if T <= 150: off = 1.0
        else: off = 10.0
        o = ModelOptions(num_topics=T, alpha=50.0/T, alpha_init=None,
                         eta=0.01, offset=off, decay=0.5, chunksize=1000, passes=numPasses)
        options.append(o)
    return options

def gridSearchConvergenceUspol1Options():
    'options to check variance of different models with same hyperparams'
    # uspol1_234_ldamodel_T50_A1.000_Eta0.010_Off1.000_Dec0.500_Chunk1000_Pass2_label_seed345556
    options = []
    o1 = ModelOptions(num_topics=50, alpha=1.0, alpha_init=0.01, eta=0.01, offset=1.0,
                     decay=0.5, chunksize=1000, passes=30)
    options.append(o1)
    return options

def gridSearchPassesTestOptions():
    'options to check variance of different models with same hyperparams'
    labels = range(10)
    options = []
    o1 = ModelOptions(num_topics=50, alpha=1.0, alpha_init=0.01, eta=0.01, offset=1.0,
                     decay=0.5, chunksize=2000, passes=5)
    options.append(o1)
    return options

def gridSearchWorldNewsOptions():
    '''
    auto priors set to lower the expected strongest topic ratio to ~45 %
    only one chunksize
    '''

    num_topics = [50, 100, 200]
    alphas = { 50: [ ('auto', 0.04), (1.0, None) ],
               100:[ ('auto',0.02), (0.5, None) ],
               200:[ ('auto', 0.009), (0.25, None) ] }
    eta = [0.01]
    passes = [2]
    offset = [1, 10 ,100 ,1000]
    decay = [0.5, 0.625, 0.75, 0.875, 1.0]
    chunksize = [2000]
    options = []
    for T in num_topics:
        for A in alphas[T]:
            for E in eta:
                for off in offset:
                    for dec in decay:
                        for chunk in chunksize:
                            for p in passes:
                                o = ModelOptions(num_topics=T, alpha=A[0], alpha_init=A[1],
                                                 eta=E, offset=off, decay=dec, chunksize=chunk,
                                                 passes=p)
                                options.append(o)
    return options

def gridSearchUsNewsOptions():
    '''
    auto priors set to lower the expected strongest topic ratio to ~45 %
    only one chunksize
    '''

    num_topics = [50, 100, 200]
    alphas = { 50: [ ('auto', 0.04), (1.0, None) ],
               100:[ ('auto',0.02), (0.5, None) ],
               200:[ ('auto', 0.009), (0.25, None) ] }
    eta = [0.01]
    passes = [2]
    offset = [1, 10 ,100 ,1000]
    decay = [0.5, 0.625, 0.75, 0.875, 1.0]
    chunksize = [2000]
    options = []
    for T in num_topics:
        for A in alphas[T]:
            for E in eta:
                for off in offset:
                    for dec in decay:
                        for chunk in chunksize:
                            for p in passes:
                                o = ModelOptions(num_topics=T, alpha=A[0], alpha_init=A[1],
                                                 eta=E, offset=off, decay=dec, chunksize=chunk,
                                                 passes=p)
                                options.append(o)
    return options

def gridSearchUsPolitics2Options():
    '''
    auto priors set to lower the expected strongest topic ratio to ~45 %
    only one chunksize, only auto priors
    '''
    num_topics = [50, 100, 200]
    alphas = { 50: [ ('auto', 0.04) ],
               100:[ ('auto',0.02) ],
               200:[ ('auto', 0.009) ] }
    eta = [0.01]
    passes = [2]
    offset = [1, 10 ,100 ,1000]
    decay = [0.5, 0.625, 0.75, 0.875, 1.0]
    chunksize = [1000]
    options = []
    for T in num_topics:
        for A in alphas[T]:
            for E in eta:
                for off in offset:
                    for dec in decay:
                        for chunk in chunksize:
                            for p in passes:
                                o = ModelOptions(num_topics=T, alpha=A[0], alpha_init=A[1],
                                                 eta=E, offset=off, decay=dec, chunksize=chunk,
                                                 passes=p)
                                options.append(o)
    return options

def gridSearchUsPoliticsOptions():
    num_topics = [50, 100, 200]
    alphas = { 50: [ ('auto', 0.01), (1.0, None) ],
               100:[ ('auto',0.005), (0.5, None) ],
               200:[ ('auto', 0.003), (0.25, None) ] }
    eta = [0.01]
    passes = [2]
    offset = [1, 10 ,100 ,1000]
    decay = [0.5, 0.625, 0.75, 0.875, 1.0]
    chunksize = [500, 1000, 2000]
    options = []
    for T in num_topics:
        for A in alphas[T]:
            for E in eta:
                for off in offset:
                    for dec in decay:
                        for chunk in chunksize:
                            for p in passes:
                                o = ModelOptions(num_topics=T, alpha=A[0], alpha_init=A[1],
                                                 eta=E, offset=off, decay=dec, chunksize=chunk,
                                                 passes=p)
                                options.append(o)
    return options

def gridSearch2Options():
    num_topics = [50, 200]
    alphas = {50:[('auto', 0.01)], 200:[('auto', 0.003)]}
    eta = [0.01]
    passes = [1,2]
    offset = [1, 10 ,100 ,1000]
    decay = [0.5, 0.625, 0.75, 0.875, 1.0]
    chunksize = [100, 1000, 2000, 5000, 10000]
    options = []
    for T in num_topics:
        for A in alphas[T]:
            for E in eta:
                for off in offset:
                    for dec in decay:
                        for chunk in chunksize:
                            for p in passes:
                                o = ModelOptions(num_topics=T, alpha=A[0], alpha_init=A[1],
                                                 eta=E, offset=off, decay=dec, chunksize=chunk,
                                                 passes=p)
                                options.append(o)
    return options


def gridSearch3Options():
    num_topics = [50, 100, 200]
    alphas = {50:[(1.0, None)], 100:[(0.5, None)], 200:[(0.25, None)]}
    eta = [0.01]
    passes = [1]
    offset = [1, 10 ,100 ,1000]
    decay = [0.5, 0.625, 0.75, 0.875, 1.0]
    chunksize = [1000, 5000, 10000]
    options = []
    for T in num_topics:
        for A in alphas[T]:
            for E in eta:
                for off in offset:
                    for dec in decay:
                        for chunk in chunksize:
                            for p in passes:
                                o = ModelOptions(num_topics=T, alpha=A[0], alpha_init=A[1],
                                                 eta=E, offset=off, decay=dec, chunksize=chunk,
                                                 passes=p)
                                options.append(o)
    return options


def gridSearchTestOptions():
    num_topics = [50]
    alphas = {50:[(1.0, None)]}
    eta = [0.01]
    passes = [1]
    offset = [1, 10]
    decay = [0.5, 0.75, 1.0]
    chunksize = [200, 1000]
    options = []
    for T in num_topics:
        for A in alphas[T]:
            for E in eta:
                for off in offset:
                    for dec in decay:
                        for chunk in chunksize:
                            for p in passes:
                                o = ModelOptions(num_topics=T, alpha=A[0], alpha_init=A[1],
                                                 eta=E, offset=off, decay=dec, chunksize=chunk,
                                                 passes=p)
                                options.append(o)
    return options

def gridSearchTestSingleOption():
    num_topics = [50]
    alphas = {50:[(1.0, None)]}
    eta = [0.01]
    passes = [1]
    offset = [1]
    decay = [0.5]
    chunksize = [1000]
    options = []
    for T in num_topics:
        for A in alphas[T]:
            for E in eta:
                for off in offset:
                    for dec in decay:
                        for chunk in chunksize:
                            for p in passes:
                                o = ModelOptions(num_topics=T, alpha=A[0], alpha_init=A[1],
                                                 eta=E, offset=off, decay=dec, chunksize=chunk,
                                                 passes=p)
                                options.append(o)
    return options

def gridSearchVarOptions():
    'options to check variance of different models with same hyperparams'
    labels = range(10)
    options = []
# ldamodel_T50_Aauto0.01_Eta0.010_Off1.000_Dec0.500_Chunk2000_Pass2
# ldamodel_T100_Aauto0.005_Eta0.010_Off1.000_Dec0.625_Chunk2000_Pass1
# ldamodel_T200_Aauto0.003_Eta0.010_Off10.000_Dec0.500_Chunk2000_Pass1
    for l in labels:
        o1 = ModelOptions(num_topics=50, alpha='auto', alpha_init=0.01, eta=0.01, offset=1.0,
                     decay=0.5, chunksize=2000, passes=1, label=l)
        o2 = ModelOptions(num_topics=100, alpha='auto', alpha_init=0.005, eta=0.01, offset=1.0,
                     decay=0.625, chunksize=2000, passes=1, label=l)
        o3 = ModelOptions(num_topics=200, alpha='auto', alpha_init=0.003, eta=0.01, offset=10.0,
                     decay=0.5, chunksize=2000, passes=1, label=l)
        options.append(o1)
        options.append(o2)
        options.append(o3)
    return options


def modelBuildOptions():
    'create set of options for specific settings'
    num_topics = [100]; alphas = [('auto', 0.005)]; eta = [0.01]
    passes = [1];
    options = []
# 15 ldamodel_T100_Aauto0.005_Eta0.010_Off1000.000_Dec0.625_Chunk100_Pass1
# 70 ldamodel_T100_Aauto0.005_Eta0.010_Off100.000_Dec0.625_Chunk100_Pass1
# 34 ldamodel_T100_Aauto0.005_Eta0.010_Off100.000_Dec0.500_Chunk100_Pass1
    odcSet = [ (1000, 0.625, 100) , (100, 0.625, 100), (100, 0.5, 100) ]
    for T in num_topics:
        for A in alphas:
            for E in eta:
                for p in passes:
                    for off, d, c in odcSet:
                        o = ModelOptions(num_topics=T, alpha=A[0], alpha_init=A[1],
                                         eta=E, offset=off, decay=d, chunksize=c, passes=p)
                        options.append(o)
                        print modelLabel(o)
    return options


class ModelOptions():
    'LdaModel options used for grid search'
    def __init__(self, num_topics, alpha, alpha_init, eta,
                 offset, decay, chunksize, passes, label = '', seed = 12345):
        #todo keep arguments in a (kword) map
        self.num_topics = num_topics
        self.alpha = alpha; self.alpha_init = alpha_init
        self.eta = eta; self.offset = offset
        self.decay = decay; self.chunksize = chunksize
        self.passes = passes
        self.label = label; self.seed = seed
        self.eval_passes = None; self.eval_results = None

    def getModel(self, dict):
        'create new LdaModel setup with this options'
        return LdaModelMod(id2word=dict, num_topics=self.num_topics, alpha=self.alpha,
                        alpha_init=self.alpha_init, eta=self.eta, offset=self.offset,
                        decay=self.decay, chunksize=self.chunksize, passes=self.passes,
                        seed = self.seed, eval_passes=self.eval_passes, eval_results=self.eval_results)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return modelLabel(self) == modelLabel(other)
            #return self.__dict__ == other.__dict__
        else:
            return False