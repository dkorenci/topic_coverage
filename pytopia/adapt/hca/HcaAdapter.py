from pytopia.topic_model.TopicModel import TopicModel, Topic
from pytopia.tools.IdComposer import IdComposer
from pytopia.context.ContextResolver import resolveIds, resolve
from pytopia.resource.ResourceBuilder import SelfbuildResourceBuilder
from pytopia.utils.file_utils.location import FolderLocation as loc
from pytopia.resource.loadSave import pickleObject

from pytopia.utils.logging_utils.setup import *
from pytopia.utils.logging_utils.tools import fullClassName, logFile

import codecs, numpy as np, os, sys
from subprocess import call
from os import path
import tempfile

class HcaAdapter(TopicModel, IdComposer):
    '''
    Adapter to pytopia TopicModel interface of hca topic modeling package:
        http://mloss.org/software/view/527/
    Tested with hca 0.63.
    Needs a pointer to a hca executable, plus a tmp folder for input/output.
    '''

    def __init__(self, corpus, dictionary, text2tokens, T, C=100, Cme=50, rseed=00613,
                 burnin=20, Bme=20, type=None, label=None, A=None, B=None, S=None,
                 hcaLocation=None, tmpFolder=None, threads=1, params=None):
        '''
        :param type: predefined type, rewrites all other parameters
        :param label: additional model description
        :param burnin: number of gibbs cycles before the start of
                 sampling of discount and concentration hyperparams
        :param C: number of train cycles
        :param Cme: number of matrix (topic-word, doc-topic) estimation cycles
        :param Bme: number of burnin cycles for matrix estimation
        :param tmpFolder: either an existing folder path, or None in which case
                a temporary folder will be used for model inference data and deleted
        :param params: misc. model/modelbuild params depending on the model type, if
            any param occurs in params, its value takes precedent over the default value
        '''
        self.corpus, self.dictionary, self.text2tokens = \
            resolveIds(corpus, dictionary, text2tokens)
        self.T, self.rseed, self.C, self.Cme, self.burnin, self.Bme = T, rseed, C, Cme, burnin, Bme
        self.hcaLocation = hcaLocation; self._tmpFolder = tmpFolder
        self._threads = threads
        atts = ['corpus', 'dictionary', 'text2tokens', 'T', 'rseed', 'C', 'Cme', 'Bme', 'burnin']
        if params is not None:
            for p, v in params.iteritems():
                atts.append(p)
                setattr(self, p, v)
            self._params = params
        if type in ['lda', 'lda-asym', 'hdp', 'pyp-doctop', 'pyp']:
            self.type = type
            if type == 'lda': self.__setupLda()
            elif type == 'lda-asym': self.__setupLdaAsym()
            elif type == 'hdp': self.__setupHdp()
            elif type == 'pyp-doctop': self.__setupPypDoctop()
            elif type == 'pyp': self.__setupPyp()
            IdComposer.__init__(self, atts + ['type'])
        else:
            self.type = None
            IdComposer.__init__(self, atts + ['label', 'A', 'B', 'S'])
        self._log = createLogger(fullClassName(self), INFO)

    ### TopicModel interface methods

    def topicIds(self): return range(self.T)

    def numTopics(self): return self.T

    def topic(self, topicId):
        return Topic(self, topicId, self.topicVector(topicId))

    def topicVector(self, topicId):
        return self._topicWord[topicId]

    def inferTopics(self, txt, batch=False, format='tokens'):
        raise NotImplementedError()

    def corpusTopicVectors(self):
        return self._docTopic

    def __setupLda(self):
        ''' Setup hca params for std. lda with symmetric dirichlet priors on doc-topic and topic-word. '''
        self.A = '%g' % (50.0/self.T) # after 'Finding scientific topics' article
        self.B = '0.01'

    def __setupLdaAsym(self):
        ''' Setup hca params for lda with asymmetric learnable diricheld priors on doc-topic dist.  '''
        self.A = 'ng' # normalized gamma prior on each topic/dimension of doc-topic dist prior
        self.B = '0.01'

    def __setupHdp(self):
        '''Setup hca params for hdp with hierarch.dir.process prior on topics,
         and standard symmetric dirichled prior on topic-words distribution. '''
        self.A = 'hdp'; self.B = '0.01'

    def __setupPypDoctop(self):
        ''' Setup hca params for nonparam.hierarch. PYP prior on topics,
        params relating to doc-topics distr. only, not topic-word. '''
        self.A, self.B = 'hpdd', '0.01'

    def __setupPyp(self):
        ''' Setup hca params for nonparam.hierarch. PYP prior on both topic-word and doc-topic. '''
        self.A, self.B = 'hpdd', 'hpdd'

    def __fetchValue(self, param, ctxHandle):
        ''' Fetch value either from class param or pytopia context '''
        v = None
        if param is not None and hasattr(self, param):
            v = getattr(self, param)
            if v is not None: return v
        v = resolve(ctxHandle)
        if v is not None: return v
        raise Exception('could not fetch value from param, ctx == %s, %s' % (param, ctxHandle))

    def __hyparamInitSampleOpts(self, init, sample):
        ''' Sets hca options for initialization and sampling of hyperparameter variables.
        :param init: map variable name -> variable value
        :param sample: map variable name -> cyc (variable is sampled every cyc cycles)
        '''
        o = ''
        for var, val in init.iteritems():
            o += ' -S%s=%g' % (var, float(val))
            o += ' -G%s,%d,%d' % (var, sample[var+'s'], self.burnin)
        return o

    def __updateParams(self, params):
        '''
        Update params with self._params, use to overwrite default values.
        :param params: parameter map
        '''
        if hasattr(self, '_params'):
            for p, v in self._params.iteritems():
                if p in params: params[p] = v

    def __aldaOptions(self):
        ''' Prior parameter sampling options for asymmetric LDA '''
        init = { 'a':0.5, 'b':10 }
        sample = { 'as':11, 'bs':3 }
        return self.__hyparamInitSampleOpts(init, sample)

    def __hdpOptions(self):
        ''' Prior parameter sampling options for HDP variant '''
        init = { 'b':10, 'b0':10 }
        sample = { 'bs':3, 'b0s':3 }
        return self.__hyparamInitSampleOpts(init, sample)

    def __pypDoctopOptions(self):
        ''' Learning algo opts for fully nonparametric Pitman Yor model '''
        init = { 'a':0.5, 'b':10, 'a0':0.5, 'b0':10 }
        sample = { 'as':11, 'bs':3, 'a0s':11, 'b0s':3 }
        return self.__hyparamInitSampleOpts(init, sample)

    def __pypOptions(self):
        ''' Learning algo opts for fully nonparametric Pitman Yor model '''
        init = { 'a':0.5, 'b':10, 'a0':0.5, 'b0':10,
                 'aw':0.5, 'bw':10, 'aw0':0.5, 'bw0':100 }
        self.__updateParams(init)
        sample = { 'as':11, 'bs':3, 'a0s':11, 'b0s':3,
                   'aws':11, 'bws':3, 'aw0s':11, 'bw0s':3 }
        self.__updateParams(sample)
        return self.__hyparamInitSampleOpts(init, sample)

    def __diagnosticOptions(self):
        # shutdown checkpoints diagnostic by setting impossbile start cycle
        self._diagOpts = '-v -v -c%d -lprog,15,15 -lalpha,15,15' % (self.burnin+self.C+self.Cme+self.Bme+10)

    def __createOptions(self):
        ''' Create options for hca executable from relevant object params. '''
        # hyperparams
        o = '-K%d' % self.T
        if self.A: o += ' -A%s' % self.A
        if self.B: o += ' -B%s' % self.B
        if self.type == 'lda-asym': o += self.__aldaOptions()
        elif self.type == 'hdp': o += self.__hdpOptions()
        elif self.type == 'pyp-doctop': o += self.__pypDoctopOptions()
        elif self.type == 'pyp': o += self.__pypOptions()
        if self.type in ['lda-asym', 'hdp', 'pyp-doctop', 'pyp']:
            # for these variants, burnin is used for sampling of hyperparams
            # so total number of learn cycles is extended to account for burnin
            learnCycles = self.C + self.burnin
        else: learnCycles = self.C # burnin is ignored
        o += ' -C%d -q%d -s%d' % (learnCycles, self._threads, self.rseed)
        self._options = o

    def cmdline(self):
        ''' Create bash command for model building. '''
        exe = self.__fetchValue('hcaLocation', 'hca_location')
        self.__createOptions(); self.__createIoPrefixes(); self.__diagnosticOptions()
        cmd = '%s %s %s %s %s' % (exe, self._diagOpts, self._options,
                                  self._inputPrefix, self._outputPrefix)
        return cmd

    def cmdlineetsim(self):
        ''' Create bash command for matrix estimation, to be run after cmdline(). '''
        exe = self.__fetchValue('hcaLocation', 'hca_location')
        opts = '-r0 -C%(totalCycles)d -ltheta,3,%(startEstim)d -lphi,3,%(startEstim)d '\
               '-q%(threads)d -s%(rseed)d' % \
               {'totalCycles':self.Cme+self.Bme, 'startEstim':self.Bme+1,
                'threads':self._threads, 'rseed':self.rseed}
        self.__createIoPrefixes(); self.__diagnosticOptions()
        cmd = '%s %s %s %s %s' % (exe, self._diagOpts, opts, self._inputPrefix, self._outputPrefix)
        return cmd

    def __createIoPrefixes(self):
        ''' create prefixes for hca in/out files based on tmp folder '''
        self._inputPrefix = path.join(self._tmpFolder, 'in')
        self._outputPrefix = path.join(self._tmpFolder, 'out')

    def __clearTmpFolder(self):
        for f in loc(self._tmpFolder).files(): os.remove(f)

    def __delTmpFolder(self):
        import shutil
        shutil.rmtree(self._tmpFolder, ignore_errors=True)

    # requires bow_corpus_builder
    def __corpusToInputFile(self):
        ''' Create input file for hca in lda-c format, in tmp folder. '''
        bowc = resolve('bow_corpus_builder')(corpus=self.corpus, text2tokens=self.text2tokens,
                                             dictionary=self.dictionary)
        self._numdoc = len(bowc)
        self.__createIoPrefixes()
        f = codecs.open(self._inputPrefix + '.ldac', 'w', 'utf-8')
        maxw = 0
        for i, doc in enumerate(bowc):
            line = ' '.join('%d:%d'%(wi,wc) for wi, wc in doc)
            line = '%d %s\n' % (len(doc), line)
            f.write(line)
            for wi, wc in doc:
                if wi > maxw: maxw = wi
        self._maxw = maxw # max. word index in the corpus == columns in hca topic matrix
        f.close()

    def __readTopicWordMatrix(self):
        # check size
        fname = self._outputPrefix + '.phi'
        sz = os.path.getsize(fname)
        # 3 32bit ints at the start + topic matrix of dim. num.topic * num.words of 32bit ints
        savedTopics = (sz/4-3)/(self._maxw+1)
        assert sz == 3 * 4 + 4 * (self._maxw + 1) * savedTopics
        # assert sz == 3*4 + 4*(self._maxw+1) * self.T
        self._log.info('saved topics: %g' % savedTopics)
        self._log.info('maxw: %d' % self._maxw)
        assert savedTopics >= self.T
        # read
        from bitstring import ConstBitStream
        s = ConstBitStream(filename=fname, length=sz*8)
        s.read(32*3) # read 3 ints from the start
        d = resolve(self.dictionary) # align matrix dimensions with model's dictionary
        tmatrix = np.zeros((self.T, d.maxIndex()+1), np.float32)
        for t in range(self.T):
            for w in range(self._maxw+1): tmatrix[t, w] = s.read('floatne:32')
        #print tmatrix[0].sum(), tmatrix.sum(), tmatrix.sum()/self.T
        self._topicWord = tmatrix

    def __readDocTopicMatrix(self):
        fname = self._outputPrefix + '.theta'
        self._docTopic = np.zeros((self._numdoc, self.T), np.float32)
        docind = []
        for i, l in enumerate(codecs.open(fname, 'r')):
            nums = l.strip().split()
            doci = int(nums[0][:-1]); docind.append(doci)
            topind = []
            for n in nums[1:]:
                if n == '=': break
                nc = n.split(':')
                ti, tw = int(nc[0]), float(nc[1])
                self._docTopic[doci, ti] = tw
                topind.append(ti)
            # assert sorted(topind) == range(self.T)
        assert i == self._numdoc-1
        assert sorted(docind) == range(self._numdoc)
        #print self._docTopic[self._numdoc/3].sum(), self._docTopic.sum()/self._numdoc

    def __logHcaLog(self):
        ''' Copy hca build log to logger folder. '''
        fname = path.join(self._tmpFolder, self._outputPrefix+'.log')
        #corpus, dictionary, text2tokens, T, C = 100, Cme = 50, rseed = 00613,
        #burnin = 20, Bme = 20, type = None,
        shortid='HCA_type[%s]_corp[%s]_dict[%s]_tok[%s]_T[%d]_C[%d]_B[%d]_Cme[%d]_Bme[%d]_rnd[%d]' %\
                (self.type, self.corpus, self.dictionary, self.text2tokens,
                 self.T, self.C, self.burnin, self.Cme, self.Bme, self.rseed)
        newfname = shortid+'.log'
        res = logFile(fname, newfname)
        message = '%s saved build log' % ('successfully' if res else 'unsuccessfully')
        self._log.info(message)

    def __copyTmpFolder(self):
        res = logFile(self._tmpFolder, 'hca_tmp_folder')
        message = '%s copied tmp folder' % ('successfully' if res else 'unsuccessfully')
        self._log.info(message)

    def build(self):
        from time import time
        if self._tmpFolder: self.__clearTmpFolder(); delTmp = False
        else: self._tmpFolder = tempfile.mkdtemp(); delTmp = True
        try:
            self.__corpusToInputFile()
            self._log.info('train cmd: %s' % self.cmdline())
            t0 = time()
            call(self.cmdline(), shell=True)
            self._log.info('time 4 train: %g' % (time() - t0))
            self._log.info('matrix estim. cmd: %s' % self.cmdlineetsim())
            t0 = time()
            call(self.cmdlineetsim(), shell=True)
            self._log.info('time 4 matrix estim: %g' % (time() - t0))
            self.__readTopicWordMatrix()
            self.__readDocTopicMatrix()
            self.__logHcaLog()
            if delTmp: self.__delTmpFolder()
        except:
            from traceback import format_exception
            e = sys.exc_info()
            strace = ''.join(format_exception(e[0], e[1], e[2]))
            self._log.error('build for model %s failed' % self.id)
            self._log.error('temporary folder: %s' % self._tmpFolder)
            self._log.error('stacktrace:\n%s'%strace)
            raise

    ### persistence

    def __getstate__(self):
        return IdComposer.__getstate__(self), self.hcaLocation, self._threads

    def __setstate__(self, state):
        IdComposer.__setstate__(self, state[0])
        self.hcaLocation, self._threads = state[1:]

    def save(self, folder):
        pickleObject(self, folder)
        np.save(self.__topicWordsFname(folder), self._topicWord)
        np.save(self.__docTopicsFname(folder), self._docTopic)

    def load(self, folder):
        self._topicWord = np.load(self.__topicWordsFname(folder))
        self._docTopic = np.load(self.__docTopicsFname(folder))

    def __topicWordsFname(self, f): return path.join(f, 'topicWordMatrix.npy')
    def __docTopicsFname(self, f): return path.join(f, 'docTopicsMatrix.npy')

HcaAdapterBuilder = SelfbuildResourceBuilder(HcaAdapter)

def testCreate():
    hca = HcaAdapter('corpus', 'dictionary', 'text2tokens', 50, type='lda',
                     hcaLocation='/data/code/hca/HCA-0.63/hca/hca',
                     tmpFolder='/datafast/topic_coverage/test_hca/')
    print hca.id
    print hca.cmdline()

def testBitstring():
    from bitstring import ConstBitStream
    f = '/datafast/topic_coverage/test_hca/out.phi'
    s = ConstBitStream(filename=f)
    W = s.read(32)[::-1].int
    T = s.read(32)[::-1].int
    C = s.read(32)[::-1].int
    print W, T, C
    c = 0; tstr = ''; tsum = 0
    for t in range(T):
        for w in range(W):
            tw = s.read('floatne:32')
            tstr += '%.6f,' % tw
            tsum += tw
            c += 1
        print tsum
        print tstr
        return

if __name__ == '__main__':
    #testCreate()
    testBitstring()


