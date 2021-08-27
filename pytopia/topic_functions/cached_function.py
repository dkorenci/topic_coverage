import cPickle
from os import path
from atexit import register

from pytopia.utils.logging_utils.setup import *
from pytopia.utils.logging_utils.tools import fullClassName

def saveCache(cache, cfile):
    '''
    Function saving CachedFunction's cache - made for registering
    it for execution at program exit.
    '''
    if cache:
        cPickle.dump(cache, open(cfile, 'wb'))

class CachedFunction():
    #todo synchronization in the case two CachedCohs write to same file in parallel

    def __init__(self, function, cacheFolder='.', saveEvery=50, verbose=False):
        '''
        :param function: callable with .id property
        :param cacheFolder: file for saving computed values
        :param saveEvery: save cache for saveEvery new coherence calculations
                    cache is always saved when object is destroyed
        '''
        self.cacheFolder = cacheFolder
        self.cacheFile = path.join(cacheFolder, self.__hid(function.id) + '.pickle')
        self.__cache = None
        self.__log = createLogger(fullClassName(self), INFO)
        self.function = function
        self.saveEvery = saveEvery
        self.newCalcCnt = 0
        if hasattr(function, 'measure'):
            self.measure = function.measure
        if verbose: self.logCacheInitMessage()
        #print self.__cacheInitMessage()

    @property
    def id(self): return self.function.id

    def __cacheInitMessage(self):
        msg = 'Cache Function Loaded\n'
        msg += 'function id: %s\n' % self.function.id
        msg += 'cache folder: %s\n' % self.cacheFolder
        msg += 'cache file: %s\n' % (self.__hid(self.function.id) + '.pickle')
        return msg

    def logCacheInitMessage(self):
        self.__log.info(self.__cacheInitMessage())

    def __hid(self, id):
        '''
        Since id can be used to create a file with the name corresponding to id,
        and ids can be longer than max. allowed file size, this function
        produces a hash of a string id, that is shorter then max. file size.
        Hash is reproducible across runs and machines.
        Possibilities of collision should be astronomically small.
        '''
        from hashlib import pbkdf2_hmac
        h = pbkdf2_hmac('sha512', str(id), str(self.__class__.__name__),
                        10000, dklen=50)
        hid = 'hid'+''.join('%d'%ord(b) for b in h)
        return hid

    def __call__(self, *args, **kwargs):
        '''
        :param words: list of whitespace separated strings
        :return:
        '''
        pid = self.__paramId(*args, **kwargs)
        coh = self.__load(pid)
        if coh is None:
            coh = self.function(*args, **kwargs)
            self.__save(pid, coh)
        return coh

    def __paramId(self,  *args, **kwargs):
        '''
        Return unique string id composed of arguments, which can
         be used as an id for querying and saving to cache.
        '''
        def v2s(obj):
            '''value 2 string'''
            import types
            if obj == None: return None
            if hasattr(obj, 'id'):
                return obj.id
            else:
                if isinstance(obj, types.FunctionType):
                    return obj.__name__
                elif isinstance(obj, types.ClassType):
                    return obj.__name__
                else:
                    return str(obj)
        astr = ','.join( v2s(a) for a in args )
        kwastr = ','.join( '%s:%s'%(v2s(k), v2s(v)) for k, v in kwargs.iteritems() )
        id_ = 'ARGS[%s]_KWARGS[%s]'%(astr, kwastr)
        return id_

    def __load(self, pid):
        '''load coherence from cache'''
        if self.__cache is None: self.__loadCreateCache()
        return self.__cache[pid] if pid in self.__cache else None

    def __loadCreateCache(self):
        '''create cache or load it from file'''
        if path.exists(self.cacheFile):
            self.__cache = cPickle.load(open(self.cacheFile, 'rb'))
        else:
            self.__cache = {}
        # register save at program exit since garbage collect is not guaranteed to happen
        register(saveCache, self.__cache, self.cacheFile)

    def __save(self, pid, coh):
        if pid in self.__cache: return
        self.__cache[pid] = coh
        self.newCalcCnt += 1
        if self.newCalcCnt % self.saveEvery == 0:
            self.saveCache()
            self.newCalcCnt = 0

    def saveCache(self):
        if self.__cache and self.newCalcCnt > 0:
            cPickle.dump(self.__cache, open(self.cacheFile, 'wb'))

    def __del__(self):
        self.saveCache()
        if self.__cache is not None: self.__cache.clear()

    @staticmethod
    def unite(target, sources, checkEquality=True):
        '''
        Add all the param->value mappings from sources cached functions
        to targed cached function and save the result.
        :param target: cached function
        :param sources: list of cached functions
        :param checkEquality: if true, check equality of values for duplicated source keys
        :return:
        '''
        if not isinstance(target, CachedFunction): return
        for s in sources:
            if not isinstance(s, CachedFunction): return
        target.__loadCreateCache()
        for s in sources:
            s.__loadCreateCache()
            if checkEquality:
                for k, v in s.__cache.iteritems():
                    if k in target.__cache:
                        assert target.__cache[k] == s.__cache[k]
            target.__cache.update(s.__cache)
            print len(s.__cache)
        target.newCalcCnt = 1 # without, it will not save
        target.saveCache()

def testHash():
    from hashlib import pbkdf2_hmac
    for p in ['a long id1', 'a long id2']:
        h = pbkdf2_hmac('sha512', str(p), str(CachedFunction.__name__), 10000,
                        dklen=50)
        print type(h), len(h)
        hid = 'hid' + ''.join('%d' % ord(b) for b in h)
        print hid, len(hid)
    #for i in h: print type(i[0]), ord(i)

if __name__ == '__main__':
    testHash()