from pytopia.tools.IdComposer import IdComposer, deduceId

class FunctionComposition(IdComposer):
    '''
    Composition of a sequence of callables that are pytopia resources,
        also an identifiable pytopia resource.
    '''

    def __init__(self, *funcs):
        '''
        :param funcs: sequence of identifiable callables, to be applied
            on an argument from right to left (in 'mathematical' order)
        '''
        self.funcs = list(funcs)
        # create id composed of _fK[functionKid] strings
        # in order of application
        atts = []; N = len(self.funcs)
        for i, f in enumerate(self.funcs):
            att = 'f%d'%(N-i); atts.append(att)
            setattr(self, att, deduceId(self.funcs[i]))
        IdComposer.__init__(self, attributes=atts, class_='Composition')
        self.funcs.reverse() # functions will be applied from right to left

    def __call__(self, a):
        '''
        Apply functions sequentially on given argument.
        '''
        for f in self.funcs:
            a = f(a)
            if a is None: return None
        return a

if __name__ == '__main__':
    c = FunctionComposition(1,2,3)
    print c.id