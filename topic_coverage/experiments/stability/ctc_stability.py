from pytopia.tools.IdComposer import IdComposer

class ModelmatchCtc(IdComposer):

    def __init__(self, ctc):
        self.ctc = ctc
        IdComposer.__init__(self)

    def __call__(self, m1, m2):
        return (self.ctc(m1, m2) + self.ctc(m2, m1))/2.0

