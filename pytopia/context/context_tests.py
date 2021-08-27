from pytopia.context.GlobalContext import GlobalContext
from pytopia.context.Context import Context
from pytopia.context.ContextResolver import resolve

def testStacking1(printGlobal=True):
    ctx1 = Context('context1')
    ctx1['obj1'] = 'obj1'
    ctx2 = Context('context2')
    ctx2['obj2'] = 'obj2'
    assert resolve('obj1') == None
    if printGlobal: print GlobalContext.get()
    with ctx1:
        if printGlobal: print GlobalContext.get()
        assert resolve('obj1') == 'obj1'
        assert resolve('obj2') == None
        with ctx2:
            if printGlobal: print GlobalContext.get()
            assert resolve('obj2') == 'obj2'
        assert resolve('obj2') == None
    assert resolve('obj1') == None

def testContext():
    class Cls():
        def __init__(self, id): self.id = id
    o1, o2 = Cls('o1'), Cls('o2')
    with Context('ctxid', o1, o2) as ctx:
        assert ctx.id == 'ctxid'
        assert resolve(o1) == o1
        assert resolve(o2) == o2

if __name__ == '__main__':
    testStacking1()