class yoink(object):
    def __new__(cls, *args, **kwargs):
        hey = super().__new__(cls)
        raw_init = cls.__dict__["__init__wrapped__"]
        if raw_init is not None:
            if type(raw_init) != type(lambda x: x):
                init = raw_init.__func__ # unwrap
            else:
                init = raw_init

            init(hey, *args, **kwargs) 
        return hey

class Lollolol(yoink):
    @staticmethod
    def __init__wrapped__(idk):
        idk.lol = 50

    def test(burger):
        return burger.lol

lmfao = Lollolol()
print(lmfao.test())


# slots
class A:
    __slots__ = ("x",)
    def __init__(self):
        self.x = 10


class B(A):
    __slots__ = ("y",)

a = A()
c = A()
b = B()
b.x = 1
b.y = 2

print(b.x, b.y, a.x)

c.x = 3

print(b.x, b.y, a.x)
