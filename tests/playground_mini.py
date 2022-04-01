class T:
    def __init__(self, delegatee):
        self.delegatee = delegatee

    def __getattr__(self, name):
        return getattr(self.delegatee, name)

    def access(self):
        return self.foo


class D:
    def __init__(self):
        self.foo = "bar"


d = D()
t = T(d)
r = t.access()
print(r)
