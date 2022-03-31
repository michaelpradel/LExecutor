class C:
    def m(self):
        self.b += 23
        print("Got here")

c=C()
c.m()
print(c.b)