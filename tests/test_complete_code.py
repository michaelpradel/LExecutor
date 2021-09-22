from unittest import TestCase

class SimpleCodeTest(TestCase):
    def test_variable_read(self):
        x = 23
        y = x
        self.assertEqual(x, 23) # don't instrument
        self.assertEqual(y, 23) # don't instrument
        