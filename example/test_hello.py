import unittest

class TestHello(unittest.TestCase):

    def test_hello_runs(self):
        from example.hello import hello
        hello()
