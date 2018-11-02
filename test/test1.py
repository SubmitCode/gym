import unittest

class Test_TestIncrementDecrement(unittest.TestCase):
    def test_increment(self):
        self.assertEquals(4, 4)

    def test_decrement(self):
        self.assertEquals(3, 4)

if __name__ == '__main__':
    unittest.main()