import unittest
import src.Helper as hlp
import numpy as np
import tensorflow as tf

class Test_DQNetwork(unittest.TestCase):
    def test_Tensorflow(self):
        tf.res
        with tf.Session() as sess:
            res = sess.run(1+1)
        
        self.assertEqual(2, res)

if __name__ == '__main__':
    unittest.main()