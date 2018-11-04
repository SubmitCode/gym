import unittest
import src.Helper as hlp
import numpy as np

class Test_Helper(unittest.TestCase):
    def test_preProcessingImage(self):
        helper = hlp.Helper()
        testImg = np.load('./test/testPic.npy')        
        img = helper.preProcessImage(testImg, startRow=20, endRow=-20, startCol=20, endCol=-20)
        self.assertEqual(len(img.shape), 2) # check if its gray
        
    


if __name__ == '__main__':
    unittest.main()