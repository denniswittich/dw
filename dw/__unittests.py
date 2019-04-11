import dw.image as dwi
import dw.eval as dwe
import numpy as np
import unittest

class TestImageMethods(unittest.TestCase):

    def test_sub_pixel(self):
        I = np.arange(0,9,dtype=np.float32).reshape((3,3,1))
        sp_1 = dwi.sub_pixel(I,0.0,0.0)
        sp_2 = dwi.sub_pixel(I,0.5,0.5)
        self.assertEqual(sp_1, 0.0)
        self.assertEqual(sp_2, 2.0)
        with self.assertRaises(AssertionError):
            dwi.sub_pixel(I, 1.0, -1.0)
        sp_3 = dwi.sub_pixel(I, 1.0, -1.0, allow_oob=True)
        self.assertEqual(sp_3, 3.0)
        with self.assertRaises(AssertionError):
            dwi.sub_pixel(I, 0.0, 4.0)
        sp_4 = dwi.sub_pixel(I, 0.0, 4.0, allow_oob=True)
        self.assertEqual(sp_4, 2.0)

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


class TestEvalMethods(unittest.TestCase):
    def test_confusion_update_and_metrics(self):
        cm = np.zeros((3, 3))
        predictions = np.array([[1,1,2], [0,2,1]])
        reference = np.array([[1,1,2], [0,1,2]])
        dwe.update_confusion_matrix(cm, predictions, reference)
        predictions = np.array([1])
        reference = np.array([0])
        dwe.update_confusion_matrix(cm, predictions, reference)
        metrics = dwe.get_confusion_metrics(cm)
        self.assertEqual(metrics['oa'], 4/7)

if __name__ == '__main__':
    unittest.main()
