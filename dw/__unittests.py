import dw.image as image
import numpy as np
import unittest

class TestImageMethods(unittest.TestCase):

    def test_sub_pixel(self):
        I = np.arange(0,9,dtype=np.float32).reshape((3,3,1))
        self.assertEqual(image.sub_pixel(I,0.0,0.0),np.array((0.0,0.0,0.0)))

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
