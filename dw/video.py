import imageio
import numpy as np

class gif_writer:
    def __init__(self, range = [0,255]):
        self.imgs = []
        self.range = range

    def append(self, img):
        img = np.copy(img)
        assert isinstance(img, np.ndarray), "Function requres numpy array"
        assert img.ndim == 3, "Array must have three dimensions"
        assert img.shape[-1] == 3 or img.shape[-1] == 1 , "image must have one or three dimensions"
        if not img.dtype == np.ubyte:
            img -= self.range[0]
            img /= (self.range[1]-self.range[0])/255
            img.clip(0,255)
            img = img.astype(np.ubyte)
        self.imgs.append(img)

    def reset(self):
        del (self.imgs)
        self.imgs = []

    def write(self, path):
        if not path.endswith('.gif'):
            path += '.gif'
        imageio.mimwrite(path, self.imgs)
        print('gif written to ', path)


