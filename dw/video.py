import imageio
import numpy as np
import cv2

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

def gif2mp4(in_path, out_path, frame_rate=25):
    gif = imageio.get_reader(in_path)
    for i, frame in enumerate(gif):
        if i == 0:
            hw = frame.shape[:2]
            out = cv2.VideoWriter(out_path,0x7634706d , frame_rate, hw)
            if frame.ndim == 3:
                d = frame.shape[2]
            else:
                d = 1
        if d >= 3:
            im = frame[:, :, :3][:, :, ::-1]
        else:
            im = np.dstack((frame,frame,frame))
        out.write(im)
    out.release()
