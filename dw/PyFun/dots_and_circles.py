import numpy as np
import cv2
import imageio
from numba import jit
from matplotlib import pyplot as plt

images = []

size = np.array((500, 500))
canvas = np.zeros((size[0], size[1], 3), dtype=np.ubyte)
CW = np.ones(3) * 255
draw_scale = 4


@jit(nopython=True)
def draw_circle_smooth(cx, cy, radius, thickness, ci, I):
    h, w = I.shape[:2]
    rs = radius ** 2
    for x in range(h):
        dxs = (x - cx) ** 2
        for y in range(w):
            dys = (y - cy) ** 2
            if abs(dxs + dys - rs) < thickness:
                I[x, y, ci] += 255


def cos_lerp(source, target, frac):
    f = -0.5 * np.cos(frac * (np.pi)) + 0.5
    return source + f * (target - source)


class Circle():
    def __init__(self, color):
        self.center = np.zeros(2, dtype=np.float32)
        self.radius = 0.0
        self.color = color

    def tick(self, p1, p2, p3):
        x1, y1 = p1.pos[0], p1.pos[1]
        x2, y2 = p2.pos[0], p2.pos[1]
        x3, y3 = p3.pos[0], p3.pos[1]
        x1s, y1s = x1 ** 2, y1 ** 2
        x2s, y2s = x2 ** 2, y2 ** 2
        x3s, y3s = x3 ** 2, y3 ** 2

        A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
        B = (x1s + y1s) * (y3 - y2) + (x2s + y2s) * (y1 - y3) + (x3s + y3s) * (y2 - y1)
        C = (x1s + y1s) * (x2 - x3) + (x2s + y2s) * (x3 - x1) + (x3s + y3s) * (x1 - x2)
        D = (x1s + y1s) * (x3 * y2 - x2 * y3) + (x2s + y2s) * (x1 * y3 - x3 * y1) + (x3s + y3s) * (x2 * y1 - x1 * y2)

        self.center[0] = -B / (2 * A)
        self.center[1] = -C / (2 * A)
        self.radius = ((B ** 2 + C ** 2 - 4 * A * D) / (4 * A ** 2)) ** 0.5
        self.big_canvas = np.zeros((size[0] * draw_scale, size[1] * draw_scale, 3), np.ubyte)

    def draw(self, I):
        t = 30 - self.radius / 80
        if t > 0.0:
            draw_circle_smooth(self.center[0], self.center[1], self.radius, 200, 0, I)
            # self.big_canvas *= 0
            # cv2.circle(self.big_canvas,
            #            (int(self.center[0] * draw_scale), int(self.center[1] * draw_scale)),
            #            int(self.radius * draw_scale), self.color, thickness=t * draw_scale, lineType=cv2.LINE_AA)
            # self.small_canvas = cv2.resize(self.big_canvas, (size[0], size[1]), interpolation=cv2.INTER_CUBIC).astype(
            #     np.ubyte)
            # I += self.small_canvas  # / 3).astype(np.ubyte)


class Point():
    def __init__(self):
        self.source = np.random.rand(2) * size
        self.target = np.random.rand(2) * size
        self.pos = np.copy(self.source)
        self.steps = 200
        self.step = 0

    def tick(self):
        self.step += 1
        self.pos = cos_lerp(self.source, self.target, self.step / self.steps)
        if self.step >= self.steps:
            self.source = np.copy(self.target)
            self.target = np.random.rand(2) * size
            self.steps = 200
            self.step = 0

    def draw(self, I):
        pass
        # cv2.circle(I, (int(self.pos[0]), int(self.pos[1])), 5, (255, 255, 255))



def run(out_path):

    points = [Point() for i in range(4)]

    circles = [Circle(c) for c in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]]
    circles[0].tick(points[0], points[1], points[2])
    circles[1].tick(points[3], points[1], points[2])
    circles[2].tick(points[0], points[3], points[2])

    frames = 1000
    frame = 0
    out = cv2.VideoWriter(out_path + 'noise.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (128, 128))

    while frame < frames or True:
        frame += 1
        canvas = canvas * 0.0

        for point in points:
            point.tick()
        circles[0].tick(points[0], points[1], points[2])
        circles[1].tick(points[3], points[1], points[2])
        circles[2].tick(points[0], points[3], points[2])

        for point in points:
            point.draw(canvas)
        for circle in circles:
            circle.draw(canvas)

        canvas_b = np.clip(canvas, 0, 255).astype(np.ubyte)

        # images.append(I)
        # out.write(I)

        cv2.imshow('frame', canvas_b)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    out.release()
    imageio.mimsave(out_path + 'noise.gif', images)



if __name__ == '__main__':
    run('./')