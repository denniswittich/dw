import numpy as np
import cv2
import imageio
from numba import jit
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import filedialog
from tkinter import _setit
import imageio
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from scipy import signal
import random

images = []


@jit(nopython=True)
def draw_circle_smooth(cx, cy, r, grabbed, I):
    r2 = 2 * r
    rs = r ** 2
    for x in range(cx - r2, cx + r2 + 1):
        dxs = (x - cx) ** 2
        for y in range(cy - r2, cy + r2 + 1):
            sum = dxs + (y - cy) ** 2
            #f = 0.0
            if sum <= 2 * rs:
                #f = 1.0
                #if sum <= 2 * rs:
                f = 2 - sum / rs
                if grabbed:
                    I[x, y] += 230 * f
                else:
                    I[x, y] += 250 * f


@jit(nopython=True)
def draw_line_smooth(start_x, start_y, end_x, end_y, w, I):
    if start_x < end_x:
        x_low = start_x
        x_high = end_x
    else:
        x_low = end_x
        x_high = start_x
    if start_y < end_y:
        y_low = start_y
        y_high = end_y
    else:
        y_low = end_y
        y_high = start_y

    bx = end_x - start_x
    by = end_y - start_y
    wn = w * (bx * bx + by * by) ** 0.5

    dx = x_high - x_low
    dy = y_high - y_low

    if dx > dy:
        slope = by / bx
        for x in range(x_low, x_high):
            px = x - start_x
            y_mid = (x - start_x) * slope + start_y
            for y in range(y_mid - w, y_mid + w + 1):
                py = y - start_y
                d = abs(px * by - py * bx)

                if d < wn:
                    d = abs(px * by - py * bx) / ((bx * bx + by * by) ** 0.5)
                    I[x, y] += 200.0 * 0.5 ** d
    else:
        slope = bx / by
        for y in range(y_low, y_high):
            py = y - start_y
            x_mid = (y - start_y) * slope + start_x
            for x in range(x_mid - w, x_mid + w + 1):
                px = x - start_x
                d = abs(px * by - py * bx)

                if d < wn:
                    d = abs(px * by - py * bx) / ((bx * bx + by * by) ** 0.5)
                    I[x, y] += 200.0 * 0.5 ** d


class Line():

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.w = 5

    def draw(self, I):
        draw_line_smooth(self.start.pos[0], self.start.pos[1], self.end.pos[0], self.end.pos[1], self.w, I)


class Nob():
    grabbed = False

    def __init__(self, width):
        self.pos = np.random.randint(0 + 16, width - 16, size=(2,)).astype(np.float32)
        self.r = 12.0
        self.rs = self.r ** 2

    def draw(self, I):
        draw_circle_smooth(self.pos[0], self.pos[1], self.r / 2, self.grabbed, I)

    def grab(self, x, y):
        dx = abs(x - self.pos[0])
        if dx > self.r:
            return False
        dy = abs(y - self.pos[1])
        if dy > self.r:
            return False
        if dx * dx + dy * dy <= self.rs:
            self.grabbed = True
            return True

    def shift(self, delta):
        self.pos += delta

    def ungrab(self):
        self.grabbed = False


class mainWindow():

    def cos_lerp(source, target, frac):
        f = -0.5 * np.cos(frac * (np.pi)) + 0.5
        return source + f * (target - source)

    root = None  # --------------------------------------------------------------- tkinter gui root element
    width = 800  # --------------------------------------------------------------- side length of thumbnail images
    canvas = np.random.randint(0, 255, (width, width, 3)).astype(np.float)  # ------ canvas
    nobs = []  # ------------------------------------------------------------------ Nots
    lines = []  # ----------------------------------------------------------------- Edges

    l_prev = np.zeros((2,), dtype=np.int)
    grabbed = []

    def __init__(self):
        # ============== GUI ROOT ELEMENT ================

        self.root = Tk()
        self.root.resizable(0, 0)

        # ============== THUMBNAILS =================

        self.root.in_img = PhotoImage(width=self.width, height=self.width)
        self.canvas_input = Canvas(self.root, width=self.width, height=self.width, bg="#000000", cursor="plus")
        self.canvas_input.create_image((self.width, self.width), anchor='nw', image=self.root.in_img, state="normal")
        self.canvas_input.grid(row=2, column=0, rowspan=6, columnspan=5, padx=10, pady=10)

        self.canvas_input.bind("<Button-1>", self.l_down)
        self.canvas_input.bind("<B1-Motion>", self.l_motion)
        self.canvas_input.bind("<ButtonRelease-1>", self.l_up)
        self.canvas_input.bind("<Button-3>", self.r_down)
        self.canvas_input.bind("<B3-Motion>", self.r_motion)
        self.canvas_input.bind("<ButtonRelease-3>", self.r_up)
        self.canvas_input.bind("<Button-2>", self.m_down)

        # ============= LOADING MENU ===============

        Label(self.root, text="PyTangle", justify='left').grid(
            row=0, column=1, columnspan=3, sticky='NSEW', padx=5, pady=5)

        # =============== SLOT VARS =================

        Button(self.root, text="Start", command=self.initgame).grid(
            row=9, column=1, columnspan=3, sticky='NSEW', pady=5, padx=5)

        # =============== START GUI =================

        self.root.mainloop()

    def initgame(self, N=64):

        n = int(N ** 0.5)
        nobgrid = [[Nob(self.width) for i in range(n)] for j in range(n)]
        self.nobs = []
        self.lines = []

        p = 0.75

        for x in range(n):
            for y in range(n):
                self.nobs.append(nobgrid[x][y])
                if x > 0 and random.random() < p:
                    self.lines.append(Line(nobgrid[x][y], nobgrid[x - 1][y]))
                # if x < n - 1 and random.random() < p:
                #     self.lines.append(Line(nobgrid[x][y], nobgrid[x + 1][y]))
                if y > 0 and random.random() < p:
                    self.lines.append(Line(nobgrid[x][y], nobgrid[x][y - 1]))
                # if y < n - 1 and random.random() < p:
                #     self.lines.append(Line(nobgrid[x][y], nobgrid[x][y + 1]))
                if x > 0 and y > 0  and random.random() < p:
                    if random.random() > 0.5:
                        self.lines.append(Line(nobgrid[x][y], nobgrid[x - 1][y - 1]))
                    else:
                        self.lines.append(Line(nobgrid[x-1][y], nobgrid[x][y - 1]))

        # add border skipcons

        self.draw()

    def l_down(self, pos):
        self.l_prev[0] = pos.x
        self.l_prev[1] = pos.y
        if len(self.grabbed) > 0:
            return
        for nob in self.nobs:
            if nob.grab(pos.x, pos.y):
                self.grabbed.append(nob)
                break
        self.draw()

    def m_down(self, pos):
        xpos = [nob.pos[0] for nob in self.nobs]
        ypos = [nob.pos[1] for nob in self.nobs]
        x_min = min(xpos)
        x_max = max(xpos)
        y_min = min(ypos)
        y_max = max(ypos)

        offset = np.array((x_min,y_min))

        scale = np.array((0.8*self.width/(x_max-x_min),0.8*self.width/(y_max-y_min)))

        for nob in self.nobs:
            nob.pos -= offset
            nob.pos *=scale
            nob.pos += 0.1*self.width

        self.draw()


    def l_motion(self, pos):

        for nob in self.grabbed:
            nob.shift(np.array((pos.x - self.l_prev[0], pos.y - self.l_prev[1])))
        self.l_prev[0] = pos.x
        self.l_prev[1] = pos.y
        self.draw()

    def l_up(self, pos):
        # if len(self.grabbed) > 1:
        #     return
        for nob in self.grabbed:
            nob.ungrab()
        self.grabbed = []
        self.draw()

    def r_down(self, pos):
        for nob in self.grabbed:
            nob.ungrab()
        self.grabbed = []
        self.draw()

    def r_motion(self, pos):
        for nob in self.nobs:
            if nob in self.grabbed:
                continue
            if nob.grab(pos.x, pos.y):
                self.grabbed.append(nob)
        self.draw()

    def r_up(self, pos):
        # print('r_up', pos)
        # self.draw()
        pass

    def draw(self):
        self.canvas *= 0.0
        for line in self.lines:
            line.draw(self.canvas)
        for nob in self.nobs:
            nob.draw(self.canvas)
        im = Image.fromarray(np.clip(self.canvas, 5, 250).astype(np.ubyte).transpose((1, 0, 2)), mode='RGB')

        self.imTk_in = ImageTk.PhotoImage(im)
        self.canvas_input.create_image((2, 2), anchor='nw', image=self.imTk_in, state="normal")


if __name__ == "__main__":
    mainWindow()
