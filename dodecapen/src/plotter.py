import numpy as np
import cv2
import pdb

UNITS = ['m', 'cm', 'mm']

class Plotter:
    def __init__(self, phy_size=None, origin=None, units='m'):
        # TODO: support automatic unit convertion?
        if units not in UNITS:
            raise TypeError("{} is not a valid unit".format(units))
        self._phy_size = phy_size
        self._origin = None
        self._canvas = np.ones((phy_size[0], phy_size[1], 3))
    
    def plot(self, X):
        for i in range(X.shape[0]):
            if self._origin is None:
                self._origin = -1*X[i,:] + np.array(list(self._phy_size[:1])) // 2
            coord = tuple((X[i,:] + self._origin).astype(np.int)) # TODO: implement pixel splatting instead of just casting to int
            color = self._get_zcolor(coord[2])
            print(color)
            self._canvas = cv2.circle(self._canvas, coord[:2], radius=5, color=color, thickness=-1)

    def _get_zcolor(self, z):
        w = (self._phy_size[2] - z) / self._phy_size[2]
        color = (int(255*w), int(255*w), int(255*w))
        return color

    def show(self):
        cv2.imshow('plotter', self._canvas)

    def plotshow(self, X):
        self.plot(X)
        self.show()