import numpy as np
import cv2

UNITS = ['m', 'cm', 'mm']

class Plotter:
    def __init__(self, phy_size=None, origin=None, units='m'):
        # TODO: support automatic unit convertion?
        if units not in UNITS:
            raise TypeError("{} is not a valid unit".format(units))
        self._phy_size = phy_size
        self._origin = None
        self._canvas = np.ones(phy_size)
    
    def plot(self, X):
        for i in range(X.shape[0]):
            if self._origin is None:
                self._origin = -1*X[i,:] + np.array(list(self._phy_size[:1])) // 2
            coord = tuple((X[i,:] + self._origin).astype(np.int)) # TODO: implement pixel splatting instead of just casting to int
            print("raw: {} adjusted: {}".format(X[i,:].astype(np.int), coord))
            self._canvas = cv2.circle(self._canvas, coord, radius=10, color=(255,0,0), thickness=-1)

    def show(self):
        cv2.imshow('plotter', self._canvas)

    def plotshow(self, X):
        self.plot(X)
        self.show()