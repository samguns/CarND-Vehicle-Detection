import numpy as np


class Vehicle():
    def __init__(self):
        self.detected = False # was the Vehicle detected in the last iteration?
        self.n_detections = 0 # Number of times this vehicle has been?
        self.n_nondetections = 0 # Number of consecutive times this car has not been detected
        self.xpixels = None # Pixel x values of last detection
        self.ypixels = None # pixel y values of last detection
        self.recent_xfitted = [] # x position of the last n fits
        self.bestx = None # average x position of the last n fits
        self.recent_yfitted = [] # y position of the last n fits
        self.besty = None # average y position of the last n fits
        self.recent_wfitted = [] # width of the last n fits of the bounding box
        self.bestw = None # average width of the last n fits
        self.recent_hfitted = [] # height of the last n fits of the bounding box
        self.besth = None # average height of the last n fits


    def update_detection(self, xpixels, ypixels):
        self.xpixels = xpixels
        self.ypixels = ypixels
        x = np.min(self.xpixels)
        y = np.min(self.ypixels)
        w = np.max(self.xpixels)
        h = np.max(self.ypixels)
        self.recent_xfitted.append(x)
        self.recent_yfitted.append(y)
        self.recent_wfitted.append(w)
        self.recent_hfitted.append(h)

        self.n_detections += 1
        self.detected = True
        return


    def get_bbox(self):
        if self.n_detections > 5:
            self.bestx = int(sum(self.recent_xfitted[:-self.n_detections]) / self.n_detections)
            self.besty = int(sum(self.recent_yfitted[:-self.n_detections]) / self.n_detections)
            self.bestw = int(sum(self.recent_wfitted[:-self.n_detections]) / self.n_detections)
            self.besth = int(sum(self.recent_hfitted[:-self.n_detections]) / self.n_detections)
            bbox = ((self.bestx, self.besty), (self.bestx+self.bestw, self.besty+self.besth))
            return True, bbox

        return False, None


    def update_nondetection(self):
        self.n_nondetections += 1