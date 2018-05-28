import numpy as np


class Vehicle:
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


    def new_detection(self, xpixels, ypixels):
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
        return


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
        if self.detected == False:
            self.n_nondetections += 1

        if self.n_nondetections >= 2:
            self.n_detections = 0
            self.recent_xfitted = []
            self.recent_yfitted = []
            self.recent_wfitted = []
            self.recent_hfitted = []


        if self.detected:# and self.n_detections > 3:
            self.bestx = np.min(self.xpixels)
            self.besty = np.min(self.ypixels)
            self.bestw = np.max(self.xpixels)
            self.besth = np.max(self.ypixels)

            # self.bestx = np.mean(self.recent_xfitted, axis=0).astype(int)
            # self.besty = np.mean(self.recent_yfitted, axis=0).astype(int)
            # self.bestw = np.mean(self.recent_wfitted, axis=0).astype(int)
            # self.besth = np.mean(self.recent_hfitted, axis=0).astype(int)
            # self.bestx = int(sum(self.recent_xfitted[-5:]) / 5)
            # self.besty = int(sum(self.recent_yfitted[-5:]) / 5)
            # self.bestw = int(sum(self.recent_wfitted[-5:]) / 5)
            # self.besth = int(sum(self.recent_hfitted[-5:]) / 5)
            bbox = ((self.bestx, self.besty), (self.bestw, self.besth))

            self.detected = False

            return True, bbox

        return False, None