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
        if (w - x) < 40:
            self.recent_wfitted.append(x+40)
        else:
            self.recent_wfitted.append(w)

        if (h - y) < 40:
            self.recent_hfitted.append(y+40)
        else:
            self.recent_hfitted.append(h)

        self.recent_xfitted = self.recent_xfitted[-10:]
        self.recent_yfitted = self.recent_yfitted[-10:]
        self.recent_wfitted = self.recent_wfitted[-10:]
        self.recent_hfitted = self.recent_hfitted[-10:]

        self.n_detections += 1
        self.n_nondetections = 0
        self.detected = True
        return


    def get_bbox(self):
        if not self.detected:
            print("Non-Detect", self.n_nondetections, self.recent_xfitted, self.recent_yfitted)
            self.n_nondetections += 1

        if self.n_nondetections >= 10:
            self.n_detections = 0
            self.n_nondetections = 0
            print("Remove", self.recent_xfitted, self.recent_yfitted)
            self.recent_xfitted = []
            self.recent_yfitted = []
            self.recent_wfitted = []
            self.recent_hfitted = []


        if self.n_detections != 0:# and self.n_detections > 2:
            self.bestx = np.mean(self.recent_xfitted, axis=0).astype(int)
            self.besty = np.mean(self.recent_yfitted, axis=0).astype(int)
            self.bestw = np.mean(self.recent_wfitted, axis=0).astype(int)
            self.besth = np.mean(self.recent_hfitted, axis=0).astype(int)
            print("deteced", self.n_detections, len(self.recent_xfitted), self.recent_xfitted, "selft.bestx/y", self.bestx, self.besty)
            bbox = ((self.bestx, self.besty), (self.bestw, self.besth))

            self.detected = False

            return True, bbox

        return False, None