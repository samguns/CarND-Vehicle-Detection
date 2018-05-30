import numpy as np


class Vehicle:
    def __init__(self):
        self.detected = False
        self.n_detections = 0 # Number of times this vehicle has been?
        self.n_nondetections = 0 # Number of consecutive times this car has not been detected
        self.recent_ltx_fitted = [] # left top x position of the last n fits
        self.best_top_x = None # average left top x position of the last n fits
        self.recent_lty_fitted = [] # left top y position of the last n fits
        self.best_top_y = None # average left top y position of the last n fits
        self.recent_rbx_fitted = [] # right bottom x of the last n fits of the bounding box
        self.best_bottom_x = None # average right bottom x of the last n fits
        self.recent_rby_fitted = [] # right bottom y of the last n fits of the bounding box
        self.best_bottom_y = None # average right bottom y of the last n fits


    def update_detection(self, xpixels, ypixels):
        left_top_x = np.min(xpixels)
        left_top_y = np.min(ypixels)
        right_bottom_x = np.max(xpixels)
        right_bottom_y = np.max(ypixels)
        self.recent_ltx_fitted.append(left_top_x)
        self.recent_lty_fitted.append(left_top_y)

        width = right_bottom_x - left_top_x
        if width < 80:
            self.recent_rbx_fitted.append(left_top_x+40)
        else:
            self.recent_rbx_fitted.append(right_bottom_x)

        height = right_bottom_y - left_top_y
        if height < 80:
            self.recent_rby_fitted.append(left_top_y+40)
        else:
            self.recent_rby_fitted.append(right_bottom_y)

        self.recent_ltx_fitted = self.recent_ltx_fitted[-10:]
        self.recent_lty_fitted = self.recent_lty_fitted[-10:]
        self.recent_rbx_fitted = self.recent_rbx_fitted[-10:]
        self.recent_rby_fitted = self.recent_rby_fitted[-10:]

        self.n_detections += 1
        self.n_nondetections = 0
        self.detected = True
        return


    def get_bbox(self):
        if not self.detected:
            self.n_nondetections += 1

        if self.n_nondetections >= 10:
            self.n_detections = 0
            self.n_nondetections = 0
            self.recent_ltx_fitted = []
            self.recent_lty_fitted = []
            self.recent_rbx_fitted = []
            self.recent_rby_fitted = []

        if self.n_detections != 0:
            self.best_top_x = np.mean(self.recent_ltx_fitted, axis=0).astype(int)
            self.best_top_y = np.mean(self.recent_lty_fitted, axis=0).astype(int)
            self.best_bottom_x = np.mean(self.recent_rbx_fitted, axis=0).astype(int)
            self.best_bottom_y = np.mean(self.recent_rby_fitted, axis=0).astype(int)
            bbox = ((self.best_top_x, self.best_top_y), (self.best_bottom_x, self.best_bottom_y))

            self.detected = False

            return True, bbox

        return False, None