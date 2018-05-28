import pickle
import numpy as np
from utils import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip



svm_model = pickle.load(open("svm_model.p", "rb"))
svc = svm_model["svc"]
X_scaler = svm_model["X_scaler"]


color_space = hog_params["color_space"]
conv_color = hog_params["conv_color"]
orient = hog_params["orient"]
pix_per_cell = hog_params["pix_per_cell"]
cell_per_block = hog_params["cell_per_block"]
hog_channel = hog_params["hog_channel"]
spatial_size = hog_params["spatial_size"]
hist_bins = hog_params["hist_bins"]
spatial_feat = hog_params["spatial_feat"]
hist_feat = hog_params["hist_feat"]
hog_feat = hog_params["hog_feat"]

def find_cars(img, scale, ystart, ystop):
    heatmap = np.zeros_like(img[: ,: ,0])
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv_color)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1 ] /scale), (np.int(imshape[0 ] /scale))))

    ch1 = ctrans_tosearch[: ,: ,0]
    ch2 = ctrans_tosearch[: ,: ,1]
    ch3 = ctrans_tosearch[: ,: ,2]

    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1

    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            hog_feat1 = hog1[ypos:ypos +nblocks_per_window, xpos:xpos +nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos +nblocks_per_window, xpos:xpos +nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos +nblocks_per_window, xpos:xpos +nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop +window, xleft:xleft +window], (64, 64))

            # Get features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)

            # Scale extracted features to be fed to classifier
            test_features = X_scaler.transform(features)
            # Predict using your classifier
            test_prediction = svc.predict(test_features)

            test_decision = svc.decision_function(test_features)

            if test_prediction == 1 and test_decision > 0.6:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                heatmap[ytop_draw +ystart:ytop_draw +win_draw +ystart, xbox_left:xbox_left +win_draw] += 1
                bboxes.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    return bboxes


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


class Vehicle:
    def __init__(self):
        self.bboxes = []
        # Accumulation length
        self.bboxFilter = 8
        self.failedDetectCount = 0
        # Clear if threshold reached
        self.failedDetectThresh = 2

    def update_pos(self, bbox):
        if bbox == None:
            self.failedDetectCount += 1
            if self.failedDetectCount > self.failedDetectThresh:
                self.bboxes = []
        else:
            self.failedDetectCount = 0
            # Remove accumulated boxes if current position is much different
            if len(self.bboxes):
                if (abs(bbox[0][0] - np.mean(self.bboxes, axis=0).astype(int)[0][0])) > 55 or \
                        (abs(bbox[1][0] - np.mean(self.bboxes, axis=0).astype(int)[1][0]) > 55):
                    self.bboxes = []

            self.bboxes.append(bbox)
            # Remove the oldest accumulated if more than configured filter
            if len(self.bboxes) > self.bboxFilter:
                self.bboxes = self.bboxes[1:]

    def get_bbox(self):
        if self.bboxes != []:
            # smooth bbox
            bbox = np.mean(self.bboxes, axis=0).astype(int)
            return True, bbox
        else:
            return False, None


# Class for vehicle detection
class VehicleDetector:
    def __init__(self):
        # Threshold to keep bounding boxes from previous frames
        self.smooth_threshold = 8
        # Min heat threshold
        self.heat_threshold = 6
        self.smooth_bboxes = []
        self.vehicles = []
        # scales and respective areas to search for in the image
        self.scalemap = {1.27: (388, 520), 1.28: (388, 520), 1.29: (388, 520), 1.75: (400, 656), 1.90: (400, 656),
                         2: (400, 656)}

    def process_frame(self, img):
        bboxes = []
        for scale in self.scalemap:
            ystart, ystop = self.scalemap[scale]
            bboxes_scale = find_cars(img, scale, ystart, ystop)
            if bboxes_scale != None:
                bboxes.extend(bboxes_scale)

        all_box_img = draw_boxes(img, bboxes)

        heat = np.zeros_like(img[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat, bboxes)

        # Apply threshold to help remove false positives from current frame
        heat = apply_threshold(heat, self.heat_threshold)

        # Add to accumulated bboxes from previous frames
        self.smooth_bboxes.append(heat)
        if (len(self.smooth_bboxes) > self.smooth_threshold):
            self.smooth_bboxes = self.smooth_bboxes[1:]

        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        for i in range(len(self.smooth_bboxes)):
            heat = heat + self.smooth_bboxes[i]

        # Apply threshold to help remove false positives after smoothing over multiple frames
        heat = apply_threshold(heat, self.heat_threshold * 2)

        # Visualize the heatmap when displaying
        heat = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heat)

        # Update/Create the vechicles detected
        self.update_vehicles(labels)

    # Accumulate bounding boxes to update tracked vehicles
    def update_vehicles(self, labels):
        # Iterate through all detected vehicles
        for vehicle_number in range(1, max(len(self.vehicles), labels[1]) + 1):
            # Find pixels with each vehicle_number label value
            nonzero = (labels[0] == vehicle_number).nonzero()

            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # Check if previous vehicle not found
            if len(nonzerox):
                # Define a bounding box based on min/max x and y
                bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            else:
                bbox = None

            # create a new vehicle object if a new vechicle detected
            if len(self.vehicles) < vehicle_number:
                self.vehicles.append(Vehicle())
            # update the vehicle bbox
            self.vehicles[vehicle_number - 1].update_pos(bbox)

    # Smooth the vehicle bounding boxes
    def smooth_vehicle_boxes(self, img):
        bboxes = []
        for vehicle in self.vehicles:
            ret, bbox = vehicle.get_bbox()
            if ret:
                # Draw the box on the image
                bboxes.append(bbox)

        heat = np.zeros_like(img[:, :, 0]).astype(np.float)

        # Again add heat to each box in box list
        heat = add_heat(heat, bboxes)
        heat = np.clip(heat, 0, 255)
        # Track smoothed boxes from the heat
        labels = label(heat)
        # Draw the labelled boxes
        return self.draw_labeled_bboxes(np.copy(img), labels)

    # Draw the labelled boxes
    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for vehicle_number in range(1, max(len(self.vehicles), labels[1]) + 1):

            # Find pixels with each car_number label value
            nonzero = (labels[0] == vehicle_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            if len(nonzerox):
                # print('vehicle_number', vehicle_number, (max(nonzerox) - min(nonzerox)), (max(nonzeroy) - min(nonzeroy)))
                # Remove the small bounding boxes
                if (max(nonzerox) - min(nonzerox)) >= 40 and \
                        (max(nonzeroy) - min(nonzeroy)) >= 40:

                    # Define a bounding box based on min/max x and y
                    bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

                    # Draw the box on the image
                    cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
                else:
                    bbox = None

            else:
                bbox = None

            # Add the final drawn box to the accumulated respective vehicle
            self.vehicles[vehicle_number - 1].update_pos(bbox)

        # Return the image
        return img


vehicle_detector = VehicleDetector()
def vehicle_detect(img):
    vehicle_detector.process_frame(np.copy(img))
    return vehicle_detector.smooth_vehicle_boxes(img)

video_file = 'project_video.mp4'
track_output = 'tracked_' + video_file
clip = VideoFileClip(video_file)#.subclip(7)
video_clip = clip.fl_image(vehicle_detect)
video_clip.write_videofile(track_output, audio=False)