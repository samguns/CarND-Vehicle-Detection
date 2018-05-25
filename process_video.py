import pickle
import glob
import numpy as np
from utils import *
from vehicle import Vehicle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


svm_model = pickle.load(open("svm_model.p", "rb"))
svc = svm_model["svc"]
X_scaler = svm_model["X_scaler"]


color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off


def find_cars(img, scale, ystart, ystop):
    heatmap = np.zeros_like(img[: ,: ,0])
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch)
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

            if test_prediction == 1 and test_decision > 0.5:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                heatmap[ytop_draw +ystart:ytop_draw +win_draw +ystart, xbox_left:xbox_left +win_draw] += 1

    return heatmap


def process_image(img):
    heat_map_scale_1 = find_cars(img, 1, 350, 500)
    heat_map_scale_2 = find_cars(img, 2, 400, 650)

    heat_map = np.asarray(heat_map_scale_1) + np.asarray(heat_map_scale_2)
    heat = apply_threshold(heat_map, 3)
    labels = label(heat)

    draw_img = get_labeled_bboxes(np.copy(img), labels)
    return draw_img

vehicles = {}
def get_labeled_bboxes(img, labels):
    global vehicles
    #print("Found " + str(labels[1]) + " potential cars")
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        #bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        search_key = (np.min(nonzerox), np.min(nonzeroy))

        if len(vehicles) == 0:
            v = Vehicle()
            v.update_detection(nonzerox, nonzeroy)
            vehicles.update({search_key: v})
        else:
            updated = False
            for k in list(vehicles):
                x_diff = abs(search_key[0] - k[0])
                y_diff = abs(search_key[1] - k[1])
                if x_diff <= 10 and y_diff <= 10:
                    vehicles[k].update_detection(nonzerox, nonzeroy)
                    vehicles[search_key] = vehicles.pop(k)
                    updated = True
                    break

            if updated == False:
                v = Vehicle()
                v.update_detection(nonzerox, nonzeroy)
                vehicles.update({search_key: v})
        #print("car " + str(car_number) + " bbox " + str(bbox))
        # Draw the box on the image
        # cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    for _, vehicle in vehicles.items():
        ret, bbox = vehicle.get_bbox()
        if ret == True:
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    # Return the image
    return img


def process_video():
    video_file = 'test_video.mp4'
    track_output = 'track_' + video_file
    clip = VideoFileClip(video_file)
    track_clip = clip.fl_image(process_image)
    track_clip.write_videofile(track_output, audio=False)#, verbose=True, progress_bar=False)
    return


process_video()
# example_images = glob.glob('test_video_images/*.jpg')
# for img_src in example_images:
#     img = mpimg.imread(img_src)
#
#     out_img, heat_map = find_cars(img, scale)