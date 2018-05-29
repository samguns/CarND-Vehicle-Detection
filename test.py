import pickle
import numpy as np
from utils import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from collections import OrderedDict
from bisect import bisect_left
from vehicle import Vehicle


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
                #bboxes.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    return heatmap


scaled_regions = [(1.2, 380, 520), (1.5, 400, 600), (2, 400, 660)]
vehicles = OrderedDict()
heat_stack = []
frames = 0
def process_image(img):
    global heat_stack
    global frames
    print("frame", frames)
    heat_map = np.zeros_like(img[:, :, 0])
    for scaled_region in scaled_regions:
        found_heat_map = find_cars(img, scaled_region[0], scaled_region[1], scaled_region[2])
        thresholded_heat = apply_threshold(found_heat_map, 3)
        heat_map += np.asarray(thresholded_heat)

    mpimg.imsave("frames/heat"+str(frames)+".jpg", heat_map)

    heat_stack.append(heat_map)
    heat_stack = heat_stack[-10:]

    mean_heats = np.mean(heat_stack, axis=0).astype(int)
    mean_heats = apply_threshold(mean_heats, 1)
    mpimg.imsave("frames/mean_heat" + str(frames) + ".jpg", mean_heats)
    heat = np.clip(mean_heats, 0, 255)
    labels = label(heat)
    frames += 1

    draw_img = get_labeled_bboxes(np.copy(img), labels)
    return draw_img


def get_labeled_bboxes(img, labels):
    global vehicles
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        search_key = (np.min(nonzerox), np.min(nonzeroy))

        if len(vehicles) == 0:
            v = Vehicle()
            v.new_detection(nonzerox, nonzeroy)
            vehicles.update({search_key: v})
        else:
            updated = False
            if search_key in vehicles:
                vehicles[search_key].update_detection(nonzerox, nonzeroy)
                updated = True
            else:
                index = bisect_left(list(vehicles.keys()), search_key)
                k = []
                if 0 < index < len(vehicles):
                    k_1 = list(vehicles.keys())[index-1]
                    k_2 = list(vehicles.keys())[index]
                    diff_1 = list(abs(np.asarray(k_1) - np.asarray(search_key)))
                    diff_2 = list(abs(np.asarray(k_2) - np.asarray(search_key)))
                    if diff_1 < diff_2:
                        k = k_1
                    else:
                        k = k_2
                elif index == len(vehicles):
                    k = list(vehicles.keys())[index-1]
                else:
                    k = list(vehicles.keys())[0]

                x_diff = abs(search_key[0] - k[0])
                y_diff = abs(search_key[1] - k[1])
                if x_diff <= 70 and y_diff <= 70:
                    vehicles[k].update_detection(nonzerox, nonzeroy)

                    if x_diff != 0 or y_diff != 0:
                        vehicles[search_key] = vehicles.pop(k)
                        vehicles = OrderedDict(sorted(vehicles.items()))
                    updated = True

            if updated is False:
                v = Vehicle()
                v.new_detection(nonzerox, nonzeroy)
                vehicles.update({search_key: v})
                vehicles = OrderedDict(sorted(vehicles.items()))

    for _, vehicle in vehicles.items():
        ret, bbox = vehicle.get_bbox()
        if ret is True:
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    # Return the image
    return img


def process_video():
    video_file = 'test_video.mp4'
    track_output = 'tracking_' + video_file
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