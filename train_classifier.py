from random import shuffle
import glob
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import time
from utils import *

cars = []
notcars = []

extra_notcars_images = glob.glob('dataset/non-vehicles/Extras/*.png')
for image in extra_notcars_images:
    notcars.append(image)

gti_notcars_images = glob.glob('dataset/non-vehicles/GTI/*.png')
for image in gti_notcars_images:
    notcars.append(image)

gti_far_cars_images = glob.glob('dataset/vehicles/GTI_Far/*.png')
for image in gti_far_cars_images:
    cars.append(image)

gti_left_cars_images = glob.glob('dataset/vehicles/GTI_Left/*.png')
for image in gti_left_cars_images:
    cars.append(image)

gti_middleclose_cars_images = glob.glob('dataset/vehicles/GTI_MiddleClose/*.png')
for image in gti_middleclose_cars_images:
    cars.append(image)

gti_right_cars_images = glob.glob('dataset/vehicles/GTI_Right/*.png')
for image in gti_right_cars_images:
    cars.append(image)

kitti_cars_images = glob.glob('dataset/vehicles/KITTI_extracted/*.png')
for image in kitti_cars_images:
    cars.append(image)

sample_size = 3000
shuffle(cars)
shuffle(notcars)
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

data_info = data_look(cars, notcars)

print('Your function returned a count of',
      data_info["n_cars"], ' cars and',
      data_info["n_notcars"], ' non-cars')
print('of size: ', data_info["image_shape"], ' and data type:',
      data_info["data_type"])


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

car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

svm_model = {}
svm_model["svc"] = svc
svm_model["X_scaler"] = X_scaler
pickle.dump(svm_model, open('svm_model.p', 'wb'))