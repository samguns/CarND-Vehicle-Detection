**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog_spacial_features.png
[image3]: ./output_images/scaled_sliding_window_search.png
[image4]: ./output_images/bboxes_and_heat.png
[video1]: ./traced_project_video.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in function *get_hog_features* of `utils.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images, randomly shuffled them and selected 2,500 from each set for SVM training. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and their respective `(32, 32)` binned spatial features, along with HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and ended up with HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` on all three channels of YCrCb.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a combined HOG features described above, `(32, 32)` binned spatial features and histograms of three YCrCb channels.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used a scale of 1.0 to search regions `(380, 520)` in Y axis, a scale of 1.5 to search `(400, 600)` and a 2.0 for `(400, 660)` in Y axis. I found the 1.2 and 1.5 scaled searching generally identified the car correctly, but the resulted bounding boxes were a bit small. So I tried some larger scaled searching. They not only returned a larger shape, but more false positives as well. Thus, I ended up using one 2.0 scaled searching. I wrapped up this scaled sliding window search in function `find_cars` through line 27 to line 90. The searching window size is 64 (8x8) at a step of 2 pixels, which has a 75% overlapping.

After combining these three scaled searching, I got an acceptable results. Here are some example images:

![alt text][image3]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the detected heatmap in each frame of the video. Then I accumulate current heatmap with previous 9 heats and take the mean value of the accumulated heatmap. I threshold the mean heatmap by 1 to eliminate all false positives. Because false positives appear intermittently and almost can't accumulate to 10 in a certain region. From the positive detections used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are 10 frames and their corresponding heatmaps:

![alt text][image4]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As the video shows, my detector losses target when the car is smaller (far away from the camera). Another problem is the detected bounding boxes are sometimes too thin-shaped, I have to supplement it with a manual correction to make the box reflecting a real shape. I think a more accurate model might help in improving these two problems. I'd like to try training the SVM model with more data. Another approach is to try some complex models such as AlexNet, ResNet, to replace this HOG+SVM implementation.

Reflecting on this project, the processing speed is so slow, it made me frustrated every time when I tried my pipeline for the video. If I were to pursue this project further, I'd definitely try those state-of-art approaches such as YOLO, SSD.
