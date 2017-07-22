## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! The code for this project can be found in the IPython notebook `vehicle-detection.ipynb`

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

As explained in the lesson, I uese the hog function from skimage.feature which takes a number of orientations, pixels per cell and how many cells each block should contain. The code for this step is contained in the 5th code cell of the IPython notebook in the function `get_hog_features`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

To achieve color invariance it is adviced not to use the `RGB` color channel. Cars can have different colors which will result in many different clusters in the `RGB` channel. To avoid this I tested other color spaces as well (`HSV`, `HLS`, `LUV`, `YCrCb`). `LUV` and `YCrCb` resulted in the best test accuracy. In the lessons `YCrCb` was suggested which is another reason I chose this color space.

From the original HOG paper it was suggested to use orientations up to 9 to improve the classification. A higher number yields more features but comes with the cost of increased computation time. Because time is not really and issue in this project (but in the real world), I choose the suggested 9 orientations.

As suggested in the lessons, I set `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. These numbers are multiples of the chosen window size of 64 and lead to a nice result.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

As described above I tried various combinations of parameters and finally used the following parameters (refer to cell 18):

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations 9 are suggested by the original paper
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL" using all color channels to get more features

I use `YCrCb` as suggested in the lessons and it resulted in high classification accuracy. 9 orientations were suggested by the original paper and increase the size of the feature vector.
I did not play around with the following parameters too much `pixels_per_cell=(8, 8)` `cells_per_block=(2, 2)` because they are multiples of the chosen window size 64.
Using all color channels to detect hog features results in a larger feature vector and helped to improve the accuracy.



#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

For training refer to code cell 19.

First I split up data into randomized training and test sets. Then I trained a linear SVM using sklearn's `svc = LinearSVC()` and the following method for training `svc.fit(X_train, y_train)` with the randomized data and corresponding labels, see cell 19. This classifier was suggested in the lesson to start with and is suited for the given data set. To tune its parameters, grid search should be used to optimize the SVM C parameter, which is used to specify misclassification vs margin (distance of the seperating hyperplane) to the training samples:
http://scikit-learn.org/stable/modules/grid_search.html

The mentioned parameters above gave the following results:
Using: 9 orientations 8 pixels per cell and 2 cells per block 32 histogram bins, and (32, 32) spatial sampling
Feature vector length: 8460
8.87 Seconds to train SVC...
Test Accuracy of SVC =  0.9904

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions in a region at the bottom of the image `y=(ystart = 400
ystop = 656)` where cars are expected and at two different scales (1 and 1.5). The scaling was done by resizing the whole image instead of the windows (see function `find_cars` in code cell 23). I applied an overlapping of 87.5% for the windows with the parameters `pix_per_cell = 8`, `cells_per_step = 1` and scale of 1. This resulted in more accurate detection but lead to higher computation required due to the small step size. Chosing a step size of 2 would lead to an overlap of 75% but this decreased the detection accuracy.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. This results in 294 features, which is quite large but provided a nice result. To search for cars/features on different scales I ran the find_cars function twice and finally added the resulting heat maps. Instead, I altered `find_cars` to extract features on the two mentioned scales, see cell 23 . Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and added heat to this map (`add_heat`) if multiple bounding boxes were found using `find_cars` and then stored the heatmaps for `n_frames=6` in a list, which I then summed. I thresholded the resulting heatmap using a threshold of 20. I then used that thresholded map to identify vehicle positions, which was done using `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

First I extracted the features (HOG, spatial binning, color histograms, color conversion) on a labeled training set of images and train a Linear SVM classifier. Using two different scales of the image a sliding window was used to find the vehicles in video frames with the trained classifier. This resulted in a heat map to which I added heat corresponding to the found bounding boxes and then stored this heatmap into a list. This list stores only the last six frames which I summed to generate a heatmap that was thresholded to avoid false positive detections (code cell 25). To distinguish the cars inside the heat map, I used the scipy `label` function inside `process_image` (see cell 27). The min/max x and y values of these labeled heat "islands" were used to draw blue bounding boxes using `cv2.rectangel`

To optimize the runtime the function find_cars should be altered to account for the different scales instead of calling the cost intensive function twice (containing two for loops).

To avoid flickering bounding boxes, I implemented the described heat map averaging method over consecutive frames. Furthermore, the locations (x, y, width, height) of the bounding boxes should be averaged over time by using a class for each found vehicle.

Sometimes cars on the other side of the road are detected. This could be avoided by considering changing region of interest, especially in curves. A simple way to do this would be to limit the xmin and xmax search region. However, this could lead to degraded detections in some situations, e.g. curves.
